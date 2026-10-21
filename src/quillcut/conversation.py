"""Conversation understanding: words -> sentences -> turns -> Q&A units -> complete, ranked clip candidates.

Everything here is pure logic over a timestamped transcript, so it is deterministic and testable.

Rules that drive the design:
  * A clip starts at a sentence start and ends at a finished sentence. Never mid-thought.
  * The requested duration is a target, not permission to cut an answer off. A clip is extended to the
    end of the thought, up to a hard maximum; strict mode refuses instead of cutting.
  * A question and its answer form one unit. Partial answers are only used when the answer is longer
    than the hard maximum.
  * Candidates are ranked only after they pass the completeness gates.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------- vocabulary

TERMINAL = re.compile(r"[.!?…]['\")\]]*$")
ABBREV = re.compile(r"^(mr|mrs|ms|dr|prof|sr|jr|vs|etc|st|e\.g|i\.e|u\.s|no)\.$", re.I)
QUESTION_STARTS = {"what", "why", "how", "when", "where", "who", "whom", "whose", "which", "do", "does", "did",
                   "can", "could", "would", "will", "should", "is", "are", "was", "were", "have", "has",
                   "tell", "explain", "describe", "walk"}
REQUEST_STARTS = {"tell", "explain", "describe", "walk", "talk", "share", "give", "show", "take"}
# Words that cannot end a finished sentence. Words that can ("for", "you", "it", "that", "in") are left out on
# purpose: "…nobody asked for." and "…will not save you." are complete, and the transcript's own punctuation
# is trusted for everything not in this list.
DANGLING_END = {"and", "but", "or", "so", "because", "which", "whose", "to", "of", "the", "a", "an", "with",
                "as", "if", "when", "than", "my", "your", "our", "their", "his", "her", "its", "this", "these",
                "those", "also", "then", "although", "while", "whether"}
CONTEXT_START = {"it", "that", "this", "they", "those", "these", "he", "she", "them", "so", "and", "but", "because",
                 "also", "then", "which", "however", "anyway", "yeah", "yes", "no", "right", "well", "exactly",
                 "which", "such", "both", "either", "another", "otherwise", "therefore", "plus", "or"}
CONTINUATION_START = {"and", "but", "so", "because", "which", "also", "plus", "or", "then", "however", "yet",
                      "although", "though", "while", "instead", "otherwise", "therefore"}
CONTRAST = {"never", "actually", "mistake", "wrong", "secret", "truth", "nobody", "everyone", "biggest", "first",
            "only", "worst", "best", "myth", "lie", "stop", "realized", "changed", "surprised", "problem",
            "struggled", "failed", "learned", "why", "how"}
STOP = set("""a about above after again all also am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just let me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself yourselves yeah okay really gonna got get one going know right think thing
things like kind sort mean actually basically just um uh""".split())

MIN_ABSOLUTE_SECONDS = 12.0      # shorter than this is not a reel, whatever the target
LEAD_PAD = 0.12                  # seconds of room before the first word
TAIL_PAD = 0.35                  # seconds of natural tail after the last word


# --------------------------------------------------------------------------------------------- data


@dataclass
class Sentence:
    idx: int
    start: float
    end: float
    text: str
    words: list[dict]
    gap_before: float = 0.0
    question: bool = False
    complete: bool = True
    speaker: str | None = None

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Unit:
    """One meaningful conversational unit: an optional question plus the paragraphs of its answer."""
    idx: int
    kind: str                                   # "qa" | "monologue"
    question: int | None                        # sentence index of the question, if any
    paragraphs: list[tuple[int, int]]           # (first, last) sentence index, inclusive
    start: float = 0.0
    end: float = 0.0

    @property
    def first(self) -> int:
        return self.paragraphs[0][0]

    @property
    def last(self) -> int:
        return self.paragraphs[-1][1]


@dataclass
class Candidate:
    unit: int
    first: int                                  # first sentence index in the clip (the question when included)
    last: int
    start: float                                # seconds, after padding
    end: float
    question_included: bool
    partial: bool                               # stops before the end of a Q&A answer (answer longer than max)
    reasons: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    score: float = 0.0
    text: str = ""
    hook_idx: int | None = None
    topic: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Selection:
    reel: int
    target: float
    candidate: Candidate | None
    actual: float = 0.0
    extended: bool = False
    note: str = ""


# --------------------------------------------------------------------------------------------- text helpers


def clean_word(text: str) -> str:
    return re.sub(r"[^a-z0-9']", "", text.lower())


def content_words(text: str) -> list[str]:
    out = []
    for token in re.findall(r"[A-Za-z][A-Za-z']+|\d+(?:[.,]\d+)?", text):
        w = token.lower().strip("'")
        if w in STOP or len(w) < 3:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na, nb = math.sqrt(sum(v * v for v in a.values())), math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def bag(sentences: list[Sentence]) -> dict:
    out: dict = {}
    for s in sentences:
        for w in content_words(s.text):
            out[w] = out.get(w, 0) + 1
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# --------------------------------------------------------------------------------------------- sentences


def _is_question(text: str, n_words: int) -> bool:
    t = text.strip()
    if t.endswith("?"):
        return True
    head = [clean_word(w) for w in t.split()[:5]]
    if head and head[0] in REQUEST_STARTS and n_words <= 24 and any(w in ("us", "me", "you", "your") for w in head):
        return True                          # "Tell us about yourself." is a question in everything but punctuation
    if TERMINAL.search(t):
        return False                         # transcript punctuation is trusted when present
    first = clean_word(t.split()[0]) if t.split() else ""
    return first in QUESTION_STARTS and n_words <= 30


def _is_complete(text: str, n_words: int) -> bool:
    t = text.strip()
    if not TERMINAL.search(t):
        return False
    last = clean_word(t.split()[-1]) if t.split() else ""
    if n_words >= 3 and last in DANGLING_END and not t.endswith("?"):
        return False
    return n_words >= 1


def split_sentences(words: list[dict], pause: float = 0.75) -> list[Sentence]:
    """Group transcript words into sentences. Punctuation first, long pauses second."""
    raw: list[list[dict]] = []
    cur: list[dict] = []
    clean = [w for w in words if str(w.get("word", "")).strip()]
    for i, w in enumerate(clean):
        cur.append(w)
        txt = w["word"].strip()
        nxt = clean[i + 1] if i + 1 < len(clean) else None
        gap = (nxt["start"] - w["end"]) if nxt else math.inf
        terminal = bool(TERMINAL.search(txt)) and not ABBREV.match(txt)
        if terminal or gap >= pause:
            raw.append(cur)
            cur = []
    if cur:
        raw.append(cur)

    sentences: list[Sentence] = []
    for group in raw:
        text = " ".join(w["word"].strip() for w in group)
        sentences.append(Sentence(idx=0, start=group[0]["start"], end=group[-1]["end"], text=text, words=group))

    # a pause-split fragment that never finished its thought belongs with what follows it
    merged: list[Sentence] = []
    for s in sentences:
        if merged and not TERMINAL.search(merged[-1].text) and s.start - merged[-1].end < 1.2:
            prev = merged.pop()
            words_ = prev.words + s.words
            s = Sentence(idx=0, start=prev.start, end=s.end, text=prev.text + " " + s.text, words=words_)
        merged.append(s)

    prev_end = None
    for i, s in enumerate(merged):
        s.idx = i
        s.gap_before = 0.0 if prev_end is None else max(0.0, s.start - prev_end)
        s.question = _is_question(s.text, s.n_words)
        s.complete = _is_complete(s.text, s.n_words)
        prev_end = s.end
    return merged


# --------------------------------------------------------------------------------------------- units


def cohesion_boundaries(sentences: list[Sentence], lo: int, hi: int, window: int = 3,
                        min_seconds: float = 12.0) -> list[int]:
    """Sentence indices (in lo+1..hi) where a new paragraph starts, from the dip in lexical overlap."""
    n = hi - lo + 1
    if n < 4:
        return []
    sims: dict[int, float] = {}
    for i in range(lo + 1, hi + 1):
        left = bag(sentences[max(lo, i - window):i])
        right = bag(sentences[i:min(hi + 1, i + window)])
        sims[i] = cosine(left, right)
    values = sorted(sims.values())
    median = values[len(values) // 2]
    cuts: list[int] = []
    for i in range(lo + 1, hi + 1):
        s = sims[i]
        local_min = all(s <= sims.get(j, 1.0) for j in range(i - 2, i + 3) if j != i)
        gap = sentences[i].gap_before
        if (local_min and s < 0.10 and s <= 0.6 * median + 1e-9) or (gap >= 1.4 and s < 0.25):
            cuts.append(i)
    # enforce a minimum paragraph length
    kept: list[int] = []
    start_t = sentences[lo].start
    for c in cuts:
        if sentences[c].start - start_t >= min_seconds and sentences[hi].end - sentences[c].start >= min_seconds / 2:
            kept.append(c)
            start_t = sentences[c].start
    return kept


def build_units(sentences: list[Sentence]) -> list[Unit]:
    """Cut the conversation into Q&A units and monologue units, then into paragraphs inside each."""
    if not sentences:
        return []
    cuts = [0]
    unit_has_content = False
    last_content_speaker: str | None = None
    for i in range(1, len(sentences)):
        s, p = sentences[i], sentences[i - 1]
        cut = False
        if s.question and not (p.question and s.gap_before < 0.9):
            cut = True                                   # a new question opens a new unit
        elif s.gap_before >= 2.2:
            cut = True
        elif (s.speaker and p.speaker and s.speaker != p.speaker and unit_has_content
              and last_content_speaker is not None and s.speaker != last_content_speaker):
            cut = True                                   # the answerer stopped; someone else is speaking
        if cut:
            cuts.append(i)
            unit_has_content = False
            last_content_speaker = None
        if not s.question:
            unit_has_content = True
            last_content_speaker = last_content_speaker or s.speaker
    cuts.append(len(sentences))

    units: list[Unit] = []
    for a, b in zip(cuts, cuts[1:]):
        lo, hi = a, b - 1
        question = lo if sentences[lo].question else None
        # consecutive question sentences are one question
        body_lo = lo
        while body_lo <= hi and sentences[body_lo].question:
            body_lo += 1
        if question is not None and body_lo > hi:
            kind, paras = "question_only", []
        else:
            kind = "qa" if question is not None else "monologue"
            first = body_lo if question is not None else lo
            cuts_in = cohesion_boundaries(sentences, first, hi)
            edges = [first, *cuts_in, hi + 1]
            paras = [(x, y - 1) for x, y in zip(edges, edges[1:])]
        if not paras:
            continue
        unit = Unit(idx=len(units), kind=kind, question=question if kind == "qa" else None, paragraphs=paras)
        unit.start = sentences[unit.question if unit.question is not None else unit.first].start
        unit.end = sentences[unit.last].end
        units.append(unit)
    return units


# --------------------------------------------------------------------------------------------- scoring

# Hook archetypes. They describe what kind of opening a line already is; they never generate wording.
ARCHETYPES = {
    "loss_avoidance": ("stop", "never", "mistake", "mistakes", "avoid", "don't", "dont", "waste", "wasted",
                       "costly", "lose", "losing", "failed", "regret"),
    "information_asymmetry": ("secret", "nobody", "no one", "truth", "hidden", "most people", "rarely",
                              "don't tell", "never tell", "what they"),
    "status_optimization": ("best", "top", "master", "optimize", "optimise", "faster", "10x", "system",
                            "habit", "habits", "productive", "leverage", "elite", "level up", "outperform"),
    "contrarian": ("actually", "myth", "opposite", "everything you know", "contrary", "overrated", "backwards",
                   "not what", "wrong"),
}


def classify_archetype(text: str) -> str | None:
    """Which of the four hook archetypes a line already is, from the speaker's own words (None if none)."""
    low = " " + re.sub(r"[^a-z0-9' ]", " ", text.lower()) + " "
    best, best_n = None, 0
    for name, cues in ARCHETYPES.items():
        n = sum(1 for cue in cues if f" {cue} " in low)
        if n > best_n:
            best, best_n = name, n
    return best


def hook_strength(s: Sentence) -> float:
    """0-1 estimate of how well one sentence opens a reel, from surface features only."""
    words = [clean_word(w["word"]) for w in s.words]
    score = 0.0
    if s.question:
        score += 0.25
    if any(re.search(r"\d", w) for w in words):
        score += 0.15
    if any(w in CONTRAST for w in words):
        score += 0.20
    if 6 <= s.n_words <= 16:
        score += 0.20
    elif s.n_words < 4 or s.n_words > 28:
        score -= 0.15
    if any(w in ("i", "my", "we", "our") for w in words[:3]):
        score += 0.10
    if words and words[0] in CONTEXT_START:
        score -= 0.30
    if s.complete:
        score += 0.10
    if classify_archetype(s.text):
        score += 0.08
    return max(0.0, min(1.0, score + 0.15))


def _density(sentences: list[Sentence]) -> float:
    dur = max(1.0, sentences[-1].end - sentences[0].start)
    cw = sum(len(content_words(s.text)) for s in sentences)
    digits = sum(1 for s in sentences for w in s.words if re.search(r"\d", w["word"]))
    return max(0.0, min(1.0, (cw / dur) / 1.6 + 0.05 * digits))


def _clarity(sentences: list[Sentence]) -> float:
    probs = [w.get("probability") for s in sentences for w in s.words if w.get("probability") is not None]
    return float(sum(probs) / len(probs)) if probs else 0.8


def pads(sentences: list[Sentence], first: int, last: int, media_end: float) -> tuple[float, float]:
    """Clip start/end in seconds: a little room before the first word and a natural tail after the last."""
    f, l = sentences[first], sentences[last]
    start = f.start - LEAD_PAD
    if first > 0:
        start = max(start, sentences[first - 1].end + 0.04)
    end = l.end + TAIL_PAD
    if last + 1 < len(sentences):
        end = min(end, sentences[last + 1].start - 0.04)
    return round(max(0.0, start), 3), round(min(media_end, max(end, l.end + 0.05)), 3)


# --------------------------------------------------------------------------------------------- candidates


def hard_max_for(target: float, explicit: float | None = None) -> float:
    if explicit:
        return max(explicit, target)
    return min(180.0, max(target * 1.75, target + 30.0))


def make_candidates(sentences: list[Sentence], units: list[Unit], *, media_end: float, hard_max: float,
                    include_question: str = "auto", min_seconds: float = MIN_ABSOLUTE_SECONDS) -> list[Candidate]:
    """Every complete, self-contained window made of whole paragraphs, plus whole units."""
    out: list[Candidate] = []
    for u in units:
        q = u.question
        q_len = (sentences[q].end - sentences[q].start) if q is not None else 0.0
        want_q = q is not None and (include_question == "yes" or (include_question == "auto" and q_len <= 15.0))
        n_par = len(u.paragraphs)
        for i in range(n_par):
            for j in range(i, n_par):
                a, b = u.paragraphs[i][0], u.paragraphs[j][1]
                first_sentence = sentences[a]
                starts_unit = i == 0
                include_q = want_q and starts_unit
                reasons: list[str] = []
                dependent = bool(clean_word(first_sentence.text.split()[0]) in CONTEXT_START) if first_sentence.text else False
                if dependent and not (starts_unit and q is not None and include_q):
                    if not starts_unit:
                        continue                    # would open mid-conversation on "so", "it", "that"
                    if q is not None and not include_q:
                        reasons.append("answer opens on a reference; the question is shown as text")
                last_s = sentences[b]
                if not last_s.complete:
                    continue                        # ends mid-sentence or on a dangling word
                nxt = sentences[b + 1] if b + 1 < len(sentences) else None
                reaches_unit_end = j == n_par - 1
                if (not reaches_unit_end and nxt is not None and nxt.words
                        and clean_word(nxt.text.split()[0]) in CONTINUATION_START and nxt.gap_before < 1.0):
                    continue                        # the next sentence continues this one
                first_idx = q if include_q else a
                start, end = pads(sentences, first_idx, b, media_end)
                dur = end - start
                if dur < min_seconds or dur > hard_max:
                    continue
                partial = u.kind == "qa" and not reaches_unit_end
                if partial:
                    reasons.append("answer is longer than the maximum, ends at a paragraph boundary")
                if i > 0 or not reaches_unit_end:
                    reasons.append("self-contained paragraph run")
                out.append(Candidate(unit=u.idx, first=first_idx, last=b, start=start, end=end,
                                     question_included=include_q, partial=partial, reasons=reasons,
                                     text=" ".join(s.text for s in sentences[first_idx:b + 1])))
    return out


def score_candidate(c: Candidate, sentences: list[Sentence], units: list[Unit]) -> Candidate:
    body = sentences[c.first:c.last + 1]
    answer = [s for s in body if not (s.question and s.idx == c.first and c.question_included)] or body
    # best opening sentence: the first answer sentence or the question
    openers = [s for s in body[:3]]
    hook = max(openers, key=hook_strength)
    c.hook_idx = hook.idx
    u = units[c.unit]
    completeness = 1.0
    if c.partial:
        completeness = 0.65
    elif u.kind == "qa" and c.first == (u.question if u.question is not None else -1):
        completeness = 1.0
    elif u.kind == "qa" and not c.question_included:
        completeness = 0.9
    c.scores = {
        "completeness": round(completeness, 3),
        "hook": round(hook_strength(hook), 3),
        "density": round(_density(answer), 3),
        "clarity": round(_clarity(body), 3),
        "self_contained": 1.0 if (c.question_included or not content_starts_dependent(body[0])) else 0.5,
    }
    return c


def content_starts_dependent(s: Sentence) -> bool:
    return bool(s.text) and clean_word(s.text.split()[0]) in CONTEXT_START


def fit_score(duration: float, target: float, hard_max: float) -> float:
    if duration <= target:
        # steep enough that a strong opening line alone can't carry a clip a quarter of the asked length
        return max(0.0, 1.0 - 1.2 * (target - duration) / max(target, 1.0))
    return max(0.0, 1.0 - 0.7 * (duration - target) / max(hard_max - target, 1.0))


BASE_WEIGHTS = {"completeness": 0.30, "fit": 0.20, "hook": 0.20, "density": 0.15, "clarity": 0.10,
                "self_contained": 0.05}
# What each kind of content rewards. Completeness is a hard gate everywhere; these only shift the ranking.
PROFILES = {
    "educational": {"density": 0.25, "hook": 0.15, "fit": 0.15},
    "storytelling": {"hook": 0.25, "density": 0.10, "self_contained": 0.10},
    "motivational": {"hook": 0.27, "density": 0.08, "self_contained": 0.10},
    "podcast": {"completeness": 0.33, "hook": 0.17, "density": 0.15},
    "interview": {"completeness": 0.36, "hook": 0.17, "fit": 0.14},
    "debate": {"hook": 0.26, "density": 0.17, "fit": 0.15},
    "business": {"density": 0.22, "completeness": 0.32, "hook": 0.14},
}


def total_score(c: Candidate, target: float, hard_max: float, profile: str | None = None) -> float:
    w = {**BASE_WEIGHTS, **PROFILES.get(profile or "", {})}
    norm = sum(w.values())
    s = c.scores
    parts = {"completeness": s["completeness"], "fit": fit_score(c.duration, target, hard_max), "hook": s["hook"],
             "density": s["density"], "clarity": s["clarity"], "self_contained": s["self_contained"]}
    return round(sum(w[k] * parts[k] for k in w) / norm, 4)


# --------------------------------------------------------------------------------------------- selection


def _span_overlap(c: Candidate, rng: tuple[float, float]) -> float:
    return max(0.0, min(c.end, rng[1]) - max(c.start, rng[0]))


def select(candidates: list[Candidate], sentences: list[Sentence], targets: list[float], *,
           strict: bool = False, hard_max: float | None = None, min_score: float = 0.5,
           profile: str | None = None, exclude: list[tuple[float, float]] | None = None,
           prefer: tuple[float, float] | None = None, allow=None) -> list[Selection]:
    """One selection per requested reel (its own target). Distinct, complete, honest about shortfalls.

    exclude: source time ranges already used (or rejected by the user); a candidate overlapping one is skipped.
    prefer: only candidates covering this range (regenerate the same moment with a new length)."""
    chosen: list[Candidate] = []
    results: list[Selection] = []
    for k, target in enumerate(targets, start=1):
        cap = hard_max_for(target, hard_max)
        pool = [c for c in candidates if c.duration <= cap + 1e-6]
        if exclude:
            pool = [c for c in pool if not any(_span_overlap(c, r) > 0.25 * c.duration for r in exclude)]
        if prefer:
            span = prefer[1] - prefer[0]
            pool = [c for c in pool if _span_overlap(c, prefer) >= 0.6 * min(span, c.duration)]
        rules_note = ""
        if allow is not None:                    # the user's rules are hard filters, applied before ranking
            before = len(pool)
            pool = [c for c in pool if allow(c)]
            if before and not pool:
                rules_note = ("No complete moment satisfies your rules, so this reel was not made. "
                              "Nothing was added to fill it.")
        conflict_note = ""
        if strict:
            # strict never trades the end of an answer for the clock: pieces that stop mid-answer don't count
            fits = [c for c in pool if c.duration <= target + 0.5 and not c.partial]
            whole_answers = [c for c in candidates if not c.partial]
            if not fits and whole_answers:
                need = min(whole_answers, key=lambda c: c.duration).duration
                conflict_note = (f"A complete answer needs about {need:.0f}s but strict duration is {target:.0f}s. "
                                 "Turn off strict mode to keep the full answer.")
            pool = fits
        best: tuple[float, Candidate] | None = None
        for c in pool:
            if any(_overlaps(c, o, sentences) for o in chosen):
                continue
            score = total_score(c, target, cap, profile)
            if score < min_score:
                continue
            if best is None or score > best[0]:
                best = (score, c)
        if best is None:
            note = conflict_note or rules_note or ("No further complete, distinct moment is long enough for this "
                                                   "reel. Adding filler would cut a thought or repeat an earlier reel.")
            results.append(Selection(reel=k, target=target, candidate=None, note=note))
            continue
        score, c = best
        c.score = score
        chosen.append(c)
        extended = c.duration > target + 1.0
        note = ""
        if extended:
            note = f"Extended from {target:.0f}s to {c.duration:.0f}s to finish the thought."
        elif c.duration < target - 5.0:
            note = f"Ends at {c.duration:.0f}s because the thought is complete; nothing was added to reach {target:.0f}s."
        results.append(Selection(reel=k, target=target, candidate=c, actual=round(c.duration, 2),
                                 extended=extended, note=note))
    return results


def _overlaps(a: Candidate, b: Candidate, sentences: list[Sentence]) -> bool:
    inter = min(a.end, b.end) - max(a.start, b.start)
    if inter > 0.10 * min(a.duration, b.duration):
        return True
    wa = set(content_words(a.text))
    wb = set(content_words(b.text))
    return jaccard(wa, wb) > 0.5


def analyse(words: list[dict], media_end: float, *, targets: list[float] | None = None, strict: bool = False,
            include_question: str = "auto", hard_max: float | None = None,
            speakers: list[str | None] | None = None, profile: str | None = None,
            exclude: list[tuple[float, float]] | None = None, prefer: tuple[float, float] | None = None,
            allow=None, min_seconds: float = MIN_ABSOLUTE_SECONDS) -> dict:
    """Whole-conversation analysis. Returns sentences, units, ranked candidates and selections."""
    sentences = split_sentences(words)
    if speakers is not None and len(speakers) == len(sentences):
        for s, label in zip(sentences, speakers):
            s.speaker = label
    units = build_units(sentences)
    cap = max(hard_max_for(t, hard_max) for t in (targets or [60.0]))
    candidates = [score_candidate(c, sentences, units)
                  for c in make_candidates(sentences, units, media_end=media_end, hard_max=cap,
                                           include_question=include_question, min_seconds=min_seconds)]
    selections = select(candidates, sentences, targets or [], strict=strict, hard_max=hard_max, profile=profile,
                        exclude=exclude, prefer=prefer, allow=allow)
    return {"sentences": sentences, "units": units, "candidates": candidates, "selections": selections}


def continuation_risk(sentences: list[Sentence], c: Candidate) -> float:
    """0-1: how likely the speaker is still on the same point right after this clip ends.

    Lexical overlap between the clip's last sentences and the next ones, zeroed when the next sentence is
    a new question or follows a long pause (a clear turn boundary)."""
    nxt = sentences[c.last + 1:c.last + 3]
    if not nxt or nxt[0].question or nxt[0].gap_before >= 1.5:
        return 0.0
    tail = sentences[max(c.first, c.last - 1):c.last + 1]
    risk = cosine(bag(tail), bag(nxt))
    if nxt[0].words and clean_word(nxt[0].text.split()[0]) in CONTINUATION_START:
        risk = max(risk, 0.5)
    return round(min(1.0, risk), 3)


def distinct_capacity(candidates: list[Candidate], sentences: list[Sentence], target: float = 45.0) -> int:
    """How many distinct, complete reels this conversation can support (greedy upper estimate)."""
    cap = hard_max_for(target)
    picked: list[Candidate] = []
    for c in sorted((c for c in candidates if c.duration <= cap),
                    key=lambda c: -total_score(c, target, cap)):
        if total_score(c, target, cap) < 0.5:
            continue
        if not any(_overlaps(c, p, sentences) for p in picked):
            picked.append(c)
    return len(picked)

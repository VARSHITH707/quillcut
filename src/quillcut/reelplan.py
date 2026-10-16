"""Selected conversation moment -> EditPlan.

Everything on screen comes from the transcript: the hook is a verbatim line (or the interviewer's question),
callouts are phrases the speaker actually said, and cuts only remove silence or change framing. No words are
moved, rewritten or invented.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import conversation as cv
from .captions import callout_card, cta_card, hook_card, lower_third_card, speech_cards
from .types import AudioPlan, EditPlan, MusicBed, OutputSpec, Segment, SourceRef

ASPECTS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}
INTENSITY = {
    # cut: (shortest, longest) seconds between framing changes; zooms cycle through `zooms`
    "clean": {"cut": (3.0, 4.0), "zooms": [1.0, 1.05], "callouts": 0, "lower_third": False, "cold_open": False,
              "progress": False, "emphasis": 4},
    "dynamic": {"cut": (2.0, 2.5), "zooms": [1.0, 1.10, 1.14], "callouts": 1, "lower_third": True,
                "cold_open": True, "progress": False, "emphasis": 6},
    "high_energy": {"cut": (1.2, 1.8), "zooms": [1.0, 1.12, 1.18], "callouts": 2, "lower_third": True,
                    "cold_open": True, "progress": True, "emphasis": 8},
}
PAUSE_MAX = 0.40          # silences longer than this are tightened
PAUSE_KEEP = 0.18         # what is left of a tightened silence (0.15-0.20 s)
FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "hm", "mm", "mhm"}   # sounds, never meaning
HOOK_SECONDS = 3.0        # the hook lives in the first three seconds
HOOK_FRAME_WORDS = 5      # at most 5 words on screen per hook frame
HOOK_MAX_WORDS = 10       # two frames of up to five words
CAPTION_MAX_WORDS = 5     # 3-5 words per caption card
CAPTION_Y = 0.60          # speech captions sit in the centre third of the frame (35%-65%)
CAPTION_BAND = (0.35, 0.65)
CALLOUT_Y = 0.45          # above the captions, still inside the safe area


@dataclass
class PlanOptions:
    intensity: str = "dynamic"
    aspect: str = "9:16"
    reframe: bool = True
    cold_open: bool = True
    cta: str | None = None
    denoise: bool = True
    music: dict | None = None          # {"file": ..., "licence": ..., "level_db": ...}
    accent: str = "F85830"
    show_question_label: bool = True
    logo: str | None = None            # path to a brand image shown on every frame
    caption_y: float | None = None     # caption centre as a fraction of height; kept inside 0.35-0.65
    hook_style: str | None = None      # from the user's rules: "question" | "cold_open" | "opening"


@dataclass
class Built:
    plan: EditPlan
    hook: dict
    notes: list[str] = field(default_factory=list)


def safe_for(aspect: str, base: dict) -> dict:
    if aspect == "9:16":
        return base
    return {"top": 0.06, "bottom": 0.10, "left": 0.06, "right": 0.06}


def layout_for(ref: SourceRef, out_w: int, out_h: int, max_upscale: float) -> str:
    cw, ch = ref.content_box[0], ref.content_box[1]
    aspect = out_w / out_h
    scale = out_h / ch if cw / ch >= aspect else out_w / cw
    if abs(cw / ch - aspect) < 0.02 or scale <= max_upscale:
        return "center_crop"
    return "wide_fill" if aspect < 1 else "fit_blur"


def trim_pauses(words: list[dict], start: float, end: float) -> list[tuple[float, float]]:
    """Intervals of the range with dead air and filler sounds taken out. Never cuts through a spoken word.

    A silence longer than PAUSE_MAX is tightened to PAUSE_KEEP. A filler ("um", "uh") is removed together
    with the silence around it; only these non-word sounds are ever dropped."""
    cuts: list[tuple[float, float]] = []
    prev: dict | None = None
    pending: list[dict] = []
    for w in words:
        if cv.clean_word(w["word"]) in FILLERS:
            pending.append(w)
            continue
        if prev is not None:
            gap = w["start"] - prev["end"]
            if pending:                                  # cover the filler completely
                c0 = min(prev["end"] + PAUSE_KEEP / 2, pending[0]["start"] - 0.02)
                c1 = max(w["start"] - PAUSE_KEEP / 2, pending[-1]["end"] + 0.02)
                if c1 - c0 > 0.04:
                    cuts.append((max(c0, prev["end"] + 0.01), min(c1, w["start"] - 0.01)))
            elif gap > PAUSE_MAX:
                cuts.append((prev["end"] + PAUSE_KEEP / 2, w["start"] - PAUSE_KEEP / 2))
        pending = []
        prev = w
    out, t = [], start
    for c0, c1 in cuts:
        if c0 > t:
            out.append((round(t, 3), round(c0, 3)))
        t = max(t, c1)
    out.append((round(t, 3), round(end, 3)))
    # an interval is dropped only when it holds no spoken word (pure silence); a short one with a word stays
    spoken = [w for w in words if cv.clean_word(w["word"]) not in FILLERS]
    return [(a, b) for a, b in out
            if b - a >= 0.1 and any(w["start"] >= a - 1e-6 and w["end"] <= b + 1e-6 for w in spoken)]


def split_by_velocity(interval: tuple[float, float], words: list[dict],
                      cut: tuple[float, float]) -> list[tuple[float, float]]:
    """Cut an interval every `cut` seconds so the framing can change. Cuts fall between two words (or in a
    pause), the pieces are contiguous, and the audio runs straight through every join."""
    lo, hi = cut
    a, b = interval
    inside = [w for w in words if w["end"] > a and w["start"] < b]
    bounds = [round((p["end"] + n["start"]) / 2, 3) for p, n in zip(inside, inside[1:])
              if a + 0.15 < (p["end"] + n["start"]) / 2 < b - 0.15 and n["start"] >= p["end"] - 1e-6]
    if b - a <= hi or not bounds:
        return [interval]
    pieces, cur, k = [], a, 0
    cycle = (lo, hi, (lo + hi) / 2)
    while b - cur > hi:
        want = cur + cycle[k % 3]
        k += 1
        usable = [t for t in bounds if t > cur + 0.6 * lo and b - t >= 0.6 * lo]
        if not usable:
            break
        nxt = min(usable, key=lambda t: abs(t - want))
        pieces.append((cur, nxt))
        cur = nxt
    pieces.append((cur, b))
    return pieces


def hook_frames(text: str) -> list[str]:
    """Split a verbatim hook into frames of at most HOOK_FRAME_WORDS words (two frames over the first 3 s).
    A line longer than two frames is shortened and marked with an ellipsis."""
    words = text.rstrip("…").split()
    shortened = text.endswith("…") or len(words) > HOOK_MAX_WORDS
    words = words[:HOOK_MAX_WORDS]
    if len(words) <= HOOK_FRAME_WORDS:
        frames = [words]
    else:
        # split near the middle, preferring a clause boundary
        mid = len(words) // 2
        best = min(range(max(1, len(words) - HOOK_FRAME_WORDS), min(HOOK_FRAME_WORDS, len(words) - 1) + 1),
                   key=lambda k: (not re.search(r"[,;:.]$", words[k - 1]), abs(k - mid)))
        frames = [words[:best], words[best:]]
    out = [" ".join(f).rstrip(",;:") for f in frames if f]
    if shortened and out:
        out[-1] = out[-1].rstrip(".") + "…"
    return out


def key_terms(sentences: list[cv.Sentence], limit: int) -> list[str]:
    counts = Counter(w for s in sentences for w in cv.content_words(s.text) if len(w) > 4)
    return [w for w, n in counts.most_common(limit * 2) if n >= 2][:limit]


def key_phrases(sentences: list[cv.Sentence], limit: int) -> list[tuple[str, str]]:
    """Frequent adjacent content-word pairs as (spoken form, first word) so callouts quote the speaker."""
    pairs: Counter = Counter()
    surface: dict[str, tuple[str, str]] = {}
    for s in sentences:
        toks = [(cv.clean_word(w["word"]), w["word"]) for w in s.words]
        for (a, aw), (b, bw) in zip(toks, toks[1:]):
            if a in cv.STOP or b in cv.STOP or len(a) < 4 or len(b) < 4:
                continue
            key = f"{a} {b}"
            pairs[key] += 1
            surface.setdefault(key, (f"{re.sub(r'[^A-Za-z0-9-]', '', aw)} {re.sub(r'[^A-Za-z0-9-]', '', bw)}", a))
    ranked = [k for k, n in pairs.most_common() if n >= 2]
    return [surface[k] for k in ranked[:limit]]


def choose_hook(sentences: list[cv.Sentence], c: cv.Candidate, allow_cold_open: bool,
                hook_style: str | None = None) -> dict:
    """A truthful hook: a stronger line from later in the answer, the question, or the opening line.

    hook_style (from the user's rules): "question" prefers the question, "opening" forbids a cold open."""
    body = sentences[c.first:c.last + 1]
    answer = [s for s in body if not (s.question and c.question_included and s.idx == c.first)]
    first = answer[0] if answer else body[0]
    question = sentences[c.first] if c.question_included and sentences[c.first].question else None
    cold = None
    if hook_style in ("question", "opening"):
        allow_cold_open = False
    if hook_style == "question" and question is not None and question.n_words > 18:
        question = None
    if allow_cold_open and len(answer) >= 3:
        base = cv.hook_strength(first)
        for s in answer[1:]:
            if s.start - body[0].start < 8.0 or s.duration > 7.0 or not s.complete:
                continue
            if cv.content_starts_dependent(s) or not (6 <= s.n_words <= 18):
                continue
            strength = cv.hook_strength(s)
            if strength >= 0.6 and strength >= base + 0.15 and (cold is None or strength > cv.hook_strength(cold)):
                cold = s
    # The words are always the speaker's. The archetype only labels what kind of hook the line is.
    if cold is not None:
        return {"kind": "cold_open", "sentence": cold, "text": cold.text,
                "archetype": cv.classify_archetype(cold.text)}
    if question is not None and question.n_words <= 18:
        return {"kind": "question", "sentence": question, "text": question.text,
                "archetype": cv.classify_archetype(question.text)}
    return {"kind": "opening", "sentence": first, "text": first.text, "archetype": cv.classify_archetype(first.text)}


def map_words(segments: list[Segment], words: list[dict]) -> list[dict]:
    out, offset = [], 0.0
    for seg in segments:
        for w in words:
            if seg.src_in - 1e-6 <= w["start"] and w["end"] <= seg.src_out + 1e-6:
                out.append({**w, "start": round(offset + (w["start"] - seg.src_in) / seg.speed, 3),
                            "end": round(offset + (w["end"] - seg.src_in) / seg.speed, 3)})
        offset += seg.out_duration
    return out


def speech_spans(words: list[dict], gap: float = 0.45) -> list[tuple[float, float]]:
    spans: list[list[float]] = []
    for w in words:
        if spans and w["start"] - spans[-1][1] <= gap:
            spans[-1][1] = w["end"]
        else:
            spans.append([w["start"], w["end"]])
    return [(round(a, 3), round(b, 3)) for a, b in spans]


def hook_cards(hook: dict, start: float, end: float, style: dict, safe: dict) -> list:
    """The hook as one or two on-screen frames of at most five words each, shown across the first seconds.
    A frame that does not fit the safe area is shortened (and marked with an ellipsis), never overflowed."""
    frames = hook_frames(hook["text"])
    cards, shown = [], []
    span = (end - start) / max(len(frames), 1)
    for i, frame in enumerate(frames):
        words = frame.rstrip("…").split()
        marked = frame.endswith("…")
        for n in range(len(words), 1, -1):
            text = " ".join(words[:n]).rstrip(",;:.") + ("…" if marked or n < len(words) else "")
            try:
                cards.append(hook_card(text, round(start + i * span, 3), round(start + (i + 1) * span, 3), style, safe))
            except ValueError:
                continue
            shown.append(text)
            break
        else:
            raise ValueError(f"hook cannot be fitted on screen: {frame!r}")
    hook["frames"] = shown
    hook["text"] = " ".join(shown)
    return cards


def _first_time(mapped: list[dict], first_word: str, after: float) -> float | None:
    for w in mapped:
        if w["start"] > after and cv.clean_word(w["word"]) == first_word:
            return w["start"]
    return None


def build(c: cv.Candidate, sentences: list[cv.Sentence], ref: SourceRef, cfg: dict, style: dict, *,
          clip_id: str, project_id: str, rights: dict, options: PlanOptions, crop_centers=None) -> Built:
    """Create the EditPlan. `crop_centers(shots) -> list[float | None]` supplies face-follow crop positions."""
    knobs = INTENSITY[options.intensity]
    out_w, out_h = ASPECTS[options.aspect]
    safe = safe_for(options.aspect, cfg["safe_area"])
    st = {**style, "canvas": [out_w, out_h]}
    st["colors"] = {**st["colors"], "accent": options.accent}
    notes: list[str] = []

    body = sentences[c.first:c.last + 1]
    words = [w for s in body for w in s.words]
    layout = layout_for(ref, out_w, out_h, cfg["source_quality"]["max_upscale"])

    hook = choose_hook(sentences, c, options.cold_open and knobs["cold_open"], options.hook_style)
    cold_seg: tuple[float, float] | None = None
    if hook["kind"] == "cold_open":
        hs: cv.Sentence = hook["sentence"]
        cold_seg = cv.pads(sentences, hs.idx, hs.idx, ref.duration)

    shots: list[tuple[float, float]] = []
    for interval in trim_pauses(words, c.start, c.end):
        shots += split_by_velocity(interval, words, knobs["cut"])
    all_shots = ([cold_seg] if cold_seg else []) + shots

    centers: list[float | None] = [None] * len(all_shots)
    if (options.reframe and layout in ("center_crop", "wide_fill") and ref.content_box[0] > ref.content_box[1]
            and crop_centers is not None and options.aspect != "16:9"):
        centers = crop_centers(all_shots)

    segments: list[Segment] = []
    strongest = max(body, key=cv.hook_strength) if body else None
    for n, ((a, b), cx) in enumerate(zip(all_shots, centers)):
        is_cold = cold_seg is not None and n == 0
        k = n - (1 if cold_seg else 0)
        zoom = 1.0 if is_cold else knobs["zooms"][k % len(knobs["zooms"])]
        if strongest is not None and not is_cold and a <= strongest.start < b + 1e-6 and options.intensity != "clean":
            zoom = max(zoom, 1.14)
        segments.append(Segment(id=f"s{n:02d}", source="a", src_in=round(a, 3), src_out=round(b, 3),
                                layout=layout, zoom=zoom, crop_x=cx,
                                rationale="cold-open hook line" if is_cold else f"conversation shot {k + 1}, zoom {zoom:g}"))
    duration = round(sum(s.out_duration for s in segments), 3)

    # captions: centre third of the frame, 3-5 words per card
    cy = min(max(options.caption_y or CAPTION_Y, CAPTION_BAND[0]), CAPTION_BAND[1])
    st["speech"] = {**st["speech"], "y": cy, "words_per_card": min(int(st["speech"].get("words_per_card", 4)), 5)}

    cards = []
    hook_start, hook_end = 0.1, min(HOOK_SECONDS, max(1.5, duration * 0.2))
    cards += hook_cards(hook, hook_start, hook_end, st, safe)
    mapped = map_words(segments, words)
    cards += speech_cards(mapped, duration, st, safe, set(key_terms(body, knobs["emphasis"])))

    # when the title already is the question, a "Question" label on top of it only adds clutter
    if (knobs["lower_third"] and c.question_included and options.show_question_label
            and sentences[c.first].question and hook["kind"] != "question"):
        q = sentences[c.first]
        q_start = 0.0 if cold_seg is None else (cold_seg[1] - cold_seg[0])
        q_end = min(q_start + (q.end - q.start) + 0.2, duration)
        if q_end - q_start >= 1.2:
            # just under the top safe line, clear of the hook title and of the bottom captions
            cards.append(lower_third_card("Question", q_start + 0.2, q_end, st, safe,
                                          y=safe["top"] + 0.045))
    for phrase, first_word in key_phrases(body, knobs["callouts"]):
        t = _first_time(mapped, first_word, hook_end + 0.5)
        if t is not None and t < duration - 2.0:
            text = phrase.lower() if st.get("case") == "lower" else phrase
            try:
                cards.append(callout_card(text, t, min(t + 2.6, duration), st, safe, y=CALLOUT_Y))
            except ValueError:
                pass
    if options.cta and duration > 6:
        cards.append(cta_card(options.cta, max(0.0, duration - 2.6), duration, st, safe, y=0.36))

    audio = AudioPlan(target_lufs=cfg["audio"]["target_lufs"], true_peak_max=cfg["audio"]["true_peak_max"],
                      fade_out=0.25, denoise=options.denoise)
    if options.music and Path(options.music["file"]).exists():
        audio.music = MusicBed(file=options.music["file"], licence=options.music.get("licence", "unknown"),
                               start_offset=0.0, level_db=options.music.get("level_db", -14.0),
                               duck_db=-22.0, duck=speech_spans(mapped))
    elif options.music:
        notes.append("Background music was requested but no music track is installed; rendered without it.")

    plan = EditPlan(
        clip_id=clip_id, project_id=project_id, format_key=cfg["format"]["key"], mode="commentary",
        script="conversation", sources={"a": ref}, rights=rights,
        moment={"type": "conversation", "unit": c.unit, "start": c.start, "end": c.end,
                "question_included": c.question_included, "partial": c.partial, "scores": c.scores,
                "score": c.score, "reasons": c.reasons,
                "hook": {"kind": hook["kind"], "text": hook["text"], "archetype": hook.get("archetype"),
                         "frames": hook.get("frames", [])},
                "rationale": "complete conversational unit selected from the full transcript"},
        segments=segments, text_cards=cards, caption_style=st, audio=audio,
        output=OutputSpec(width=out_w, height=out_h, fps=ref.fps),
        logo=({"file": options.logo, "width": 0.20, "top": safe["top"] + 0.012}
              if options.logo and Path(options.logo).exists() else None),
        progress_bar=options.accent if knobs["progress"] else None,
        hard_floor_seconds=cv.MIN_ABSOLUTE_SECONDS - 2.0,
        rationale=[f"layout {layout}", f"intensity {options.intensity}", f"hook: {hook['kind']}"] + c.reasons)
    return Built(plan=plan, hook=hook, notes=notes)

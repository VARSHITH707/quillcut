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
    # seg_max: longest stretch without a change of framing; zooms cycle through `zooms`
    "clean": {"seg_max": 40.0, "zooms": [1.0], "callouts": 0, "lower_third": False, "cold_open": False,
              "progress": False, "emphasis": 4},
    "dynamic": {"seg_max": 9.0, "zooms": [1.0, 1.10], "callouts": 1, "lower_third": True, "cold_open": True,
                "progress": False, "emphasis": 6},
    "high_energy": {"seg_max": 6.0, "zooms": [1.0, 1.12, 1.0, 1.18], "callouts": 2, "lower_third": True,
                    "cold_open": True, "progress": True, "emphasis": 8},
}
PAUSE_MAX = 0.85          # silences longer than this are tightened
PAUSE_KEEP = 0.30         # what is left of a tightened silence
HOOK_SECONDS = 2.8
HOOK_MAX_WORDS = 14


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
    """Keep intervals of the range with long silences tightened. Never cuts through a word."""
    cuts: list[tuple[float, float]] = []
    for a, b in zip(words, words[1:]):
        if b["start"] - a["end"] > PAUSE_MAX:
            cuts.append((a["end"] + PAUSE_KEEP / 2, b["start"] - PAUSE_KEEP / 2))
    out, t = [], start
    for c0, c1 in cuts:
        if c0 > t:
            out.append((round(t, 3), round(c0, 3)))
        t = c1
    out.append((round(t, 3), round(end, 3)))
    return [(a, b) for a, b in out if b - a >= 0.3]


def split_at_sentences(interval: tuple[float, float], sentences: list[cv.Sentence],
                       seg_max: float) -> list[tuple[float, float]]:
    """Cut a long interval at sentence boundaries so framing can change; contiguous pieces share the cut."""
    a, b = interval
    if b - a <= seg_max:
        return [interval]
    bounds = []
    for p, n in zip(sentences, sentences[1:]):
        if a < p.end and n.start < b and n.start - p.end > 0.0:
            bounds.append(round(p.end + (n.start - p.end) / 2, 3))
    pieces, cur = [], a
    for t in bounds:
        if t - cur >= 0.6 * seg_max and b - t >= 1.2:
            pieces.append((cur, t))
            cur = t
    pieces.append((cur, b))
    return pieces


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


def choose_hook(sentences: list[cv.Sentence], c: cv.Candidate, allow_cold_open: bool) -> dict:
    """A truthful hook: a stronger line from later in the answer, the question, or the opening line."""
    body = sentences[c.first:c.last + 1]
    answer = [s for s in body if not (s.question and c.question_included and s.idx == c.first)]
    first = answer[0] if answer else body[0]
    question = sentences[c.first] if c.question_included and sentences[c.first].question else None
    cold = None
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
    if cold is not None:
        return {"kind": "cold_open", "sentence": cold, "text": cold.text}
    if question is not None and question.n_words <= HOOK_MAX_WORDS + 4:
        return {"kind": "question", "sentence": question, "text": question.text}
    text = first.text
    if first.n_words > HOOK_MAX_WORDS:
        words = text.split()
        cut = HOOK_MAX_WORDS
        for k in range(HOOK_MAX_WORDS, 5, -1):          # prefer a clause boundary
            if re.search(r"[,;:]$", words[k - 1]):
                cut = k
                break
        text = " ".join(words[:cut]).rstrip(",;:") + "…"    # the ellipsis marks it as an excerpt, not a full quote
    return {"kind": "opening", "sentence": first, "text": text}


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


def fit_hook(hook: dict, start: float, end: float, style: dict, safe: dict):
    """Hook card for the verbatim line; if it is too long for the safe area, keep a shorter verbatim prefix
    ending in an ellipsis so it still reads as an excerpt."""
    words = hook["text"].rstrip("…").split()
    for n in range(len(words), 2, -1):
        text = " ".join(words[:n]).rstrip(",;:.")
        if n < len(words) or hook["text"].endswith("…"):
            text += "…"
        try:
            card = hook_card(text, start, end, style, safe)
        except ValueError:
            continue
        hook["text"] = text
        return card
    raise ValueError(f"hook cannot be fitted on screen: {hook['text']!r}")


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

    hook = choose_hook(sentences, c, options.cold_open and knobs["cold_open"])
    cold_seg: tuple[float, float] | None = None
    if hook["kind"] == "cold_open":
        hs: cv.Sentence = hook["sentence"]
        cold_seg = cv.pads(sentences, hs.idx, hs.idx, ref.duration)

    shots: list[tuple[float, float]] = []
    for interval in trim_pauses(words, c.start, c.end):
        shots += split_at_sentences(interval, body, knobs["seg_max"])
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

    cards = []
    hook_start, hook_end = 0.15, min(HOOK_SECONDS, max(1.5, duration * 0.2))
    cards.append(fit_hook(hook, hook_start, hook_end, st, safe))
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
                cards.append(callout_card(text, t, min(t + 2.6, duration), st, safe))
            except ValueError:
                pass
    if options.cta and duration > 6:
        cards.append(cta_card(options.cta, max(0.0, duration - 2.6), duration, st, safe, y=0.42))

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
                "score": c.score, "reasons": c.reasons, "hook": {"kind": hook["kind"], "text": hook["text"]},
                "rationale": "complete conversational unit selected from the full transcript"},
        segments=segments, text_cards=cards, caption_style=st, audio=audio,
        output=OutputSpec(width=out_w, height=out_h, fps=ref.fps),
        logo=({"file": options.logo, "width": 0.20, "top": safe["top"] + 0.012}
              if options.logo and Path(options.logo).exists() else None),
        progress_bar=options.accent if knobs["progress"] else None,
        hard_floor_seconds=cv.MIN_ABSOLUTE_SECONDS - 2.0,
        rationale=[f"layout {layout}", f"intensity {options.intensity}", f"hook: {hook['kind']}"] + c.reasons)
    return Built(plan=plan, hook=hook, notes=notes)

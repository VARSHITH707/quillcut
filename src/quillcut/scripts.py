"""Script templates → EditPlan. Format-agnostic: templates live in formats/<key>/scripts/*.yaml.

A template is a list of sections tiling [0, duration]. Each section pulls footage from a
source (a, b) relative to that source's peak, so the same template works on any video.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .captions import hook_card, speech_cards
from .config import load_yaml
from .paths import FORMATS
from .types import AudioPlan, EditPlan, OutputSpec, Segment, SourceRef

SLOWMO_SOURCE = 0.8      # seconds of source played at 0.5x across the payoff
PEAK_LEAD = 0.3          # start this far before the peak frame
MIN_SHOT = 0.4
ACCEL_RATIO = 0.8        # each accelerating shot is 80% of the previous one


class PlanRejected(Exception):
    pass


@dataclass
class SourceCtx:
    key: str
    ref: SourceRef
    t_peak: float
    lo: float
    hi: float
    peak_value: float = 0.0
    words: list[dict] = field(default_factory=list)   # source-time transcript words


def load_templates(format_key: str) -> dict[str, dict]:
    templates = {}
    for path in sorted((FORMATS / format_key / "scripts").glob("*.yaml")):
        tpl = load_yaml(path)
        validate_template(tpl, path.name)
        templates[tpl["key"]] = tpl
    return templates


def validate_template(tpl: dict, name: str = "") -> None:
    sections = tpl.get("sections") or []
    if not sections:
        raise ValueError(f"{name}: no sections")
    t = 0.0
    for i, sec in enumerate(sections):
        if abs(sec["start"] - t) > 1e-6:
            raise ValueError(f"{name}: section {i} starts at {sec['start']}, expected {t}")
        if sec["end"] <= sec["start"]:
            raise ValueError(f"{name}: section {i} is empty")
        if sec.get("hold") and i != len(sections) - 1:
            raise ValueError(f"{name}: hold is only allowed on the last section")
        t = sec["end"]
    if abs(t - tpl["duration"]) > 1e-6:
        raise ValueError(f"{name}: sections end at {t}, duration is {tpl['duration']}")


def shot_lengths(total: float, n: int, accelerate: bool) -> list[float]:
    raw = [ACCEL_RATIO ** i for i in range(n)] if accelerate else [1.0] * n
    lengths = [total * r / sum(raw) for r in raw]
    if min(lengths) < MIN_SHOT:
        raise PlanRejected(f"{n} shots in {total:.1f}s leaves a shot under {MIN_SHOT}s")
    return lengths


def _window(src: SourceCtx, sec: dict, width: float, used_end: dict) -> tuple[float, float]:
    anchor = sec.get("anchor", "around")
    p = src.t_peak
    if anchor == "from_peak":
        start = p - sec.get("lead", PEAK_LEAD)
        w0, w1 = start, start + width
    elif anchor == "before":
        end = p - sec.get("gap", PEAK_LEAD)
        w0, w1 = end - width, end
    elif anchor == "after":
        start = p + sec.get("gap", 1.0)
        w0, w1 = start, start + width
    elif anchor == "continue":
        start = used_end.get(src.key, p - PEAK_LEAD)
        w0, w1 = start, start + width
    elif anchor == "around":
        w0, w1 = p - 0.6 * width, p + 0.4 * width
    else:
        raise ValueError(f"unknown anchor '{anchor}'")
    if w1 - w0 > src.hi - src.lo:
        raise PlanRejected(f"source {src.key} range {src.hi - src.lo:.1f}s is too short for a {width:.1f}s window")
    if w0 < src.lo:
        w0, w1 = src.lo, src.lo + (w1 - w0)
    if w1 > src.hi:
        w0, w1 = src.hi - (w1 - w0), src.hi
    return w0, w1


def _place(w0: float, w1: float, lengths: list[float]) -> list[tuple[float, float]]:
    gap = (w1 - w0 - sum(lengths)) / (len(lengths) - 1) if len(lengths) > 1 else 0.0
    shots, t = [], w0
    for length in lengths:
        shots.append((round(t, 3), round(t + length, 3)))
        t += length + gap
    return shots


def choose_layout(ref: SourceRef, max_upscale: float) -> tuple[str, float]:
    """center_crop when a full-height 9:16 crop stays under max_upscale, else wide_fill."""
    cw, ch = ref.content_box[0], ref.content_box[1]
    if cw / ch >= 9 / 16:
        center = 1920 / ch
    else:
        center = 1080 / cw
    if center <= max_upscale:
        return "center_crop", round(center, 3)
    wide = 1080 / min(cw, ch * 0.8)
    return "wide_fill", round(wide, 3)


def build_plan(tpl: dict, sources: dict[str, SourceCtx], cfg: dict, style: dict, *, clip_id: str,
               project_id: str, mode: str, rights: dict, text: str | None = None,
               layout: str | None = None) -> EditPlan:
    max_up = cfg["source_quality"]["max_upscale"]
    for key in tpl.get("sources", ["a"]):
        if key not in sources:
            raise PlanRejected(f"script '{tpl['key']}' needs source '{key}'")
    layouts = {k: (layout, 0.0) if layout else choose_layout(s.ref, max_up) for k, s in sources.items()}

    segments: list[Segment] = []
    used_end: dict[str, float] = {}
    for n, sec in enumerate(tpl["sections"]):
        length = sec["end"] - sec["start"]
        hold = float(sec.get("hold", 0.0))
        footage = length - hold
        kind = sec.get("kind", "side")
        label = f"{sec['start']:g}-{sec['end']:g}s {kind}"
        if kind == "split":
            top, bottom = sources[sec["top"]], sources[sec["bottom"]]
            t0, t1 = _window(top, sec, footage, used_end)
            b0, _ = _window(bottom, sec, footage, used_end)
            segments.append(Segment(id=f"s{n}", source=top.key, src_in=round(t0, 3), src_out=round(t1, 3),
                                    layout="split", second_source=bottom.key, second_in=round(b0, 3),
                                    rationale=f"{label}: {top.key} over {bottom.key}, anchor {sec.get('anchor', 'around')}"))
            used_end[top.key] = t1
            used_end[bottom.key] = b0 + footage
        elif kind == "slowmo":
            src = sources[sec["source"]]
            lead = footage - SLOWMO_SOURCE / 0.5
            p = src.t_peak
            if lead >= 0.2:
                segments.append(Segment(id=f"s{n}a", source=src.key, src_in=round(p - SLOWMO_SOURCE / 2 - lead, 3),
                                        src_out=round(p - SLOWMO_SOURCE / 2, 3), layout=layouts[src.key][0],
                                        rationale=f"{label}: lead-in to the slow-motion payoff"))
            segments.append(Segment(id=f"s{n}b", source=src.key, src_in=round(p - SLOWMO_SOURCE / 2, 3),
                                    src_out=round(p + SLOWMO_SOURCE / 2, 3), speed=0.5, layout=layouts[src.key][0],
                                    rationale=f"{label}: 0.5x across the peak at {p:.2f}s"))
            used_end[src.key] = p + SLOWMO_SOURCE / 2
        else:
            src = sources[sec["source"]]
            cuts = int(sec.get("cuts", 1))
            lengths = shot_lengths(footage, cuts, bool(sec.get("accelerate")))
            spread = float(sec.get("spread", 1.0 if cuts == 1 else 2.0))
            w0, w1 = _window(src, sec, footage * spread, used_end)
            for i, (s0, s1) in enumerate(_place(w0, w1, lengths)):
                segments.append(Segment(
                    id=f"s{n}_{i}", source=src.key, src_in=s0, src_out=s1, layout=layouts[src.key][0],
                    audio_gain=float(sec.get("audio_gain", 1.0)), push_in=float(sec.get("push_in", 0.0)) if i == 0 else 0.0,
                    rationale=f"{label}: shot {i + 1}/{cuts} from {src.key}, anchor {sec.get('anchor', 'around')}"))
            used_end[src.key] = w1
        if hold:
            segments[-1] = segments[-1].model_copy(update={"hold_after": hold})

    duration = sum(s.out_duration for s in segments)
    floor = cfg["duration"]["hard_floor_seconds"]
    if duration < floor - 0.01:
        raise PlanRejected(f"cannot_reach_duration_floor_without_padding ({duration:.1f}s < {floor}s)")

    safe = cfg["safe_area"]
    cards = []
    for item in tpl.get("text", []):
        content = text if text is not None else item.get("text")
        if content:
            cards.append(hook_card(content, item["start"], item["end"], style, safe))
    if tpl.get("speech_captions") == "auto" and mode == "commentary":
        cards += speech_cards(map_words(segments, sources), duration, style, safe)

    a = sources["a"]
    fps = a.ref.fps
    return EditPlan(
        clip_id=clip_id, project_id=project_id, format_key=cfg["format"]["key"], mode=mode,
        script=tpl["key"], sources={k: s.ref for k, s in sources.items()}, rights=rights,
        moment={"type": tpl.get("pattern"), "t_peak": a.t_peak, "peak_intensity": a.peak_value,
                "rationale": f"peak of fused intensity in {a.lo:.1f}-{a.hi:.1f}s of source a"},
        segments=segments, text_cards=cards, caption_style=style,
        audio=AudioPlan(target_lufs=cfg["audio"]["target_lufs"], true_peak_max=cfg["audio"]["true_peak_max"],
                        fade_out=float(tpl.get("fade_out", 0.0))),
        output=OutputSpec(fps=fps), hard_floor_seconds=floor,
        placeholders_used=["caption_position", "shot_lengths", "peak_lead"],
        rationale=[f"layout {k}: {v[0]} (upscale {v[1]})" for k, v in layouts.items()]
                  + ([tpl["notes"]] if tpl.get("notes") else []),
    )


def map_words(segments: list[Segment], sources: dict[str, SourceCtx]) -> list[dict]:
    """Source-time words → clip time, following segment order and speed."""
    out, offset = [], 0.0
    for seg in segments:
        for w in sources[seg.source].words:
            if seg.src_in <= w["start"] and w["end"] <= seg.src_out:
                out.append({**w, "start": round(offset + (w["start"] - seg.src_in) / seg.speed, 3),
                            "end": round(offset + (w["end"] - seg.src_in) / seg.speed, 3)})
        offset += seg.out_duration
    return out

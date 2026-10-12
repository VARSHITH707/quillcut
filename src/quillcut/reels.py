"""Reel books: hand-directed reels assembled from complete source shots.

A reel is a list of parts `[source, from, to]`. Every part is snapped so it starts on a source
shot cut that is not mid-sentence and ends after the sentence in progress, on the end of the
shot that contains it: no cut ever lands inside a shot or a spoken line. A shared end card
(the game's own title shot) closes every reel so viewers know what game it is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .captions import hook_card, speech_cards
from .config import load_yaml
from .paths import CLIPPING
from .scripts import PlanRejected, SourceCtx, choose_layout, map_words
from .types import AudioPlan, EditPlan, MusicBed, OutputSpec, Segment, SfxHit

MUSIC_ROOT = CLIPPING / "Sound Effects and Music"
SENTENCE_GAP = 0.45      # words closer than this belong to one spoken run
START_SEARCH = 2.5       # seconds either side of a part's requested start
DUCK_PAD = 0.25
PACE_TRACKS = {"fast": "qc_phonk_150", "tense": "qc_dark_trap_140", "build": "qc_tension_120",
               "calm": "qc_ambient_96"}


@dataclass
class ShotSource:
    ctx: SourceCtx
    shots: list[tuple[float, float]]
    extra_cuts: tuple[float, ...] = ()      # hand-marked cuts the detector misses (dissolves)

    @property
    def cuts(self) -> list[float]:
        return sorted({a for a, _ in self.shots} | {self.shots[-1][1]} | set(self.extra_cuts))


def apply_fixes(words: list[dict], fixes: list[dict]) -> list[dict]:
    """Each fix replaces every word inside [start, end] with `text` (spread evenly), or deletes
    them when text is empty. Used for verified mis-hearings and for lines we cannot verify."""
    out = list(words)
    for fix in sorted(fixes, key=lambda f: f["start"]):
        a, b = float(fix["start"]), float(fix["end"])
        out = [w for w in out if not (a - 0.01 <= w["start"] and w["end"] <= b + 0.01)]
        tokens = (fix.get("text") or "").split()
        if tokens:
            total = sum(len(t) for t in tokens)
            t = a
            for tok in tokens:
                d = (b - a) * len(tok) / total
                out.append({"word": tok, "start": round(t, 3), "end": round(t + d, 3), "probability": 1.0})
                t += d
    return sorted(out, key=lambda w: w["start"])


def apply_word_fixes(words: list[dict], fixes: dict[str, str]) -> list[dict]:
    """Whole-word, case-insensitive replacements, e.g. {"boxable": "BOXABL"}; punctuation kept."""
    if not fixes:
        return words
    table = {k.lower(): v for k, v in fixes.items()}
    out = []
    for w in words:
        m = re.match(r"^(\W*)(.*?)(\W*)$", w["word"])
        lead, core, trail = m.groups()
        repl = table.get(core.lower())
        out.append({**w, "word": f"{lead}{repl}{trail}"} if repl is not None else w)
    return out


def speech_runs(words: list[dict]) -> list[tuple[float, float]]:
    runs: list[list[float]] = []
    for w in words:
        if runs and w["start"] - runs[-1][1] <= SENTENCE_GAP:
            runs[-1][1] = max(runs[-1][1], w["end"])
        else:
            runs.append([w["start"], w["end"]])
    return [(a, b) for a, b in runs]


def _inside(t: float, runs: list[tuple[float, float]]) -> tuple[float, float] | None:
    return next(((a, b) for a, b in runs if a + 0.05 < t < b - 0.05), None)


def snap_part(src: ShotSource, start: float, end: float) -> tuple[float, float, list[str]]:
    runs = speech_runs(src.ctx.words)
    cuts = src.cuts
    notes = []
    near = [c for c in cuts if abs(c - start) <= START_SEARCH and not _inside(c, runs)]
    if near:
        s = min(near, key=lambda c: abs(c - start))
    else:
        run = _inside(start, runs)
        s = max([c for c in cuts if c <= (run[0] if run else start)], default=0.0)
    if abs(s - start) > 0.01:
        notes.append(f"start {start:.2f}->{s:.2f} (shot cut, not mid-sentence)")
    run = _inside(end, runs)
    target = run[1] + 0.1 if run else end
    after = [c for c in cuts if c >= target - 0.04]
    e = after[0] if after else src.ctx.ref.duration
    if abs(e - end) > 0.01:
        notes.append(f"end {end:.2f}->{e:.2f} ({'finish the line, then ' if run else ''}end of shot)")
    if e - s < 0.5:
        raise PlanRejected(f"part {start}-{end} collapses after snapping")
    return round(s, 3), round(e, 3), notes


def load_music_library() -> dict[str, dict]:
    manifest = load_yaml(MUSIC_ROOT / "manifest.yaml")
    lib = {}
    for kind in ("music", "sfx"):
        for item in manifest.get(kind, []):
            if not item.get("licence") or not item.get("licence_url") or not item.get("platforms_allowed"):
                raise ValueError(f"LicenceError: {item.get('file')} has no complete licence record")
            name = item["file"].split("/")[-1].rsplit(".", 1)[0]
            lib[name] = {**item, "path": str(MUSIC_ROOT / item["file"]), "kind": kind}
    return lib


def build_reel(reel: dict, book: dict, sources: dict[str, ShotSource], cfg: dict, style: dict,
               library: dict, project_id: str, first_visible=None) -> EditPlan:
    """first_visible(source_key, t) -> t' skips a fade-in from black at the reel's first frame."""
    max_up = cfg["source_quality"]["max_upscale"]
    segments: list[Segment] = []
    notes: list[str] = []
    for i, (key, a, b, *opts) in enumerate(reel["parts"]):
        src = sources[key]
        s, e, why = snap_part(src, float(a), float(b))
        if i == 0 and first_visible:
            visible = first_visible(key, s)
            if visible > s + 0.01:
                why.append(f"start {s:.2f}->{visible:.2f} (skip fade-in from black)")
                s = visible
        extra = next((o for o in opts if isinstance(o, dict)), {})
        if extra.get("exact"):                   # hand-checked shot cuts: use as given
            s, e, why = float(a), float(b), ["exact cut points (hand-checked)"]
        if extra.get("exact_start") or extra.get("exact_end"):
            ws = src.ctx.words
            if extra.get("exact_start"):
                starts = [w["start"] for w in ws if w["start"] >= float(a) - 0.05]
                s = round(max(0.0, (min(starts) if starts else float(a)) - 0.05), 3)
                why.append(f"start at word boundary {s:.2f}")
            if extra.get("exact_end"):
                ends = [w["end"] for w in ws if w["end"] <= float(b) + 0.05 and w["start"] >= s]
                e = round(min(src.ctx.ref.duration, (max(ends) if ends else float(b)) + 0.12), 3)
                why.append(f"end at word boundary {e:.2f}")
        named = next((o for o in opts if isinstance(o, str)), None)
        layout = named or choose_layout(src.ctx.ref, max_up)[0]
        why_text = "; ".join(why) or "on cuts"
        t, k = s, 0
        for at, dur, vkey, vfrom in sorted(extra.get("broll", [])):   # [main time, seconds, source, from]
            at, dur = max(float(at), s), float(dur)
            if 0.0 <= at - t < 0.2:          # an insert right at the start replaces it, no 0.1 s sliver
                at = t
            if at + dur > e or at - t < 0.0:
                raise PlanRejected(f"b-roll at {at} does not fit part {s:.2f}-{e:.2f}")
            if at - t >= 0.2:
                segments.append(Segment(id=f"p{i}_{k}", source=key, src_in=round(t, 3), src_out=round(at, 3),
                                        layout=layout, rationale=f"part {i + 1}: {key} {t:.2f}-{at:.2f}; {why_text}"))
                k += 1
            vlayout = choose_layout(sources[vkey].ctx.ref, max_up)[0]
            segments.append(Segment(id=f"p{i}_{k}", source=key, src_in=round(at, 3), src_out=round(at + dur, 3),
                                    layout=vlayout, video_source=vkey, video_in=float(vfrom),
                                    rationale=f"b-roll: {vkey} {vfrom}+{dur}s over {key} audio"))
            k += 1
            t = at + dur
        if e - t >= 0.2:
            segments.append(Segment(id=f"p{i}_{k}", source=key, src_in=round(t, 3), src_out=e, layout=layout,
                                    rationale=f"part {i + 1}: {key} {t:.2f}-{e:.2f}; {why_text}"))
        notes += why
    card = book.get("end_card")
    body = sum(sg.out_duration for sg in segments)
    if card:
        segments.append(Segment(id="end", source=card["source"], src_in=card["from"], src_out=card["to"],
                                layout=card.get("layout", "fit_blur"),
                                rationale="end card: the game's own title shot"))
    duration = sum(sg.out_duration for sg in segments)
    lo, hi = cfg["duration"]["hard_floor_seconds"], cfg["duration"]["max_seconds"]
    if not lo - 0.01 <= duration <= hi + 0.01:
        raise PlanRejected(f"{reel['id']}: {duration:.1f}s outside {lo}-{hi}s after snapping")

    safe = cfg["safe_area"]
    words = map_words(segments, {k: s.ctx for k, s in sources.items()})
    cards = []
    cta = reel.get("cta")             # optional closing question: {text, seconds}; set by the book or a creative repair
    cta_start = max(0.0, body - float(cta["seconds"])) if cta else None
    if reel.get("hook"):
        seconds = reel.get("hook_seconds", book.get("hook_seconds", 3.0))
        hook_end = body if seconds == "full" else min(float(seconds), segments[0].out_duration)
        if cta_start is not None:                 # the question takes the headline's place at the end
            hook_end = min(hook_end, max(0.5, cta_start))
        cards.append(hook_card(reel["hook"], 0.0, hook_end, style, safe))
    if cta:
        cards.append(hook_card(cta["text"], cta_start, body, style, safe, kind="label"))
    if reel.get("captions", True):
        cards += speech_cards(words, body, style, safe)
    mark = book.get("watermark")
    if mark:
        cards.append(hook_card(mark, 0.0, duration, style, safe, kind="label", block="watermark"))
    if card and book.get("end_text"):
        cards.append(hook_card(book["end_text"], body, duration, style, safe, kind="label"))

    duck, cur = [], None
    for w in words:
        a, b = w["start"] - DUCK_PAD, w["end"] + DUCK_PAD
        if cur and a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            if cur:
                duck.append(tuple(cur))
            cur = [a, b]
    if cur:
        duck.append(tuple(cur))

    music = None
    track_name = (reel.get("music") or {}).get("track") or PACE_TRACKS.get(reel.get("pace", "tense"))
    if book.get("music") is False:          # campaign rules: source audio only
        track_name = None
    if track_name and track_name != "none":
        track = library[track_name]
        drop_at = (reel.get("music") or {}).get("drop_at", 0.0 if not track.get("drop") else 6.0)
        music = MusicBed(file=track["path"], licence=track["licence"],
                         start_offset=round(track.get("drop", 0.0) - drop_at, 3),
                         level_db=reel.get("music_level_db", -6.0), duck_db=reel.get("music_duck_db", -17.0),
                         duck=[(round(a, 3), round(b, 3)) for a, b in duck])
    sfx = []
    if book.get("sfx") is False:
        library = {}
    if card and "impact" in library:
        sfx.append(SfxHit(file=library["impact"]["path"], licence=library["impact"]["licence"], t=body, gain_db=-4))
    if card and "whoosh" in library:
        sfx.append(SfxHit(file=library["whoosh"]["path"], licence=library["whoosh"]["licence"],
                          t=max(0.0, body - 0.35), gain_db=-10))

    logo = book.get("logo")
    if logo and not Path(logo["file"]).is_file():
        raise PlanRejected(f"{reel['id']}: logo file not found: {logo['file']}")
    first = sources[reel["parts"][0][0]].ctx
    return EditPlan(
        clip_id=reel["id"], project_id=project_id, format_key=cfg["format"]["key"], mode="commentary",
        script=f"reel:{book.get('key', 'book')}", sources={k: s.ctx.ref for k, s in sources.items()},
        rights=book.get("rights_record", {"status": "unknown", "resolved_by": "reel book"}),
        moment={"type": reel.get("pattern", "montage"), "t_peak": first.t_peak, "topic": reel.get("topic"),
                "rationale": reel.get("why", "hand-directed reel")},
        segments=segments, text_cards=cards, caption_style=style,
        audio=AudioPlan(target_lufs=cfg["audio"]["target_lufs"], true_peak_max=cfg["audio"]["true_peak_max"],
                        fade_out=book.get("fade_out", 0.0), music=music, sfx=sfx),
        output=OutputSpec(fps=first.ref.fps), look=book.get("look", {}),
        progress_bar=book.get("progress_bar"), logo=logo, hard_floor_seconds=lo,
        placeholders_used=["music_levels", "caption_position"],
        rationale=notes + [f"music: {track_name or 'none'}"],
    )

"""Findings -> minimal, reel-book-safe edits.

Repairs are expressed on the reel-book entry (its `parts`, an optional `cta`, a caption position),
never on rendered media: the edited reel goes back through `reels.build_reel`, so sentence/shot
snapping, captions, ducking, the end card and every campaign rule stay exactly as the book defines.

A fix is applied only when it is (1) confident enough, (2) allowed by the book (nothing marked
`essential`, `exact` or `locked` is touched; instead the conflict is reported with the smallest
acceptable correction) and (3) provably buildable: the edited reel must still plan within the
duration limits. Anything that needs a human decision is skipped, with the reason recorded.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Callable

from ..scripts import PlanRejected
from .facts import ClipFacts
from .schema import SEVERITY_RANK, CreativeReport, Finding

ORDER = {"trim_end": 0, "extend_end": 0, "reselect": 0, "trim_start": 1, "cut_span": 2,
         "move_captions": 3, "retime_cta": 4, "add_question_overlay": 5}
STRUCTURAL = {"trim_end", "extend_end", "reselect", "trim_start", "cut_span"}
RENUMBERS = {"cut_span", "reselect"}


class Skip(Exception):
    pass


class Conflict(Exception):
    pass


@dataclass(frozen=True)
class State:
    reel: dict
    caption_y: float | None = None          # speech caption y override (style is not part of the reel)
    alt: int = 0                            # alternates already used


@dataclass
class Outcome:
    state: State
    applied: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)


def split_part(part: list) -> tuple[str, float, float, str | None, dict]:
    key, a, b, *opts = part
    layout = next((o for o in opts if isinstance(o, str)), None)
    return key, float(a), float(b), layout, dict(next((o for o in opts if isinstance(o, dict)), {}))


def join_part(key, a, b, layout, opts) -> list:
    return [key, round(a, 3), round(b, 3)] + ([layout] if layout else []) + ([opts] if opts else [])


def protection(reel: dict, i: int) -> str | None:
    if reel.get("locked"):
        return "the reel is locked"
    opts = split_part(reel["parts"][i])[4]
    if opts.get("essential"):
        return f"part {i + 1} is marked essential"
    if opts.get("exact"):
        return f"part {i + 1} has hand-checked exact cut points"
    return None


def _with_parts(state: State, parts: list) -> State:
    reel = copy.deepcopy(state.reel)
    reel["parts"] = parts
    return replace(state, reel=reel)


def _edit_part(state: State, i: int, **changes) -> State:
    parts = copy.deepcopy(state.reel["parts"])
    key, a, b, layout, opts = split_part(parts[i])
    a, b = changes.pop("a", a), changes.pop("b", b)
    parts[i] = join_part(key, a, b, layout, {**opts, **changes})
    return _with_parts(state, parts)


def _guard(state: State, i: int | None, fd: Finding) -> int:
    if i is None or not 0 <= i < len(state.reel["parts"]):
        raise Skip("the finding is not in an editable reel part (end card or non-reel plan)")
    why = protection(state.reel, i)
    if why:
        raise Conflict(f"{why}; smallest acceptable correction: {fd.recommended_fix}")
    return i


def _fix_end(state: State, fd: Finding, f: ClipFacts, build) -> State:
    i = _guard(state, fd.params.get("part"), fd)
    src_out = fd.params["src_out"]
    wanted = [(abs(t - src_out), name, t) for name, t in
              (("extend_end", fd.params.get("extend")), ("trim_end", fd.params.get("trim"))) if t is not None]
    if fd.edit_action in ("extend_end", "trim_end"):                 # the recommended action first
        wanted.sort(key=lambda w: (w[1] != fd.edit_action, w[0]))
    else:
        wanted.sort()
    last = None
    for _, name, target in wanted:
        trial = _edit_part(state, i, b=target, exact_end=True)
        try:
            build(trial)
            return trial
        except (PlanRejected, ValueError) as exc:
            last = f"{name} to {target:.2f}s: {exc}"
    raise Skip("no safe boundary builds within the duration limits" + (f" ({last})" if last else ""))


def _fix_start(state: State, fd: Finding, f: ClipFacts, build) -> State:
    i = _guard(state, fd.params.get("part"), fd)
    target = fd.params.get("src_start")
    if target is None:
        raise Skip("no later sentence start to move to")
    if split_part(state.reel["parts"][i])[0] != fd.params.get("source"):
        raise Skip("the opening comes from a different source than the part")
    trial = _edit_part(state, i, a=target, exact_start=True)
    build(trial)
    return trial


def _fix_cut(state: State, fd: Finding, f: ClipFacts, build) -> State:
    if fd.params.get("jump_cut", True) and not f.policy.get("allow_jump_cuts", False):
        raise Skip("it would cut inside a shot (set creative.allow_jump_cuts: true to allow jump cuts)")
    s0, s1 = fd.params["span_src"]
    sp = f.span_at(fd.start_time + 0.01)
    i = _guard(state, sp.part, fd)
    spans = [x for x in f.spans if x.part == i]
    lo, hi = min(x.seg.src_in for x in spans), max(x.seg.src_out for x in spans)
    if not lo <= s0 < s1 <= hi:
        raise Skip("the span crosses part boundaries")
    parts = copy.deepcopy(state.reel["parts"])
    key, a, b, layout, opts = split_part(parts[i])
    broll = opts.get("broll", [])
    if any(float(at) < s1 and float(at) + float(dur) > s0 for at, dur, *_ in broll):
        raise Skip("a b-roll insert overlaps the span")
    first = {**opts, "exact_end": True, "broll": [x for x in broll if float(x[0]) + float(x[1]) <= s0]}
    second = {**opts, "exact_start": True, "broll": [x for x in broll if float(x[0]) >= s1]}
    for d in (first, second):
        if not d["broll"]:
            d.pop("broll")
    parts[i:i + 1] = [join_part(key, a, s0, layout, first), join_part(key, s1, b, layout, second)]
    trial = _with_parts(state, parts)
    build(trial)
    return trial


def _fix_captions(state: State, fd: Finding, f: ClipFacts, build) -> State:
    options = f.cc["captions"]["y_options"]
    current = state.caption_y or f.plan.caption_style["speech"]["y"]
    later = options[options.index(current) + 1:] if current in options else options[:1]
    if not later:
        raise Skip("every caption position has been tried")
    trial = replace(state, caption_y=later[0])
    build(trial)
    return trial


def _fix_cta(state: State, fd: Finding, f: ClipFacts, build) -> State:
    cta_cfg = f.cc["cta"]
    reel = copy.deepcopy(state.reel)
    if fd.edit_action == "retime_cta":
        if "cta" not in reel:
            raise Skip("the CTA is part of the book's end text, not a repairable overlay")
        reel["cta"]["seconds"] = max(float(reel["cta"]["seconds"]), cta_cfg["min_seconds"] + 0.3)
    else:
        if f.policy.get("cta") != "auto":
            raise Skip("campaign policy does not allow automatic overlays (creative.cta: auto)")
        question = fd.params.get("question")
        if not question:
            raise Skip("no verified question available (set creative.cta_questions or run the text model)")
        if "cta" in reel:
            raise Skip("an overlay is already present")
        reel["cta"] = {"text": question, "seconds": cta_cfg["seconds"]}
    trial = replace(state, reel=reel)
    build(trial)
    return trial


def _fix_reselect(state: State, fd: Finding, f: ClipFacts, build) -> State:
    alts = state.reel.get("alternates") or []
    if state.alt >= len(alts):
        raise Skip("no alternate segment is defined for this reel (add `alternates:` to the book)")
    if state.reel.get("locked"):
        raise Conflict("the reel is locked; recommended: review the ending manually")
    trial = replace(_with_parts(state, copy.deepcopy(alts[state.alt])), alt=state.alt + 1)
    build(trial)
    return trial


HANDLERS: dict[str, Callable] = {
    "trim_end": _fix_end, "extend_end": _fix_end, "trim_start": _fix_start, "cut_span": _fix_cut,
    "move_captions": _fix_captions, "retime_cta": _fix_cta, "add_question_overlay": _fix_cta,
    "reselect": _fix_reselect}


def _record(fd: Finding, **extra) -> dict:
    return {"category": fd.category, "issue": fd.issue, "action": fd.edit_action,
            "start": fd.start_time, "end": fd.end_time, "confidence": fd.confidence, **extra}


def plan_repairs(report: CreativeReport, f: ClipFacts, state: State, build: Callable[[State], object]) -> Outcome:
    """Apply as many safe repairs as possible to `state`. `build(state)` must raise if the reel no longer plans."""
    cc = f.cc
    out = Outcome(state)
    touched: set[int] = set()
    # cut_span and reselect renumber the parts, so they go last and at most once per cycle
    todo = sorted(report.findings(cc["count_min_confidence"]),
                  key=lambda x: (x.edit_action in RENUMBERS, -SEVERITY_RANK[x.severity], ORDER.get(x.edit_action, 9)))
    renumbered = False
    for fd in todo:
        if fd.severity == "info":
            continue
        if fd.edit_action in ("none", "manual"):
            out.skipped.append(_record(fd, reason="needs a human decision: " + fd.recommended_fix))
            continue
        if fd.confidence < cc["act_min_confidence"]:
            out.skipped.append(_record(fd, reason=f"confidence {fd.confidence:.2f} is below the action threshold "
                                                  f"{cc['act_min_confidence']}"))
            continue
        part = fd.params.get("part")
        if fd.edit_action in STRUCTURAL and ((part is not None and part in touched) or renumbered):
            out.skipped.append(_record(fd, reason="another edit is applied first; re-evaluated next cycle"))
            continue
        try:
            new = HANDLERS[fd.edit_action](out.state, fd, f, build)
        except Conflict as exc:
            out.conflicts.append(_record(fd, reason=str(exc)))
            continue
        except Skip as exc:
            out.skipped.append(_record(fd, reason=str(exc)))
            continue
        except (PlanRejected, ValueError) as exc:
            out.skipped.append(_record(fd, reason=f"the edit would not build: {exc}"))
            continue
        out.state = new
        renumbered |= fd.edit_action in RENUMBERS
        if part is not None:
            touched.add(part)
        out.applied.append(_record(fd, fix=fd.recommended_fix))
    return out

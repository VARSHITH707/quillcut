"""The creative pass: render a candidate, evaluate it, repair what is safely repairable, re-render,
re-evaluate, then decide. Existing production artifacts are never overwritten:

  * candidates render to  <project>/creative/<clip>/v<N>.mp4  (N continues after existing plan versions);
  * the production `final/`, `renders/` and the normal published folder are not touched;
  * an APPROVED candidate is copied to  <publish_dir>/creative/<clip>_v<N>.mp4, next to its report;
  * technical QC of every candidate runs on the actual rendered file, with its own versioned QC json.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable

from .. import qc
from ..config import load_config, load_yaml
from ..paths import FORMATS
from ..render import render_plan
from ..reels import apply_fixes, apply_word_fixes, build_reel
from ..scripts import PlanRejected
from ..storage import Project, atomic_write_json, read_json
from ..types import EditPlan
from .facts import ClipFacts
from .repair import State, plan_repairs
from .schema import Attempt, CreativeRecord, CreativeReport
from .score import build_report, compliance_checks, evaluate

log = logging.getLogger("quillcut.creative")


def next_version(project: Project, clip_id: str) -> int:
    nums = [int(m.group(1)) for p in project.sub("edit_specs").glob(f"{clip_id}.v*.editplan.json")
            if (m := re.search(r"\.v(\d+)\.editplan", p.name))]
    cdir = project.path / "creative" / clip_id
    nums += [int(p.stem[1:]) for p in cdir.glob("v*.mp4")] if cdir.exists() else []
    return max(nums, default=0) + 1


def render_candidate(plan: EditPlan, project: Project, cfg: dict, fast: bool = False):
    atomic_write_json(project.sub("edit_specs") / f"{plan.clip_id}.v{plan.plan_version}.editplan.json",
                      json.loads(plan.model_dump_json()))
    cdir = project.sub("creative") / plan.clip_id
    cdir.mkdir(exist_ok=True)
    out, work = cdir / f"v{plan.plan_version}.mp4", cdir / f"work_v{plan.plan_version}"
    r = cfg["render"]
    render_plan(plan, out, work, preset="veryfast" if fast else r["preset"], crf=r["crf"], threads=r["threads"])
    shutil.rmtree(work, ignore_errors=True)
    checks = qc.run_qc(out, plan, cfg)
    atomic_write_json(project.sub("qc") / f"{plan.clip_id}.v{plan.plan_version}.json", qc.as_dicts(checks))
    return out, checks


def _failures(checks) -> list[str]:
    return [f"{c.name}: {c.message}" for c in checks if c.status == "FAIL"]


def _evaluate_attempt(plan: EditPlan, video: Path, checks, facts: ClipFacts, models, evaluate_fn, book: dict,
                      policy: dict, cfg: dict, render_s: float) -> Attempt:
    """Evaluate one rendered candidate and wrap everything the record needs into an Attempt."""
    t = time.perf_counter()
    results, info = evaluate_fn(facts, models)
    evaluate_s = time.perf_counter() - t
    comp = _failures(compliance_checks(plan, book, policy))
    qc_fail = _failures(checks)
    report = build_report(plan.clip_id, plan.plan_version, results, cfg, qc_failures=qc_fail,
                          compliance_failures=comp, evaluator=info)
    return Attempt(plan_version=plan.plan_version, overall=report.overall, scores=report.scores(),
                   decision=report.decision, technical_qc="FAIL" if qc_fail else "PASS", qc_failures=qc_fail,
                   compliance="FAIL" if comp else "PASS", compliance_failures=comp, video=str(video), report=report,
                   render_seconds=round(render_s, 1), evaluate_seconds=round(evaluate_s, 1))


def run_reel(reel: dict, book: dict, sources: dict, cfg: dict, style: dict, library: dict, project: Project, *,
             publish_dir: Path, first_visible=None, models=None, fast: bool = False, repair: bool = True,
             render: Callable = render_candidate, evaluate_fn: Callable = evaluate) -> dict:
    """Render -> evaluate -> (repair -> re-render -> evaluate)... up to `max_repairs`, then decide and publish."""
    cc, clip = cfg["creative"], reel["id"]
    policy = {**book.get("creative", {}), **reel.get("creative", {})}
    src_words = {k: s.ctx.words for k, s in sources.items()}
    shots = {k: s.shots for k, s in sources.items()}
    state, version, prev = State(reel=copy.deepcopy(reel)), next_version(project, clip), None
    rec = CreativeRecord(clip_id=clip)

    def build(st: State) -> EditPlan:
        sty = copy.deepcopy(style)
        if st.caption_y:
            sty["speech"]["y"] = st.caption_y
        return build_reel(st.reel, book, sources, cfg, sty, library, project.id, first_visible=first_visible)

    for n in range(cc["max_repairs"] + 1 if repair else 1):
        try:
            plan = build(state).model_copy(update={"plan_version": version + n, "repair_of": prev})
        except (PlanRejected, ValueError) as exc:
            if n == 0:
                return {"clip_id": clip, "rejected": str(exc)}
            break
        t = time.perf_counter()
        try:
            video, checks = render(plan, project, cfg, fast)
        except Exception as exc:                                  # a render crash is a failed attempt, not an approval
            log.exception("%s: render failed", clip)
            rec.attempts.append(Attempt(plan_version=plan.plan_version, overall=None, scores={},
                                        decision="MANUAL_REVIEW", technical_qc="FAIL",
                                        qc_failures=[f"render error: {exc}"], compliance="FAIL"))
            break
        facts = ClipFacts(plan, src_words, shots, video, policy, cfg)
        attempt = _evaluate_attempt(plan, video, checks, facts, models, evaluate_fn, book, policy, cfg,
                                    time.perf_counter() - t)
        rec.attempts.append(attempt)
        if attempt.decision == "APPROVED" or n == cc["max_repairs"] or not repair:
            break
        outcome = plan_repairs(attempt.report, facts, state, build)
        attempt.skipped, attempt.conflicts, attempt.applied = outcome.skipped, outcome.conflicts, outcome.applied
        if not outcome.applied:
            break
        state, prev = outcome.state, plan.plan_version
    return _finish(rec, project, publish_dir, clip)


def _finish(rec: CreativeRecord, project: Project, publish_dir: Path, clip: str) -> dict:
    ok = [a for a in rec.attempts if a.video]
    if not ok:
        rec.final_decision = "MANUAL_REVIEW"
    else:
        approved = next((a for a in ok if a.decision == "APPROVED"), None)
        best = approved or max(ok, key=lambda a: (a.technical_qc == "PASS", a.overall or 0, a.plan_version))
        rec.final_decision, rec.final_plan_version = best.decision, best.plan_version
        rec.after_overall, rec.evaluator = best.overall, (best.report.evaluator if best.report else {})
        if approved:
            dest = publish_dir / "creative" / f"{clip}_v{approved.plan_version}.mp4"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(approved.video, dest)
            rec.published = str(dest)
    rec.before_overall = rec.attempts[0].overall if rec.attempts else None
    rec.retry_count = max(0, len(rec.attempts) - 1)
    path = project.sub("creative") / f"{clip}.creative.json"
    atomic_write_json(path, json.loads(rec.model_dump_json()))
    (publish_dir / "creative").mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, publish_dir / "creative" / f"{clip}.creative.json")
    project.mark_clip(clip, creative={"decision": rec.final_decision, "overall": rec.after_overall,
                                      "version": rec.final_plan_version, "record": str(path)})
    last = rec.attempts[-1].report if rec.attempts else None
    return {"clip_id": clip, "decision": rec.final_decision, "before": rec.before_overall,
            "overall": rec.after_overall, "retries": rec.retry_count, "version": rec.final_plan_version,
            "published": rec.published, "record": str(path),
            "reasons": last.reasons if last else rec.attempts[-1].qc_failures if rec.attempts else []}


# ------------------------------------------------------------------ read-only review of existing finals

def find_project(fmt: str, source: Path) -> Project | None:
    want = os.path.normcase(os.path.abspath(source))
    for pj in (FORMATS / fmt / "projects").glob("*/project.json"):
        if os.path.normcase(os.path.abspath(read_json(pj, {}).get("source_path", ""))) == want:
            return Project(pj.parent)
    return None


def production_spec(pr: Project, clip: str) -> Path | None:
    """The plan the production `final/` file was rendered from (project.json records it); creative
    candidates add newer plan versions that must not be paired with the old render."""
    recorded = pr.data.get("clips", {}).get(clip, {}).get("plan")
    if recorded and Path(recorded).exists():
        return Path(recorded)
    specs = sorted(pr.path.joinpath("edit_specs").glob(f"{clip}.v*.editplan.json"),
                   key=lambda p: int(re.search(r"\.v(\d+)\.", p.name).group(1)))
    return specs[-1] if specs else None


def evaluate_existing(book_path: Path, only: set[str] | None = None, models=None) -> list[dict]:
    """Score the finals that already exist, from cached transcripts and shots. Renders nothing, writes
    only `<project>/creative/<clip>.review.json`."""
    book = load_yaml(book_path)
    fmt = book.get("format", "gaming")
    cfg = load_config(fmt)
    projects, words, shots = {}, {}, {}
    for key, path in book["sources"].items():
        pr = find_project(fmt, Path(path))
        if pr is None:
            raise FileNotFoundError(f"no analysed project for source '{key}' ({path}); run the reel book first")
        projects[key] = pr
        raw = read_json(pr.path / "analysis" / "transcript.json", {}).get("words", [])
        raw = apply_fixes(raw, [x for x in book.get("transcript_fixes", []) if x["source"] == key])
        words[key] = apply_word_fixes(raw, book.get("word_fixes", {}))
        shots[key] = [tuple(x) for x in read_json(pr.path / "analysis" / "shots.json", [])]
    out = []
    for reel in book["reels"]:
        if only and reel["id"] not in only:
            continue
        pr = projects[reel["parts"][0][0]]
        spec = production_spec(pr, reel["id"])
        video = pr.path / "final" / f"{reel['id']}.mp4"
        if spec is None or not video.exists():
            out.append({"clip_id": reel["id"], "skipped": "no plan or final render in the project"})
            continue
        plan = EditPlan.model_validate(read_json(spec))
        policy = {**book.get("creative", {}), **reel.get("creative", {})}
        facts = ClipFacts(plan, words, shots, video, policy, cfg)
        results, info = evaluate(facts, models)
        stored = read_json(pr.path / "qc" / f"{reel['id']}.json", [])
        qc_fail = [f"{c['name']}: {c['message']}" for c in stored if c["status"] == "FAIL"]
        report = build_report(reel["id"], plan.plan_version, results, cfg, qc_failures=qc_fail,
                              compliance_failures=_failures(compliance_checks(plan, book, policy)), evaluator=info)
        atomic_write_json(pr.sub("creative") / f"{reel['id']}.review.json", json.loads(report.model_dump_json()))
        out.append({"clip_id": reel["id"], "decision": report.decision, "overall": report.overall,
                    "scores": report.scores(), "reasons": report.reasons})
    return out


def stored_report(book_path: Path, clip: str) -> CreativeReport | None:
    book = load_yaml(book_path)
    reel = next(r for r in book["reels"] if r["id"] == clip)
    pr = find_project(book.get("format", "gaming"), Path(book["sources"][reel["parts"][0][0]]))
    if pr is None:
        return None
    cdir = pr.path / "creative"
    found = [p for p in (cdir / f"{clip}.creative.json", cdir / f"{clip}.review.json") if p.exists()]
    if not found:
        return None
    newest = max(found, key=lambda p: p.stat().st_mtime)          # a later --review beats an older render record
    data = read_json(newest)
    if newest.name.endswith(".creative.json"):
        return CreativeRecord.model_validate(data).attempts[-1].report
    return CreativeReport.model_validate(data)


def render_text(report: CreativeReport) -> str:
    lines = [f"{report.clip_id} v{report.plan_version}: {report.decision}  overall "
             f"{report.overall if report.overall is not None else 'n/a'}/5"
             f"{'' if report.complete else '  (INCOMPLETE: not every component could be evaluated)'}"]
    for name, r in report.categories.items():
        s = "n/a" if r.status == "not_applicable" else ("-" if r.score is None else f"{r.score:.1f}")
        lines.append(f"  {name:<9}{s:>4}  {r.status}" + (f"  [{'; '.join(r.unavailable)}]" if r.unavailable else ""))
    lines.append("  gates: " + ", ".join(f"{g.name} {'ok' if g.ok else 'FAIL'}" for g in report.gates))
    for fd in report.findings():
        when = f" {fd.start_time:.1f}-{fd.end_time:.1f}s" if fd.start_time is not None and fd.end_time is not None else ""
        lines.append(f"  [{fd.severity}/{fd.basis} {fd.confidence:.2f}] {fd.category}{when}: {fd.issue}")
        lines.append(f"      evidence: {fd.evidence}")
        lines.append(f"      fix: {fd.recommended_fix}" + (f"  (action: {fd.edit_action})" if fd.edit_action != "none" else ""))
    lines.append("  why: " + "; ".join(report.reasons))
    return "\n".join(lines)

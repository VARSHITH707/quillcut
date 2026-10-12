"""Run the evaluators, fold in model judgement, aggregate to a weighted score and decide.

Fail-safe rules, all enforced here:
  * an evaluator that raises becomes an `unavailable` category, never a silent pass;
  * a model that is down leaves its components `unavailable`, and the clip can then at best be
    NEEDS_MODEL, never APPROVED;
  * the weighted average cannot hide a weak gate category or any critical finding;
  * technical QC and compliance are separate gates that must both pass.
"""
from __future__ import annotations

from ..qc import Check
from .checks import EVALUATORS, MODEL_COMPONENTS, _finding, category_score
from .facts import ClipFacts, content_tokens
from .llm import Models, ModelUnavailable, narration_between
from .schema import CREATIVE_VERSION, CategoryResult, CreativeReport, Finding, Gate

TEXT_CATEGORIES = ("pacing", "story", "ending", "fidelity")      # what the text model can add evidence to


def _model_finding(f: ClipFacts, cat: str, severity, issue: str, evidence: str, fix: str, score: float,
                   conf: float = 0.6, **kw) -> Finding:
    cap = f.cc["llm"]["model_confidence_cap"]
    return _finding(f, cat, severity, issue, evidence, fix, score=score, conf=min(conf, cap),
                    **kw).model_copy(update={"basis": "model"})


def _rescore(f: ClipFacts, r: CategoryResult) -> None:
    """Recompute a category's score after model findings were added (a no-op for a category with nothing)."""
    if r.findings or r.score is not None:
        r.score = category_score(f, r.findings)


def _call(judge, f: ClipFacts, why: str | None):
    if judge is None:
        return None, why or "service not available"
    try:
        return judge(f), None
    except ModelUnavailable as exc:
        return None, str(exc)
    except Exception as exc:                                   # a broken model call is still a missing judgement
        return None, f"{type(exc).__name__}: {exc}"


def _missing(results: dict[str, CategoryResult], cats, why: str) -> None:
    for cat in cats:
        if results[cat].status != "not_applicable":
            results[cat].unavailable.append(f"{MODEL_COMPONENTS[cat]}: {why}")


def _merge_text(results, f: ClipFacts, j) -> None:
    """The text model only adds evidence-bearing findings; it never sets or raises a score on its own."""
    def add(cat, *findings):
        results[cat].findings += list(findings)

    if j.context_missing.yes:
        add("story", _model_finding(f, "story", "major", "Context is missing for a standalone viewer.",
                                    f"unexplained: \"{j.context_missing.quote}\" ({j.context_missing.reason})",
                                    "Start earlier or add the explaining line.", 3.0, action="manual"))
    topic = content_tokens(f.hook_text + " " + (f.plan.moment.get("topic") or ""))
    for s in j.off_topic:
        key, a = f.clip_to_source(s.start + 0.01)
        _, b = f.clip_to_source(s.end - 0.01)
        # a 7B model's "this is off-topic" is advisory unless an objective signal agrees: the stretch must
        # share no terms with the headline or topic. Without that it is shown, never scored, never cut.
        corroborated = not (topic & content_tokens(narration_between(f, s.start, s.end)))
        add("pacing", _model_finding(
            f, "pacing", "major" if s.end - s.start > 6 else "minor", "Unnecessary footage.",
            f"\"{s.quote}\" ({s.start:.1f}-{s.end:.1f}s): {s.reason}"
            + ("" if corroborated else " [advisory: the stretch still mentions the headline's subject]"),
            "Remove this section if the campaign allows an edit there.", 3.5, action="cut_span",
            conf=0.6 if corroborated else 0.45, start=s.start, end=s.end, span_src=[round(a, 3), round(b, 3)],
            source=key))
    if j.ends_mid_thought.yes:
        add("ending", _model_finding(f, "ending", "major", "The narration stops mid-thought.",
                                     f"\"{j.ends_mid_thought.quote}\": {j.ends_mid_thought.reason}",
                                     "End on a complete sentence.", 3.0, action="manual",
                                     start=max(0.0, f.body_end - 4), end=f.body_end))
    for c in j.unsupported_claims:
        add("fidelity", _model_finding(f, "fidelity", "major", "A claim is not supported by the source.",
                                       f"\"{c.text}\" ({c.where}): {c.reason}", "Reword with what the speaker says.",
                                       3.0, action="manual", conf=0.7, start=0, end=3))
    if j.meaning_distorted.yes:
        add("fidelity", _model_finding(f, "fidelity", "major", "The edit may distort the speaker.",
                                       f"\"{j.meaning_distorted.quote}\": {j.meaning_distorted.reason}",
                                       "Review the joins.", 3.0, action="manual", conf=0.6))
    for x in results["cta"].findings:
        if x.edit_action == "add_question_overlay" and not x.params.get("question") and j.suggested_question:
            x.params["question"] = j.suggested_question
    for cat in TEXT_CATEGORIES:
        if results[cat].status != "not_applicable":
            _rescore(f, results[cat])


def _merge_vision(results, f: ClipFacts, v, rel=None) -> None:
    """The vision model supplies descriptions and advisory flags only. 'Is a subject visible' is measured from
    the pixels in checks.visual; picture-vs-narration problems need a quote and come from a separate text pass."""
    r = results["visual"]
    if rel is not None and rel.frames:
        many = len(rel.frames) >= 2                      # one frame is a hint, two is a pattern
        r.findings.append(_model_finding(
            f, "visual", "major" if many else "minor", "The footage does not support the narration.",
            "; ".join(f"{v.times[x.frame - 1]:.1f}s picture \"{v.frames[x.frame - 1].describe}\" vs "
                      f"\"{x.quote}\"" for x in rel.frames[:3]),
            "Swap the cutaway for footage that shows what is being said.", 3.0 if many else 4.0, action="manual",
            conf=0.6 if many else 0.45, start=v.times[rel.frames[0].frame - 1], end=v.times[rel.frames[-1].frame - 1] + 0.5))
    covers = [(t, x) for t, x in zip(v.times, v.frames) if x.caption_covers_face]
    if covers:                                           # advisory: the face detector is the objective check
        results["captions"].findings.append(_model_finding(
            f, "captions", "minor", "A caption may cover a face.", "; ".join(f"{t:.1f}s: {x.describe}" for t, x in covers[:3]),
            "Move the caption lower.", 4.0, action="move_captions", conf=0.45, start=covers[0][0], end=covers[-1][0] + 0.5))
    _rescore(f, r)


def evaluate(f: ClipFacts, models: Models | None = None) -> tuple[dict[str, CategoryResult], dict]:
    results: dict[str, CategoryResult] = {}
    for name, fn in EVALUATORS.items():
        try:
            results[name] = fn(f)
        except Exception as exc:                                # never approve on an evaluator error
            results[name] = CategoryResult(category=name, status="unavailable",
                                           unavailable=[f"evaluator error: {type(exc).__name__}: {exc}"])
    models = models or Models(reasons={"text": "no model configured", "vision": "no model configured"})
    j, why = _call(models.text, f, models.reasons.get("text"))
    if j is None:
        _missing(results, TEXT_CATEGORIES, why)
    else:
        _merge_text(results, f, j)
    if results["visual"].status != "unavailable":
        v, why = _call(models.vision, f, models.reasons.get("vision"))
        if v is None:
            _missing(results, ["visual"], why)
        else:
            vv, rel = v if isinstance(v, tuple) else (v, None)
            _merge_vision(results, f, vv, rel)
    info = {"creative": CREATIVE_VERSION, **models.names,
            "model_statements_rejected": getattr(j, "rejected", 0) if j else None}
    return results, info


def aggregate(results: dict[str, CategoryResult], cfg: dict) -> tuple[float | None, bool]:
    w = cfg["creative"]["weights"]
    scored = {k: r for k, r in results.items() if r.status != "not_applicable" and r.score is not None}
    total = sum(w[k] for k in scored)
    overall = round(sum(w[k] * r.score for k, r in scored.items()) / total, 2) if total else None
    complete = all(r.status == "not_applicable" or (r.score is not None and not r.unavailable)
                   for r in results.values())
    return overall, complete


def compliance_checks(plan, book: dict, policy: dict) -> list[Check]:
    status = plan.rights.get("status", "unknown")
    checks = [Check("rights", "PASS" if status != "unknown" else "FAIL", f"rights status '{status}'")]
    if book.get("music") is False:
        checks.append(Check("no_added_music", "PASS" if plan.audio.music is None else "FAIL",
                            "campaign forbids added music"))
    if book.get("sfx") is False:
        checks.append(Check("no_added_sfx", "PASS" if not plan.audio.sfx else "FAIL", "campaign forbids sound effects"))
    headline = next((c.text for c in plan.text_cards if c.kind == "hook"), "").lower()
    for term in policy.get("hook_must_include", []):
        checks.append(Check(f"headline_has_{term}", "PASS" if term.lower() in headline else "FAIL",
                            f"headline must name {term}"))
    if book.get("end_card"):
        checks.append(Check("end_card", "PASS" if any(s.id == "end" for s in plan.segments) else "FAIL",
                            "campaign end card present"))
    if book.get("logo"):
        checks.append(Check("brand_logo", "PASS" if plan.logo else "FAIL", "campaign requires a brand logo or watermark"))
    return checks


def build_report(clip_id: str, version: int, results: dict[str, CategoryResult], cfg: dict, *,
                 qc_failures: list[str], compliance_failures: list[str], evaluator: dict) -> CreativeReport:
    cc = cfg["creative"]
    overall, complete = aggregate(results, cfg)
    gates: list[Gate] = []

    def gate(name, ok, detail):
        gates.append(Gate(name=name, ok=bool(ok), detail=detail))

    gate("overall", overall is not None and overall >= cc["pass_overall"],
         f"{overall} (need {cc['pass_overall']})" if overall is not None else "nothing could be scored")
    for cat, need in cc["gates"].items():
        s = results[cat].score
        gate(cat, s is not None and s >= need, f"{s} (need {need})" if s is not None else "not evaluated")
    crit = [x for r in results.values() for x in r.findings
            if x.severity == "critical" and x.confidence >= cc["count_min_confidence"]]
    gate("no_critical_findings", not crit, "; ".join(f"{x.category}: {x.issue}" for x in crit) or "none")
    gate("technical_qc", not qc_failures, "; ".join(qc_failures) or "all checks pass")
    gate("compliance", not compliance_failures, "; ".join(compliance_failures) or "all checks pass")
    pending = [f"{k}: {u}" for k, r in results.items() for u in r.unavailable]
    gate("complete", complete, "; ".join(pending) or "every applicable component evaluated")

    failed = [g for g in gates if not g.ok and g.name != "complete"]
    if failed:
        decision, reasons = "MANUAL_REVIEW", [f"{g.name}: {g.detail}" for g in failed]
    elif not complete:
        decision, reasons = "NEEDS_MODEL", [f"could not evaluate {p}" for p in pending]
    else:
        decision, reasons = "APPROVED", ["all gates passed"]
    return CreativeReport(clip_id=clip_id, plan_version=version, categories=results, overall=overall,
                          complete=complete, gates=gates, decision=decision, reasons=reasons, evaluator=evaluator)

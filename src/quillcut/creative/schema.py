"""Creative-quality report models: evidence-first findings, 1-5 scores, an explicit decision.

Creative evaluation is deliberately separate from technical QC (`qc.py`): QC asks "is the file
valid?", this package asks "is the edit good?". A finding that cannot name its evidence does not
exist; a score that could not be measured is `None`, never a guess.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CREATIVE_VERSION = "1.0"
CATEGORIES = ("hook", "pacing", "story", "visual", "audio", "captions", "ending",
              "cta", "rhythm", "loop", "fidelity")
Severity = Literal["info", "minor", "major", "critical"]
SEVERITY_RANK = {"info": 0, "minor": 1, "major": 2, "critical": 3}
Basis = Literal["objective", "model"]
Status = Literal["scored", "not_applicable", "unavailable"]
Decision = Literal["APPROVED", "MANUAL_REVIEW", "NEEDS_MODEL"]
EditAction = Literal["none", "trim_end", "extend_end", "trim_start", "cut_span",
                     "move_captions", "add_question_overlay", "retime_cta", "reselect", "manual"]


class Finding(BaseModel):
    category: str
    severity: Severity
    score: float = Field(ge=1, le=5)        # the category score this issue alone would justify
    issue: str
    evidence: str
    recommended_fix: str
    edit_action: EditAction = "none"
    confidence: float = Field(ge=0, le=1)
    basis: Basis = "objective"
    start_time: float | None = None         # clip time
    end_time: float | None = None
    params: dict = Field(default_factory=dict)   # action arguments (source times, part index, ...)


class CategoryResult(BaseModel):
    category: str
    status: Status
    score: float | None = None              # None = nothing measurable
    findings: list[Finding] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)   # model components that could not run, with reason
    note: str = ""


class Gate(BaseModel):
    name: str
    ok: bool
    detail: str


class CreativeReport(BaseModel):
    clip_id: str
    plan_version: int
    categories: dict[str, CategoryResult]
    overall: float | None                   # weighted over categories that produced a score
    complete: bool                          # False when any applicable component could not be evaluated
    gates: list[Gate]
    decision: Decision
    reasons: list[str]
    evaluator: dict = Field(default_factory=dict)       # version + model names actually used

    def findings(self, min_confidence: float = 0.0) -> list[Finding]:
        out = [f for r in self.categories.values() for f in r.findings if f.confidence >= min_confidence]
        return sorted(out, key=lambda f: SEVERITY_RANK[f.severity], reverse=True)

    def scores(self) -> dict[str, float | None]:
        return {k: r.score for k, r in self.categories.items()}


class Attempt(BaseModel):
    plan_version: int
    overall: float | None
    scores: dict[str, float | None]
    decision: Decision
    technical_qc: str                       # PASS | FAIL
    qc_failures: list[str] = Field(default_factory=list)
    compliance: str                         # PASS | FAIL
    compliance_failures: list[str] = Field(default_factory=list)
    video: str | None = None
    render_seconds: float | None = None       # render + technical QC of the candidate
    evaluate_seconds: float | None = None     # all creative evaluators incl. any model calls
    applied: list[dict] = Field(default_factory=list)     # repairs that produced the NEXT attempt
    skipped: list[dict] = Field(default_factory=list)     # proposed but not applied, with the reason
    conflicts: list[dict] = Field(default_factory=list)   # reel-book constraints that blocked a fix
    report: CreativeReport | None = None


class CreativeRecord(BaseModel):
    clip_id: str
    attempts: list[Attempt] = Field(default_factory=list)
    final_decision: Decision = "MANUAL_REVIEW"
    final_plan_version: int | None = None
    retry_count: int = 0
    before_overall: float | None = None
    after_overall: float | None = None
    published: str | None = None
    evaluator: dict = Field(default_factory=dict)

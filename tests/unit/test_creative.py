"""Creative evaluation + repair. No Ollama, no OpenShorts, no GPU: model replies are mocked and
media is synthetic. The BOXABL regression is rebuilt from scratch, not hard-coded into the engine."""
import copy
import json
import subprocess
from pathlib import Path

import pytest

from quillcut import reels
from quillcut.config import load_config
from quillcut.creative import checks, loop, llm, repair, score
from quillcut.creative.facts import ClipFacts, end_candidates, split_sentences
from quillcut.creative.schema import CATEGORIES, Attempt, CategoryResult, CreativeRecord, Finding
from quillcut.formats import load_style
from quillcut.scripts import SourceCtx
from quillcut.storage import Project
from quillcut.types import EditPlan, SourceRef

CFG = load_config("gaming")
STYLE = load_style("gaming", "gaming_bold")


def _speech(lines, prob=0.95):
    out = []
    for start, text in lines:
        t = start
        for tok in text.split():
            out.append({"word": tok, "start": round(t, 3), "end": round(t + 0.38, 3), "probability": prob})
            t += 0.45
    return out


LINES = [(1.0, "Hello and welcome to the show."), (4.2, "Welcome to the casita tour today."), (8.0, "The home folds flat for shipping."),
         (12.0, "It unfolds in under one hour."), (16.0, "Inside you get a full kitchen."),
         (20.0, "The ceilings are nine feet tall."), (24.0, "Everything ships ready to use."),
         (28.0, "And the price starts quite low."), (32.0, "It takes less than"), (34.1, "an hour to build.")]
VARIED = ["Welcome to the casita tour today.", "The home folds flat for shipping.",
          "It unfolds in under one hour.", "Inside you get a full kitchen.", "The ceilings are nine feet tall.",
          "Everything ships ready to use.", "The price starts quite low today.", "Windows fill the room with light.",
          "The bathroom has a tall shower.", "Storage hides under the bed.", "That is the whole casita tour."]
SHOTS = [(0, 4), (4, 9.5), (9.5, 15), (15, 22), (22, 31), (31, 40), (40, 52), (52, 60)]


def _src(words=None, shots=SHOTS):
    ref = SourceRef(path="x.mp4", sha1="0" * 40, width=3840, height=2160, fps="60/1", duration=60.0,
                    has_audio=True, content_box=(3840, 2160, 0, 0))
    ws = _speech(LINES) if words is None else words
    return reels.ShotSource(SourceCtx(key="a", ref=ref, t_peak=0, lo=0, hi=60, words=ws), list(shots))


UNPROTECTED = {"exact_start": True, "exact_end": True}       # snapped to word edges, free for repairs to move


def _reel(to=34.0, frm=3.5, opts=None):
    opts = {"exact": True} if opts is None else opts          # `exact` = hand-checked cut points, protected
    return {"id": "t01", "hook": "Welcome to the Casita Tour", "topic": "casita tour and price",
            "parts": [["a", frm, to, dict(opts)]], "music": {"track": "none"}}


def _build_fn(src, book):
    return lambda st: reels.build_reel(st.reel, book, {"a": src}, CFG,
                                       _style(st), {}, "p")


def _style(st):
    s = copy.deepcopy(STYLE)
    if st.caption_y:
        s["speech"]["y"] = st.caption_y
    return s


def _facts(reel=None, book=None, policy=None, video=None, src=None):
    src, reel = src or _src(), reel or _reel()
    book = book or {"key": "t", "music": False, "rights_record": {"status": "client_licensed"}}
    plan = reels.build_reel(reel, book, {"a": src}, CFG, STYLE, {}, "p")
    return ClipFacts(plan, {"a": src.ctx.words}, {"a": src.shots}, video, policy or {}, CFG), src, book


def _results(**scores):
    return {c: CategoryResult(category=c, status="scored", score=scores.get(c, 5.0)) for c in CATEGORIES}


def _report(results, qc=(), comp=()):
    return score.build_report("t01", 1, results, CFG, qc_failures=list(qc), compliance_failures=list(comp),
                              evaluator={})


# ------------------------------------------------------------------ scoring

def test_weighted_score_matches_spec_weights():
    vals = dict(hook=5, pacing=4, story=5, visual=4, audio=5, captions=5, ending=5, cta=4, rhythm=4, loop=4,
                fidelity=5)
    overall, complete = score.aggregate(_results(**vals), CFG)
    assert overall == 4.63 and complete                    # 463 / 100


def test_not_applicable_category_renormalises_weights():
    r = _results(hook=5, pacing=4, story=5, visual=4, audio=5, captions=5, ending=5, cta=4, rhythm=4, loop=4,
                 fidelity=5)
    r["cta"] = CategoryResult(category="cta", status="not_applicable")
    overall, complete = score.aggregate(r, CFG)
    assert overall == round(443 / 95, 2) and complete


def test_all_gates_passing_approves():
    assert _report(_results()).decision == "APPROVED"


def test_strong_average_cannot_hide_a_weak_gate_or_critical_finding():
    weak_ending = _report(_results(ending=3.0))
    assert weak_ending.overall > 4.5 and weak_ending.decision == "MANUAL_REVIEW"
    assert any(g.name == "ending" and not g.ok for g in weak_ending.gates)
    r = _results()
    r["audio"].findings.append(Finding(category="audio", severity="critical", score=1, issue="x", evidence="y",
                                       recommended_fix="z", confidence=0.9))
    assert _report(r).decision == "MANUAL_REVIEW"


def test_unavailable_component_blocks_approval_but_is_not_a_failure():
    r = _results()
    r["loop"] = CategoryResult(category="loop", status="scored", score=None,
                               unavailable=["loop_and_share: ollama not reachable"])
    rep = _report(r)
    assert rep.decision == "NEEDS_MODEL" and not rep.complete


def test_technical_qc_and_compliance_are_separate_hard_gates():
    assert _report(_results(), qc=["loudness: -20"]).decision == "MANUAL_REVIEW"
    assert _report(_results(), comp=["rights: unknown"]).decision == "MANUAL_REVIEW"


def test_evaluator_exception_fails_safe(monkeypatch):
    facts, *_ = _facts()
    monkeypatch.setitem(checks.EVALUATORS, "hook", lambda f: 1 / 0)
    results, _ = score.evaluate(facts, None)
    assert results["hook"].status == "unavailable" and "ZeroDivisionError" in results["hook"].unavailable[0]
    assert score.build_report("t", 1, results, CFG, qc_failures=[], compliance_failures=[],
                              evaluator={}).decision != "APPROVED"


def test_models_down_leaves_model_only_components_unavailable_never_scored():
    facts, *_ = _facts()
    results, _ = score.evaluate(facts, llm.Models(reasons={"text": "ollama not reachable", "vision": "none"}))
    assert "ollama not reachable" in results["story"].unavailable[0]          # a model-only component stays unavailable
    assert not results["hook"].unavailable and not results["loop"].unavailable   # measured by proxies, no model needed
    assert results["loop"].score is not None and "proxy" in results["loop"].note


# ------------------------------------------------------------------ ending / captions / timeline

def test_mid_sentence_ending_is_critical_with_timestamps_and_candidates():
    facts, *_ = _facts()
    res = checks.ending(facts)
    f = res.findings[0]
    assert f.severity == "critical" and f.score == 1 and res.score == 1
    assert f.start_time >= facts.body_end - 4.01 and f.end_time == pytest.approx(facts.body_end)
    assert "than" in f.evidence and f.params["extend"] == pytest.approx(35.83, abs=0.05)
    assert f.edit_action == "extend_end"


def test_complete_sentence_ending_scores_clean():
    facts, *_ = _facts(reel=_reel(to=36.0))
    assert checks.ending(facts).score >= 4.5


def test_end_candidates_prefers_a_boundary_on_a_shot_cut():
    words = _speech([(10.0, "One thing."), (13.0, "Another thing here."), (14.4, "Done now."), (40.0, "Far.")])
    c = end_candidates(words, 12.0, 0.0, 60.0, cuts=[], trim_max=12, extend_max=8)
    assert c["trim"] == pytest.approx(10.83, abs=0.01) and c["extend"] == pytest.approx(14.28, abs=0.01)
    c2 = end_candidates(words, 12.0, 0.0, 60.0, cuts=[15.3], extend_max=8)      # scene-complete beats nearest
    assert c2["extend"] == pytest.approx(15.23, abs=0.01)


def test_incomplete_final_caption_is_flagged():
    words = _speech(LINES[:-1] + [(34.1, "an hour to build.")])
    words[-1]["probability"] = 0.3                        # final word is dropped from captions on purpose
    facts, *_ = _facts(reel=_reel(to=36.0), src=_src(words))
    res = checks.captions(facts)
    assert any(f.issue == "The final caption is incomplete." for f in res.findings)
    assert res.score < 4


def test_complete_captions_pass():
    facts, *_ = _facts(reel=_reel(to=36.0))
    assert checks.captions(facts).score >= 4.5


def test_timestamp_conversion_round_trips_and_follows_a_trim():
    reel = _reel(34.0, 1.0, UNPROTECTED)
    facts, src, book = _facts(reel=reel)
    assert facts.clip_to_source(0.0) == ("a", pytest.approx(0.95, abs=0.01))
    assert facts.clip_to_source(10.0)[1] == pytest.approx(10.95, abs=0.01)
    assert facts.source_to_clip("a", 10.95) == [pytest.approx(10.0, abs=0.01)]
    state = repair.State(reel=reel)
    new = repair._fix_end(state, checks.ending(facts).findings[0], facts, _build_fn(src, book))
    plan2 = _build_fn(src, book)(new)
    f2 = ClipFacts(plan2, {"a": src.ctx.words}, {"a": src.shots}, None, {}, CFG)
    assert f2.clip_to_source(10.0)[1] == pytest.approx(10.95, abs=0.01)    # earlier timestamps unchanged by a tail edit
    assert f2.body_end > facts.body_end and f2.words[-1]["word"] == "build."
    assert f2.source_to_clip("a", 35.83)[0] == pytest.approx(f2.body_end - 0.12, abs=0.05)


# ------------------------------------------------------------------ repair

def _report_for(facts):
    results, _ = score.evaluate(facts, None)
    return score.build_report("t01", 1, results, CFG, qc_failures=[], compliance_failures=[], evaluator={})


def test_repair_extends_to_a_complete_sentence_and_keeps_the_book_intact():
    reel = _reel(34.0, 1.0, UNPROTECTED)
    reel["parts"][0][3]["broll"] = [[10.0, 3.0, "a", 40.0]]
    facts, src, book = _facts(reel=reel)
    out = repair.plan_repairs(_report_for(facts), facts, repair.State(reel=reel), _build_fn(src, book))
    assert [a["action"] for a in out.applied][:1] == ["extend_end"]
    part = out.state.reel["parts"][0]
    assert part[1] == 1.0 and part[3]["broll"] == [[10.0, 3.0, "a", 40.0]]     # opening and b-roll preserved
    assert part[2] == pytest.approx(35.83, abs=0.05)
    plan2 = _build_fn(src, book)(out.state)
    assert plan2.audio.music is None and not plan2.audio.sfx                     # campaign: no added audio
    f2 = ClipFacts(plan2, {"a": src.ctx.words}, {"a": src.shots}, None, {}, CFG)
    assert checks.ending(f2).score >= 4.5
    assert reel["parts"][0][2] == 34.0 and "cta" not in reel         # the caller's reel is never mutated


@pytest.mark.parametrize("flag", ["essential", "exact"])
def test_protected_part_is_reported_as_conflict_not_edited(flag):
    reel = _reel(); reel["parts"][0][3] = {flag: True}
    facts, src, book = _facts(reel=reel)
    state = repair.State(reel=reel)
    out = repair.plan_repairs(_report_for(facts), facts, state, _build_fn(src, book))
    assert not out.applied and out.state.reel == reel
    assert out.conflicts and "smallest acceptable correction" in out.conflicts[0]["reason"]


def test_low_confidence_and_manual_findings_are_never_applied():
    facts, src, book = _facts()
    low = Finding(category="ending", severity="major", score=2, issue="i", evidence="e", recommended_fix="f",
                  edit_action="extend_end", confidence=0.6, params={"part": 0, "src_out": 34.0, "extend": 35.83})
    manual = low.model_copy(update={"edit_action": "manual", "confidence": 0.9})
    r = _results(); r["ending"].findings += [low, manual]
    out = repair.plan_repairs(_report(r), facts, repair.State(reel=_reel()), _build_fn(src, book))
    assert not out.applied and len(out.skipped) == 2


def test_jump_cuts_need_explicit_permission():
    rep_words = _speech([(1.0, "Hello and welcome to the show."), (4.2, "Welcome to the casita tour today."),
                         (8.0, "The home folds flat for shipping."),
                         (12.0, "The home folds flat for shipping."), (16.0, "Inside you get a full kitchen."),
                         (20.0, "The ceilings are nine feet tall."), (24.0, "Everything ships ready to use."),
                         (28.0, "And the price starts quite low."), (32.0, "It takes an hour."),
                         (37.0, "That is all, thanks for watching.")])
    reel = _reel(41.0, 1.0, UNPROTECTED)
    facts, src, book = _facts(reel=reel, src=_src(rep_words))
    res = checks.pacing(facts)
    rep = [f for f in res.findings if f.issue == "Dialogue repeats."]
    assert rep and rep[0].edit_action == "cut_span"
    r = _results(); r["pacing"].findings += rep
    out = repair.plan_repairs(_report(r), facts, repair.State(reel=reel), _build_fn(src, book))
    assert not out.applied and "jump cuts" in out.skipped[0]["reason"]
    facts.policy["allow_jump_cuts"] = True
    out = repair.plan_repairs(_report(r), facts, repair.State(reel=reel), _build_fn(src, book))
    assert [a["action"] for a in out.applied] == ["cut_span"] and len(out.state.reel["parts"]) == 2


def test_question_overlay_only_when_the_campaign_allows_it():
    q = "Would you live in a modular home?"
    for mode, expect in (("off", None), ("suggest", None), ("auto", q)):
        reel = _reel(to=36.0)
        facts, src, book = _facts(reel=reel, policy={"cta": mode, "cta_questions": [q]})
        res = checks.cta(facts)
        if mode == "off":
            assert res.status == "not_applicable"
            continue
        r = _results(); r["cta"] = res
        out = repair.plan_repairs(_report(r), facts, repair.State(reel=reel), _build_fn(src, book))
        assert (out.state.reel.get("cta") or {}).get("text") == expect
    plan = _build_fn(src, book)(out.state)
    labels = [c for c in plan.text_cards if c.kind == "label"]
    hooks = [c for c in plan.text_cards if c.kind == "hook"]
    assert labels[0].text.lower() == q.lower() and labels[0].end - labels[0].start >= 2.0   # style may change case
    assert hooks[0].end <= labels[0].start + 1e-6                     # the question replaces the headline, no overlap
    f2 = ClipFacts(plan, {"a": src.ctx.words}, {"a": src.shots}, None, {"cta": "auto"}, CFG)
    assert not [f for f in checks.captions(f2).findings if f.issue == "Two text cards overlap."]
    assert checks.cta(f2).score >= 4.5


def test_model_question_must_be_short_and_made_of_clip_subjects():
    facts, *_ = _facts()
    assert llm.safe_question("Would you live in a casita?", facts, 9)
    assert llm.safe_question("Would you buy Boxabl stock in 2030?", facts, 9) is None
    assert llm.safe_question("What do you think about it", facts, 9) is None


# ------------------------------------------------------------------ model output is verified, not trusted

class FakeOllama:
    def __init__(self, reply):
        self.reply = reply

    def chat(self, model, user, images=None, **kw):
        return self.reply


def _chk(yes=False, quote="", reason=""):
    return {"yes": yes, "quote": quote, "reason": reason}


def _reply(**over):
    """A clean reply: no problems found, one safe question, nothing off-topic."""
    base = {"context_missing": _chk(), "ends_mid_thought": _chk(), "meaning_distorted": _chk(),
            "suggested_question": "Would you live in a casita?", "off_topic": [], "unsupported_claims": []}
    return base | over


def test_a_problem_report_without_a_verifiable_quote_is_not_believed():
    facts, *_ = _facts(reel=_reel(to=36.0))
    fake = llm.judge_text(FakeOllama(_reply(context_missing=_chk(True, "the blue widget", "x"),
                                            meaning_distorted=_chk(True, "", "no quote at all"))), "m", facts)
    assert not fake.context_missing.yes and not fake.meaning_distorted.yes and fake.rejected == 2
    real = llm.judge_text(FakeOllama(_reply(context_missing=_chk(True, "the price starts", "unexplained"))), "m", facts)
    assert real.context_missing.yes and real.rejected == 0
    # 'ends mid-thought' must quote the END of the narration, not something from the middle of it
    wrong_place = llm.judge_text(FakeOllama(_reply(ends_mid_thought=_chk(True, "the home folds flat"))), "m", facts)
    assert not wrong_place.ends_mid_thought.yes
    right_place = llm.judge_text(FakeOllama(_reply(ends_mid_thought=_chk(True, "an hour to build"))), "m", facts)
    assert right_place.ends_mid_thought.yes


def test_bare_booleans_from_a_small_model_are_tolerated_but_never_believed():
    facts, *_ = _facts(reel=_reel(to=36.0))
    j = llm.judge_text(FakeOllama(_reply(context_missing=True, ends_mid_thought=False)), "m", facts)
    assert not j.context_missing.yes and j.rejected == 1                    # a bare 'true' has no evidence


def test_invented_quotes_claims_and_questions_are_dropped_and_counted():
    facts, *_ = _facts(reel=_reel(to=36.0))
    reply = _reply(off_topic=[{"start": 5, "end": 9, "quote": "totally invented sentence here", "reason": "x"},
                              {"start": 8.0, "end": 11.0, "quote": "the home folds flat", "reason": "repeat"}],
                   unsupported_claims=[{"text": "Welcome to the Casita Tour", "where": "hook", "reason": "x"},
                                       {"text": "Sold 5000 homes", "where": "hook", "reason": "y"}],
                   suggested_question="Do you want one for 5 dollars?")
    j = llm.judge_text(FakeOllama(reply), "m", facts)
    assert [s.quote for s in j.off_topic] == ["the home folds flat"]
    assert j.unsupported_claims == [] and j.suggested_question == "" and j.rejected == 4   # span, 2 claims, question


def test_the_text_model_adds_evidence_but_can_never_raise_or_set_a_score():
    facts, *_ = _facts(reel=_reel(to=36.0), policy={"cta": "suggest"})
    clean = llm.TextJudgement.model_validate(_reply())
    before, _ = score.evaluate(facts, llm.Models(reasons={"text": "down", "vision": "down"}))
    after, _ = score.evaluate(facts, llm.Models(text=lambda f: clean, reasons={"vision": "down"}))
    for cat in ("hook", "pacing", "story", "ending", "fidelity", "cta", "loop"):
        assert after[cat].score == before[cat].score                          # praise changes nothing
    assert not after["story"].unavailable and before["story"].unavailable      # but the component is now covered
    flagged = llm.TextJudgement.model_validate(_reply(ends_mid_thought=_chk(True, "an hour to build", "stops short")))
    worse, _ = score.evaluate(facts, llm.Models(text=lambda f: flagged, reasons={"vision": "down"}))
    assert worse["ending"].score < after["ending"].score
    assert any(f.basis == "model" and "stops short" in f.evidence for f in worse["ending"].findings)


def test_the_hook_check_separates_a_real_headline_from_generic_and_false_ones():
    """Negative controls from a real run: a 7B judge rated a false headline above the real one. The deterministic
    headline-vs-narration check must not."""
    def hook_for(text):
        reel = _reel(to=36.0); reel["hook"] = text
        facts, *_ = _facts(reel=reel)
        return checks.hook(facts), facts
    good, _ = hook_for("Welcome to the Casita Tour")
    assert not [f for f in good.findings if "does not match" in f.issue]
    for bad_text in ("Watch This", "How We Beat Tesla in 2030", "Elon Musk Buys 500 BOXABL Homes"):
        res, facts = hook_for(bad_text)
        mismatch = [f for f in res.findings if "does not match" in f.issue]
        assert mismatch and mismatch[0].severity == "major" and res.score < 4, bad_text
        assert "never said" in mismatch[0].evidence
    _, facts = hook_for("Elon Musk Buys 500 BOXABL Homes")
    assert any("number that is never said" in f.issue for f in checks.fidelity(facts).findings)      # the false figure


def test_opening_proxy_wants_a_figure_question_or_strong_claim():
    plain, *_ = _facts(reel=_reel(to=36.0))                                   # opens "Welcome to the casita tour today"
    assert any(f.issue == "The narration opens without a hook." for f in checks.hook(plain).findings)
    words = _speech([(4.2, "Why does this home ship in under 60 minutes?")] + LINES[2:])
    grabbing, *_ = _facts(reel=_reel(to=36.0), src=_src(words))
    assert not any(f.issue == "The narration opens without a hook." for f in checks.hook(grabbing).findings)


def test_loop_potential_is_a_transparent_proxy_not_a_model_opinion():
    facts, *_ = _facts(reel=_reel(to=36.0))
    res = checks.loop(facts)
    assert res.score is not None and "proxy score" in res.note and "a concrete figure is stated=no" in res.note
    with_figure, *_ = _facts(reel=_reel(to=36.0), book={"key": "t", "music": False, "rights_record": {"status": "x"},
                                                       "end_card": {"source": "a", "from": 52.0, "to": 54.8}},
                             src=_src(_speech([(4.2, "Prices start at 160 thousand dollars.")] + LINES[2:])))
    better = checks.loop(with_figure)
    assert better.score > res.score and "a closing brand beat exists=yes" in better.note


def test_ollama_status_reports_down_instead_of_raising():
    import httpx
    o = llm.Ollama(CFG["creative"]["llm"], client=httpx.Client(transport=httpx.MockTransport(
        lambda req: (_ for _ in ()).throw(httpx.ConnectError("refused")))))
    st = o.status()
    assert st["up"] is False and "not reachable" in st["reason"]
    with pytest.raises(llm.ModelUnavailable):
        llm.Ollama(CFG["creative"]["llm"], client=httpx.Client(transport=httpx.MockTransport(
            lambda req: httpx.Response(500)))).chat("m", "hi")


# ------------------------------------------------------------------ measurements on synthetic media

def _clip(tmp_path, name, vf="", af="", video="testsrc2=s=270x480:r=30:d=36"):
    out = tmp_path / name
    args = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", "sine=f=300:d=36"]
    if vf:
        args += ["-vf", vf]
    if af:
        args += ["-af", af]
    subprocess.run(args + ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                           "-shortest", str(out)], check=True)
    return out


def test_dead_silence_and_darkness_are_measured_from_the_file(tmp_path):
    silent = _clip(tmp_path, "silent.mp4", af="volume=enable='between(t,10,14)':volume=0")
    facts, *_ = _facts(reel=_reel(to=36.0), video=silent)
    res = checks.audio(facts)
    gap = next(f for f in res.findings if f.issue == "Dead silence.")
    assert gap.severity == "major" and 9.5 < gap.start_time < 10.5 and res.score < 4
    dark = _clip(tmp_path, "dark.mp4", video="color=c=0x0a0a0a:s=270x480:r=30:d=36")
    facts, *_ = _facts(reel=_reel(to=36.0), video=dark)
    assert any(f.issue in ("The picture is very dark.", "The picture is almost black.")
               for f in checks.visual(facts).findings)


def test_missing_render_makes_audio_and_visual_unavailable_not_perfect():
    facts, *_ = _facts()
    assert checks.audio(facts).status == "unavailable" and checks.visual(facts).score is None


def test_clean_synthetic_clip_can_reach_approval_with_a_mocked_model(tmp_path):
    good = _clip(tmp_path, "good.mp4", af="volume=enable='gte(t,34.4)':volume=0",
                 video="gradients=s=270x480:r=30:d=36:speed=0.02")        # moving picture, speech then quiet
    tight = _speech([(1.0 + 3.2 * i, s) for i, s in enumerate(VARIED)])
    src = _src(tight, shots=[(i * 5.0, (i + 1) * 5.0) for i in range(12)])
    facts, *_ = _facts(reel=_reel(36.5, 1.0, UNPROTECTED), video=good, src=src)
    vj = llm.VisionJudgement.model_validate({"frames": [
        {"frame": i, "describe": "a presenter in a factory"} for i in range(1, 7)],
        "times": [3, 9, 15, 21, 27, 33]})
    tj = llm.TextJudgement.model_validate(_reply())
    results, info = score.evaluate(facts, llm.Models(text=lambda f: tj, vision=lambda f: vj,
                                                     names={"text_model": "mock", "vision_model": "mock"}))
    rep = score.build_report("t01", 1, results, CFG, qc_failures=[], compliance_failures=[], evaluator=info)
    assert rep.decision == "APPROVED", (rep.reasons, rep.scores())


# ------------------------------------------------------------------ the retry loop

def _project(tmp_path):
    p = Project(tmp_path / "proj")
    p.data = {}
    return p


def _stub_render(tmp_path, calls):
    def render(plan, project, cfg, fast):
        path = tmp_path / f"v{plan.plan_version}.mp4"
        path.write_bytes(b"x")
        calls.append(plan.plan_version)
        return path, []
    return render


def _bad_eval(facts, models):
    r = _results()
    r["ending"] = CategoryResult(category="ending", status="scored", score=1.0, findings=[Finding(
        category="ending", severity="critical", score=1, issue="cut", evidence="e", recommended_fix="f",
        edit_action="reselect", confidence=0.95)])
    return r, {}


def test_retry_limit_is_three_repairs_then_manual_review(tmp_path):
    reel = _reel(); reel["alternates"] = [[["a", 3.5, 34.0 + i * 0.01, {"exact": True}]] for i in range(6)]
    facts, src, book = _facts(reel=reel)
    calls = []
    out = loop.run_reel(reel, book, {"a": src}, CFG, STYLE, {}, _project(tmp_path), publish_dir=tmp_path / "pub",
                        render=_stub_render(tmp_path, calls), evaluate_fn=_bad_eval)
    assert len(calls) == 4 and out["retries"] == 3 and out["decision"] == "MANUAL_REVIEW"
    assert out["published"] is None and calls == [1, 2, 3, 4]
    rec = json.loads(Path(out["record"]).read_text(encoding="utf-8"))
    assert [len(a["applied"]) for a in rec["attempts"]] == [1, 1, 1, 0]


def test_no_repair_flag_and_nothing_repairable_stop_after_one_render(tmp_path):
    facts, src, book = _facts()
    calls = []
    loop.run_reel(_reel(), book, {"a": src}, CFG, STYLE, {}, _project(tmp_path), publish_dir=tmp_path / "pub",
                  render=_stub_render(tmp_path, calls), evaluate_fn=_bad_eval, repair=False)
    assert calls == [1]
    calls.clear()
    loop.run_reel(_reel(), book, {"a": src}, CFG, STYLE, {}, _project(tmp_path / "b"), publish_dir=tmp_path / "pub",
                  render=_stub_render(tmp_path, calls), evaluate_fn=_bad_eval)      # no alternates -> cannot repair
    assert calls == [1]


def test_approved_candidate_is_published_beside_not_over_production(tmp_path):
    facts, src, book = _facts()
    pub = tmp_path / "pub"
    pub.mkdir()
    prod = pub / "t01.mp4"
    prod.write_bytes(b"production render")
    out = loop.run_reel(_reel(), book, {"a": src}, CFG, STYLE, {}, _project(tmp_path), publish_dir=pub,
                        render=_stub_render(tmp_path, []), evaluate_fn=lambda f, m: (_results(), {"creative": "t"}))
    assert out["decision"] == "APPROVED" and out["retries"] == 0
    assert prod.read_bytes() == b"production render"
    assert Path(out["published"]).parent.name == "creative" and Path(out["published"]).name == "t01_v1.mp4"
    assert (pub / "creative" / "t01.creative.json").exists()


def test_failed_technical_qc_blocks_approval_even_with_perfect_scores(tmp_path):
    from quillcut.qc import Check
    facts, src, book = _facts()

    def render(plan, project, cfg, fast):
        p = tmp_path / "v.mp4"
        p.write_bytes(b"x")
        return p, [Check("loudness", "FAIL", "-20 LUFS")]
    out = loop.run_reel(_reel(), book, {"a": src}, CFG, STYLE, {}, _project(tmp_path), publish_dir=tmp_path / "pub",
                        render=render, evaluate_fn=lambda f, m: (_results(), {}), repair=False)
    assert out["decision"] == "MANUAL_REVIEW" and out["published"] is None


def test_compliance_gates_follow_the_book():
    facts, src, book = _facts(book={"key": "t", "music": False, "creative": {}},
                              policy={"hook_must_include": ["Casita", "BOXABL"]})
    names = {c.name: c.status for c in score.compliance_checks(facts.plan, {"music": False}, facts.policy)}
    assert names["no_added_music"] == "PASS" and names["headline_has_Casita"] == "PASS"
    assert names["headline_has_BOXABL"] == "FAIL"
    unknown = facts.plan.model_copy(update={"rights": {"status": "unknown"}})
    assert score.compliance_checks(unknown, {}, {})[0].status == "FAIL"


# ------------------------------------------------------------------ backward compatibility

def test_old_edit_plans_and_reel_books_still_load_and_build():
    plan_json = _facts()[0].plan.model_dump_json()
    assert EditPlan.model_validate_json(plan_json).schema_version == "2.0"
    real = list(Path(__file__).resolve().parents[2].glob("formats/*/projects/*/edit_specs/*.editplan.json"))
    for path in real[:25]:                                   # production plans written before this feature
        EditPlan.model_validate_json(path.read_text(encoding="utf-8"))
    facts, *_ = _facts()                                     # a reel without `cta` plans exactly as before
    assert not [c for c in facts.plan.text_cards if c.kind == "label"]


def test_sentence_splitter_uses_punctuation_and_gaps():
    s = split_sentences(_speech([(1.0, "One two."), (3.0, "Three four"), (6.0, "five six.")]))
    assert [x.text for x in s] == ["One two.", "Three four", "five six."]


# ------------------------------------------------------------------ sparse-punctuation transcripts (the Boxabl interviews)

def _bare(words):
    """What long-form Whisper output often looks like: lowercase, no punctuation."""
    return [{**w, "word": w["word"].strip(".,!?").lower()} for w in words]


def test_gap_at_reports_the_pause_around_a_cut():
    from quillcut.creative.facts import gap_at
    words = _speech([(1.0, "one two three"), (5.0, "four five")])
    pause, prev, nxt = gap_at(words, words[2]["end"])
    assert prev["word"] == "three" and nxt["word"] == "four" and pause == pytest.approx(5.0 - words[2]["end"], abs=0.01)
    assert gap_at(words, words[1]["start"] + 0.2)[0] == 0.0          # a cut through a word
    assert gap_at(words, 99.0)[0] == float("inf")


def test_sparse_punctuation_lowers_confidence_and_blocks_automatic_repair():
    words = _bare(_speech(LINES))
    words[-2]["start"] -= 0.0                                         # "than" -> "an" stays a 0.37 s... make it flow
    for w in words[-4:]:
        w["start"] -= 0.3; w["end"] -= 0.3                            # continuous speech across the cut
    facts, src, book = _facts(src=_src(words))
    res = checks.ending(facts)
    f = res.findings[0]
    assert f.severity == "major" and f.confidence < CFG["creative"]["act_min_confidence"] and f.score == 2.5
    assert "little punctuation" in f.evidence and "Verify by ear" in f.recommended_fix
    r = _results(); r["ending"] = res
    out = repair.plan_repairs(_report(r), facts, repair.State(reel=_reel()), _build_fn(src, book))
    assert not out.applied and "below the action threshold" in out.skipped[0]["reason"]
    assert _report(r).decision == "MANUAL_REVIEW"                    # cannot be verified -> a human listens


def test_a_real_pause_is_a_stop_only_when_punctuation_cannot_be_trusted():
    bare = _bare(_speech(LINES))                                      # "than" ... 0.37 s ... "an hour to build"
    f_bare = checks.ending(_facts(src=_src(bare))[0])
    assert not any(x.severity == "critical" for x in f_bare.findings)
    f_punct = checks.ending(_facts()[0])
    assert any(x.severity == "critical" for x in f_punct.findings)    # same pause, but punctuation says unfinished


def test_candidates_fall_back_to_pauses_when_there_is_no_punctuation():
    from quillcut.creative.facts import stop_times
    bare = _bare(_speech([(1.0, "one two three"), (5.0, "four five six"), (9.0, "seven eight")]))
    assert stop_times(bare) == [pytest.approx(w["end"]) for w in (bare[2], bare[5], bare[7])]
    punct = _speech([(1.0, "one two three"), (5.0, "four five six."), (9.0, "seven eight")])
    assert stop_times(punct) == [pytest.approx(punct[5]["end"]), pytest.approx(punct[7]["end"])]


def test_first_word_clipped_only_when_the_in_point_runs_through_a_word(tmp_path):
    video = _clip(tmp_path, "c.mp4")
    base = _speech([(1.0, "Welcome to the casita tour today."), (4.2, "The home folds flat for shipping."),
                    (8.0, "It unfolds in under one hour."), (12.0, "Inside you get a full kitchen."),
                    (16.0, "The ceilings are nine feet tall."), (20.0, "Everything ships ready to use."),
                    (24.0, "The price starts quite low today."), (28.0, "Windows fill the room with light."),
                    (32.0, "That is the whole casita tour.")])

    def clipped(prev_end):
        words = [{"word": "earlier.", "start": prev_end - 0.4, "end": prev_end, "probability": 0.9}] + base
        facts, *_ = _facts(reel=_reel(36.0, 1.0, UNPROTECTED), video=video, src=_src(words))
        return [f for f in checks.audio(facts).findings if f.issue == "The first word is clipped."]
    assert not clipped(0.80)          # the previous word ended 150 ms before the in-point (0.95s): fine
    assert clipped(0.97)              # it runs through the in-point: clipped


def test_unfittable_caption_groups_are_split_not_dropped():
    """Regression: 'pre-industrial pre-industrial just means' did not fit two lines, so the whole card
    (4 spoken words) vanished from a published reel."""
    from quillcut import captions
    style = load_style("gaming", "brand_clean")
    words = [{"word": w, "start": 3.0 + i * 0.25, "end": 3.0 + i * 0.25 + 0.22, "probability": 0.95}
             for i, w in enumerate(["pre", "-industrial", "pre", "-industrial", "just", "means"])]
    cards = captions.speech_cards(words, 10.0, style, CFG["safe_area"])
    shown = " ".join(c.text for c in cards).split()
    assert shown == ["pre-industrial", "pre-industrial", "just", "means"]
    assert all(c.bbox_norm[2] <= 1 - CFG["safe_area"]["right"] + 1e-3 for c in cards)
    ends = [c.end for c in cards]
    assert all(b <= a2 + 1e-6 for b, a2 in zip(ends, [c.start for c in cards[1:]]))   # cards never overlap


def test_review_pairs_the_production_render_with_the_production_plan_not_the_newest_candidate(tmp_path):
    pr = _project(tmp_path)
    specs = pr.sub("edit_specs")
    for v in (1, 2):
        (specs / f"t01.v{v}.editplan.json").write_text("{}", encoding="utf-8")
    assert loop.production_spec(pr, "t01").name == "t01.v2.editplan.json"          # nothing recorded: newest
    pr.mark_clip("t01", plan=str(specs / "t01.v1.editplan.json"))                   # what render_and_export records
    assert loop.production_spec(pr, "t01").name == "t01.v1.editplan.json"


def test_show_returns_the_most_recent_report_not_just_the_render_record(tmp_path, monkeypatch):
    import os
    import yaml
    pr = _project(tmp_path)
    pr.data = {"source_path": str(tmp_path / "s.mp4")}
    pr.save()
    cdir = pr.sub("creative")
    old = CreativeRecord(clip_id="t01", attempts=[Attempt(plan_version=1, overall=4.0, scores={}, decision="NEEDS_MODEL",
                                                         technical_qc="PASS", compliance="PASS",
                                                         report=_report(_results()).model_copy(update={"decision": "NEEDS_MODEL"}))])
    (cdir / "t01.creative.json").write_text(old.model_dump_json(), encoding="utf-8")
    new = _report(_results(hook=2.0))
    (cdir / "t01.review.json").write_text(new.model_dump_json(), encoding="utf-8")
    os.utime(cdir / "t01.creative.json", (1, 1))                       # the render record is older
    book = tmp_path / "b.yaml"
    book.write_text(yaml.safe_dump({"format": "gaming", "sources": {"a": str(tmp_path / "s.mp4")},
                                    "reels": [{"id": "t01", "parts": [["a", 0, 1]]}]}), encoding="utf-8")
    monkeypatch.setattr(loop, "find_project", lambda fmt, p: pr)
    assert loop.stored_report(book, "t01").categories["hook"].score == 2.0


def _seen(*descriptions, covers=()):
    return llm.VisionJudgement(
        frames=[llm.FrameJudgement(frame=i + 1, describe=d, caption_covers_face=(i in covers))
                for i, d in enumerate(descriptions)],
        times=[3.0 + 6 * i for i in range(len(descriptions))])


def test_the_vision_prompt_contains_no_narration():
    """Regression: with narration in the prompt the model described the narration and called a black video
    'a person speaking about cost'."""
    facts, *_ = _facts(reel=_reel(36.0))
    prompt = llm.vision_prompt(6)
    assert "narration" not in prompt.lower() and not any(w["word"].lower() in prompt.lower().split()
                                                         for w in facts.words if len(w["word"]) > 6)


def test_a_black_video_fails_on_the_pixels_whatever_the_vision_model_says(tmp_path):
    """Control from a real run: the model said 'subject clear' for black frames. Visibility is now measured."""
    black = _clip(tmp_path, "black.mp4", video="color=c=black:s=270x480:r=30:d=36")
    facts, *_ = _facts(reel=_reel(36.0), video=black)
    hallucinated = _seen(*["a person speaking about cost"] * 6)                # what the old judge claimed
    results, _ = score.evaluate(facts, llm.Models(vision=lambda f: (hallucinated, None),
                                                  text=lambda f: llm.TextJudgement(), reasons={}))
    visual = results["visual"]
    assert visual.score <= 2.0 and any(f.issue == "The picture is almost black." and f.severity == "major"
                                       for f in visual.findings)


def test_picture_vs_narration_problems_need_a_quote_and_two_frames_make_a_pattern():
    facts, *_ = _facts(reel=_reel(36.0))
    seen = _seen("a man riding a scooter", "a kitchen", "a car park", "a man talking")
    near = lambda i: llm.narration_between(facts, seen.times[i] - 2.0, seen.times[i] + 2.0)
    assert near(0) and near(1)
    quote = " ".join(near(1).split()[:3])
    reply = {"frames": [{"frame": 1, "unrelated": True, "quote": "invented words nobody said", "reason": "x"},
                        {"frame": 2, "unrelated": True, "quote": quote, "reason": "picture is a kitchen"},
                        {"frame": 3, "unrelated": False}]}
    rel = llm.judge_relation(FakeOllama(reply), "m", facts, seen)
    assert [x.frame for x in rel.frames] == [2] and rel.rejected == 1          # the unquoted complaint is dropped
    results = {c: CategoryResult(category=c, status="scored", score=5.0) for c in CATEGORIES}
    score._merge_vision(results, facts, seen, rel)
    one = results["visual"].findings[0]
    assert one.severity == "minor" and one.confidence == 0.45 and one.basis == "model"      # one frame: advisory
    two = llm.RelationJudgement(frames=[llm.Relation(frame=1, unrelated=True, quote="a b"),
                                        llm.Relation(frame=2, unrelated=True, quote="c d")])
    results = {c: CategoryResult(category=c, status="scored", score=5.0) for c in CATEGORIES}
    score._merge_vision(results, facts, seen, two)
    assert results["visual"].findings[0].severity == "major" and results["visual"].findings[0].confidence == 0.6


def test_a_caption_over_a_face_reported_by_the_vision_model_is_advisory_only():
    facts, *_ = _facts(reel=_reel(36.0))
    results = {c: CategoryResult(category=c, status="scored", score=5.0) for c in CATEGORIES}
    score._merge_vision(results, facts, _seen("a man", "a man", covers=(1,)))
    cap = results["captions"]
    assert cap.findings[0].edit_action == "move_captions" and cap.findings[0].confidence == 0.45
    assert cap.score == 5.0                                                    # below the counting threshold: shown, not scored


class SeqOllama:
    """Replies in order; the last one repeats. Records how many calls were made."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, model, user, images=None, **kw):
        r = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return r


def test_a_vision_reply_without_real_descriptions_is_rejected(tmp_path):
    facts, *_ = _facts(reel=_reel(36.0), video=_clip(tmp_path, "v.mp4"))
    lazy = {"frames": [{"frame": i, "describe": ""} for i in range(1, 7)]}
    with pytest.raises(llm.ModelUnavailable, match="not grounded"):
        llm.judge_frames(FakeOllama(lazy), "m", facts)
    ok = {"frames": [{"frame": i, "describe": "a room"} for i in range(1, 7)]}
    assert len(llm.judge_frames(FakeOllama(ok), "m", facts).times) == 6


def test_a_short_or_malformed_batch_answer_falls_back_to_one_frame_at_a_time(tmp_path):
    """Real run: asked about six frames, the model returned one bare frame object. That must not decide a clip."""
    facts, *_ = _facts(reel=_reel(36.0), video=_clip(tmp_path, "v.mp4"))
    one_bare = {"frame": 1, "describe": "a man on a scooter", "caption_covers_face": False}
    fake = SeqOllama(one_bare, *[{"frame": 1, "describe": f"scene {k}"} for k in range(1, 7)])
    v = llm.judge_frames(fake, "m", facts)
    assert fake.calls == 7 and [x.describe for x in v.frames] == [f"scene {k}" for k in range(1, 7)]
    assert [x.frame for x in v.frames] == [1, 2, 3, 4, 5, 6] and len(v.times) == 6
    blank = SeqOllama({"foo": "bar"})                                          # nothing usable even frame by frame
    with pytest.raises(llm.ModelUnavailable):
        llm.judge_frames(blank, "m", facts)
    empty = SeqOllama(one_bare, {"frame": 1, "describe": ""})
    with pytest.raises(llm.ModelUnavailable, match="not grounded"):
        llm.judge_frames(empty, "m", facts)


def test_a_model_only_off_topic_span_is_advisory_unless_it_ignores_the_headline_subject():
    facts, *_ = _facts(reel=_reel(36.0, 1.0, UNPROTECTED))
    on_subject = {"start": 4.0, "end": 7.0, "quote": "the home folds flat", "reason": "repeat"}       # mentions "home"
    off_subject = {"start": 17.0, "end": 20.0, "quote": "everything ships ready", "reason": "sidetrack"}
    j = llm.TextJudgement.model_validate(_reply(off_topic=[on_subject, off_subject]))
    results, _ = score.evaluate(facts, llm.Models(text=lambda f: j, reasons={"vision": "down"}))
    spans = {f.start_time: f for f in results["pacing"].findings if f.issue == "Unnecessary footage."}
    assert spans[4.0].confidence == 0.45 and "advisory" in spans[4.0].evidence
    assert spans[17.0].confidence == 0.6 and "advisory" not in spans[17.0].evidence
    cfg = CFG["creative"]
    assert spans[4.0].confidence < cfg["count_min_confidence"] and spans[17.0].confidence < cfg["act_min_confidence"]


def _finding_for(action, **params):
    return Finding(category="x", severity="major", score=3, issue="i", evidence="e", recommended_fix="f",
                   edit_action=action, confidence=0.9, start_time=2.0, end_time=3.0, params=params)


def _apply(finding, reel, facts, src, book):
    r = _results()
    r["audio"].findings.append(finding)
    return repair.plan_repairs(_report(r), facts, repair.State(reel=reel), _build_fn(src, book))


def test_trim_start_moves_the_in_point_to_the_next_sentence():
    reel = _reel(36.0, 1.0, UNPROTECTED)
    facts, src, book = _facts(reel=reel)
    out = _apply(_finding_for("trim_start", part=0, source="a", src_start=3.9), reel, facts, src, book)
    assert [a["action"] for a in out.applied] == ["trim_start"]
    part = out.state.reel["parts"][0]
    assert part[1] == 3.9 and part[3]["exact_start"] is True
    assert _build_fn(src, book)(out.state).segments[0].src_in >= 3.9 - 0.06          # the rebuilt plan really starts later


def test_move_captions_walks_down_the_configured_positions_then_stops():
    reel = _reel(36.0, 1.0, UNPROTECTED)
    facts, src, book = _facts(reel=reel)
    first = _apply(_finding_for("move_captions"), reel, facts, src, book)
    assert first.state.caption_y == CFG["creative"]["captions"]["y_options"][1]
    state = repair.State(reel=reel, caption_y=CFG["creative"]["captions"]["y_options"][-1])
    r = _results(); r["audio"].findings.append(_finding_for("move_captions"))
    last = repair.plan_repairs(_report(r), facts, state, _build_fn(src, book))
    assert not last.applied and "every caption position" in last.skipped[0]["reason"]


def test_retime_cta_lengthens_an_existing_overlay_and_refuses_when_there_is_none():
    reel = _reel(36.0, 1.0, UNPROTECTED)
    reel["cta"] = {"text": "Would you live in a casita?", "seconds": 1.0}
    facts, src, book = _facts(reel=reel, policy={"cta": "auto"})
    out = _apply(_finding_for("retime_cta"), reel, facts, src, book)
    assert out.state.reel["cta"]["seconds"] >= CFG["creative"]["cta"]["min_seconds"]
    plain = _reel(36.0, 1.0, UNPROTECTED)
    none = _apply(_finding_for("retime_cta"), plain, *_facts(reel=plain)[:1], *_facts(reel=plain)[1:])
    assert not none.applied and "not a repairable overlay" in none.skipped[0]["reason"]


def test_fit_blur_is_not_counted_as_an_enlarged_picture():
    from types import SimpleNamespace as NS
    ref = NS(content_box=(1280, 720, 0, 0))
    f = NS(plan=NS(sources={"a": ref}))
    crop = NS(video_source=None, source="a", layout="wide_fill")
    fitted = NS(video_source=None, source="a", layout="fit_blur")
    assert checks._upscale(f, crop) > 1.8 and checks._upscale(f, fitted) < 1.0     # whole frame fitted, backdrop only


def test_loop_ignores_the_closing_brand_beat_when_the_book_has_no_end_card():
    facts, *_ = _facts(reel=_reel(to=36.0), policy={"closing_beat": False})
    note = checks.loop(facts).note
    assert "closing brand beat" not in note and "a concrete figure is stated" in note


def test_book_fade_out_reaches_the_plan_and_defaults_to_none():
    plain, *_ = _facts()
    faded, *_ = _facts(book={"key": "t", "music": False, "fade_out": 0.35, "rights_record": {"status": "x"}})
    assert plain.plan.audio.fade_out == 0.0 and faded.plan.audio.fade_out == 0.35

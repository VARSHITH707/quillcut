"""Conversation-aware pipeline: analyse the whole video, pick complete moments, edit, render, validate."""
from __future__ import annotations

import difflib
import logging
from pathlib import Path

from . import conversation as cv
from . import jobs, media, reframe, reelplan, semantic, speakers
from .config import load_config
from .formats import load_style
from .gpu import GpuLock
from .jobs import SourceRejected, merge_style, prepare, render_and_export
from .storage import atomic_write_json, read_json

log = logging.getLogger("quillcut")
SPEAKER_VERSION = "3"
VETO_RISK = 0.35          # continuation risk at which a model "unfinished" verdict is believed


def _speakers_for(prep, video: Path, sentences: list[cv.Sentence]) -> list[str]:
    path = prep.project.sub("analysis") / "speakers.json"
    key = prep.project.data["source_sha1"] + SPEAKER_VERSION + str(len(sentences))
    cached = read_json(path, {})
    if cached.get("key") == key:
        return cached["labels"]
    audio = media.decode_audio(video)
    labels = speakers.assign(audio, [(s.start, s.end) for s in sentences],
                             protect={s.idx for s in sentences if s.question})
    atomic_write_json(path, {"key": key, "labels": labels})
    return labels


def understand(video: Path, fmt: str, rights: str | None, cfg: dict):
    """Transcribe, separate voices, and split the conversation into sentences and units."""
    prep = prepare(video, fmt, "flag", rights, cfg)
    if prep.mode != "commentary" or len(prep.words) < 20:
        raise SourceRejected("No clear speech was found in this video, so there is no conversation to clip. "
                             "Quillcut clips talking: interviews, podcasts, lessons, talks.")
    sentences = cv.split_sentences(prep.words)
    labels = _speakers_for(prep, video, sentences)
    return prep, sentences, labels


def duration_note(target: float, actual: float, cold_open: bool) -> str:
    """Plain-language comparison of the requested and the rendered length."""
    extra = " (includes the opening hook line)" if cold_open else ""
    if actual > target + 1.0:
        return f"Extended from {target:.0f}s to {actual:.0f}s to finish the thought{extra}."
    if actual < target - 5.0:
        return f"Ends at {actual:.0f}s because the thought is complete; nothing was added to reach {target:.0f}s."
    return f"Close to the {target:.0f}s target at {actual:.0f}s{extra}."


def moment_summary(c: cv.Candidate, sentences: list[cv.Sentence], units: list[cv.Unit]) -> dict:
    u = units[c.unit]
    hook = sentences[c.hook_idx] if c.hook_idx is not None else sentences[c.first]
    return {"start": round(c.start, 1), "end": round(c.end, 1), "duration": round(c.duration, 1),
            "kind": u.kind, "question_included": c.question_included, "partial": c.partial,
            "hook": hook.text, "preview": c.text[:160], "score": c.score, "scores": c.scores,
            "reasons": c.reasons}


def analyse_only(video: Path, fmt: str = "gaming", rights: str | None = None, on_progress=None) -> dict:
    """What is in this video: how many distinct complete moments, their lengths and hooks. No rendering."""
    cfg = load_config(fmt)
    with GpuLock():
        if on_progress:
            on_progress("analysing")
        prep, sentences, labels = understand(video, fmt, rights, cfg)
    out = cv.analyse(prep.words, prep.ref.duration, targets=[], speakers=labels)
    cands, sents, units = out["candidates"], out["sentences"], out["units"]
    for c in cands:
        c.score = cv.total_score(c, 45.0, cv.hard_max_for(45.0))
    ranked: list[cv.Candidate] = []
    for c in sorted(cands, key=lambda c: -c.score):
        if c.score >= 0.5 and not any(cv._overlaps(c, r, sents) for r in ranked):
            ranked.append(c)
    return {
        "duration": round(prep.ref.duration, 1),
        "sentences": len(sents),
        "speakers": sorted({s.speaker for s in sents if s.speaker}),
        "questions": sum(1 for s in sents if s.question),
        "units": len(units),
        "capacity": len(ranked),
        "moments": [moment_summary(c, sents, units) for c in ranked[:12]],
        "min_seconds": cv.MIN_ABSOLUTE_SECONDS,
    }


# --------------------------------------------------------------------------------------------- validation


def _norm(words: list[str]) -> list[str]:
    return [cv.clean_word(w) for w in words if cv.clean_word(w)]


def verify_render(video: Path, plan, cfg: dict, expected_words: list[str]) -> list[dict]:
    """Check the rendered file itself: length, transcript, clean start and end, hook presence."""
    checks: list[dict] = []

    def add(name: str, ok: bool, msg: str, warn: bool = False):
        checks.append({"name": name, "status": "PASS" if ok else ("WARN" if warn else "FAIL"), "message": msg})

    info = media.video_info(video)
    add("duration_matches_plan", abs(info["duration"] - plan.duration) <= 0.2,
        f"{info['duration']:.2f}s rendered, {plan.duration:.2f}s planned")
    hook = [c for c in plan.text_cards if c.kind == "hook"]
    add("hook_in_first_seconds", bool(hook) and hook[0].start < 1.0, f"hook {hook[0].text!r}" if hook else "no hook card")
    speech_end = max((w.end for c in plan.text_cards if c.kind == "speech" for w in c.words), default=0.0)
    add("captions_inside_clip", speech_end <= plan.duration + 0.05, f"last caption ends {speech_end:.2f}s")

    speech = [c for c in plan.text_cards if c.kind == "speech"]
    too_long = [c.text for c in speech if len(c.text.split()) > reelplan.CAPTION_MAX_WORDS]
    add("caption_word_limit", not too_long,
        f"{len(speech)} caption cards, none over {reelplan.CAPTION_MAX_WORDS} words" if not too_long
        else f"over {reelplan.CAPTION_MAX_WORDS} words: {too_long[:2]}")
    lo, hi = reelplan.CAPTION_BAND
    off = [c.text for c in speech if not lo - 0.02 <= (c.bbox_norm[1] + c.bbox_norm[3]) / 2 <= hi + 0.02]
    add("captions_in_centre_third", not off, "all caption cards centred between 35% and 65% of the height"
        if not off else f"outside the centre third: {off[:2]}")
    hook_long = [c.text for c in hook if len(c.text.rstrip("…").split()) > reelplan.HOOK_FRAME_WORDS]
    add("hook_word_limit", not hook_long, f"{len(hook)} hook frame(s), each at most {reelplan.HOOK_FRAME_WORDS} words"
        if not hook_long else f"too many words: {hook_long}")
    safe = cfg["safe_area"] if plan.output.width < plan.output.height else \
        {"top": 0.06, "bottom": 0.10, "left": 0.06, "right": 0.06}
    outside = [c.text for c in plan.text_cards
               if not (c.bbox_norm[0] >= safe["left"] - 1e-3 and c.bbox_norm[2] <= 1 - safe["right"] + 1e-3
                       and c.bbox_norm[1] >= safe["top"] - 1e-3 and c.bbox_norm[3] <= 1 - safe["bottom"] + 1e-3)]
    add("overlays_inside_safe_area", not outside, f"{len(plan.text_cards)} overlays inside the safe area"
        if not outside else f"outside the safe area: {outside[:3]}")
    try:
        from .models.asr import FasterWhisperASR
        a = cfg["asr"]
        heard = FasterWhisperASR("small", a["device"], a["compute_type"]).transcribe(media.decode_audio(video))
        got = _norm([w["word"] for w in heard["words"]])
        want = _norm(expected_words)
        if not got or not want:
            add("speech_continuity", False, "could not hear speech in the rendered reel")
            return checks
        ratio = difflib.SequenceMatcher(a=want, b=got, autojunk=False).ratio()
        add("speech_matches_plan", ratio >= 0.80, f"{ratio:.0%} of the planned words are heard in order")
        # the closing words must be present: the clip must not stop in the middle of the answer
        tail_want = want[-3:]
        add("ends_on_finished_sentence", all(w in got[-6:] for w in tail_want[-2:]),
            f"expected ending {' '.join(tail_want)!r}, heard ending {' '.join(got[-4:])!r}")
        add("starts_on_planned_words", all(w in got[:6] for w in want[:2]),
            f"expected start {' '.join(want[:3])!r}, heard {' '.join(got[:4])!r}")
    except Exception as exc:  # noqa: BLE001 - verification must not take the render down with it
        log.warning("speech verification unavailable: %s", exc)
        add("speech_verification", False, "speech verification could not run", warn=True)
    return checks


# --------------------------------------------------------------------------------------------- run


def run(video: Path, *, rights: str | None, targets: list[float], fmt: str = "gaming", caption_style: str | None = None,
        style_overrides: dict | None = None, options: reelplan.PlanOptions | None = None, strict: bool = False,
        include_question: str = "auto", hard_max: float | None = None, fast: bool = False,
        publish_dir: Path | None = None, use_llm: bool = True, verify: bool = True, on_progress=None,
        profile: str | None = None, exclude: list[tuple[float, float]] | None = None,
        prefer: tuple[float, float] | None = None) -> dict:
    options = options or reelplan.PlanOptions()
    cfg = load_config(fmt)

    def progress(stage: str, done: int = 0, total: int = 0) -> None:
        if on_progress:
            on_progress(stage, done, total)

    progress("waiting_for_gpu")
    with GpuLock():
        progress("transcribing")
        prep, sentences, labels = understand(video, fmt, rights, cfg)
        progress("understanding")
        out = cv.analyse(prep.words, prep.ref.duration, targets=targets, strict=strict,
                         include_question=include_question, hard_max=hard_max, speakers=labels,
                         profile=profile, exclude=exclude, prefer=prefer)
        sents, units = out["sentences"], out["units"]
        selections = out["selections"]

        llm = cfg.get("creative", {}).get("llm", {})
        reviews: dict[tuple[int, int], dict] = {}
        if use_llm and any(s.candidate for s in selections):
            progress("reviewing")
            for _ in range(2):                               # a vetoed moment is replaced by the next best one
                vetoed = False
                for sel in selections:
                    c = sel.candidate
                    if c is None or (c.first, c.last) in reviews:
                        continue
                    before = " ".join(s.text for s in sents[max(0, c.first - 3):c.first])
                    after = " ".join(s.text for s in sents[c.last + 1:c.last + 3])
                    r = semantic.review(c.text, before, after, url=llm.get("url", "http://localhost:11434"),
                                        model=llm.get("text_model", "qwen2.5:7b-instruct-q4_K_M"))
                    if r is None:
                        break
                    reviews[(c.first, c.last)] = r
                    c.topic = r["topic"]
                    # a 7B model is a noisy judge: it can only veto when the text that follows really does
                    # continue the same point, which is checked deterministically
                    if r["complete"] is False and cv.continuation_risk(sents, c) >= VETO_RISK:
                        out["candidates"] = [x for x in out["candidates"] if (x.first, x.last) != (c.first, c.last)]
                        vetoed = True
                    elif r["complete"] is False:
                        c.reasons.append("model unsure the point is finished; text after the clip starts a new point")
                if not vetoed:
                    break
                selections = cv.select(out["candidates"], sents, targets, strict=strict, hard_max=hard_max,
                                       profile=profile, exclude=exclude, prefer=prefer)

        style = merge_style(load_style(fmt, caption_style or cfg["editing"]["caption_style"]), style_overrides)
        results: list[dict] = []
        total = sum(1 for s in selections if s.candidate)
        done = 0
        for sel in selections:
            if sel.candidate is None:
                results.append({"reel": sel.reel, "target": sel.target, "rejected": sel.note})
                continue
            c = sel.candidate
            progress("editing", done, total)
            clip_id = f"r_{sel.reel:02d}"
            built = reelplan.build(
                c, sents, prep.ref, cfg, style, clip_id=clip_id, project_id=prep.project.id,
                rights=prep.project.data["rights"], options=options,
                crop_centers=lambda shots, p=prep: reframe.crop_centers(Path(p.ref.path), shots, p.ref.content_box))
            progress("rendering", done, total)
            res = render_and_export(built.plan, prep.project, cfg, fast=fast, publish_dir=publish_dir)
            done += 1
            actual = round(built.plan.duration, 2)
            entry = {
                "reel": sel.reel, "clip_id": clip_id, "target": sel.target,
                "actual": actual, "extended": actual > sel.target + 1.0,
                "note": duration_note(sel.target, actual, built.hook["kind"] == "cold_open"),
                "source_start": round(c.start, 2), "source_end": round(c.end, 2),
                "question_included": c.question_included, "partial": c.partial,
                "hook": {"kind": built.hook["kind"], "text": built.hook["text"],
                         "archetype": built.hook.get("archetype"), "frames": built.hook.get("frames", [])},
                "topic": c.topic or "", "scores": c.scores, "score": c.score, "reasons": c.reasons,
                "intensity": options.intensity, "aspect": options.aspect, "notes": built.notes,
                "qc": res["qc"], "failures": res["failures"], "warnings": res["warnings"],
                "final": res.get("final"), "published": res.get("published"), "duration": res["duration"],
            }
            if res["qc"] == "PASS" and verify:
                progress("validating", done, total)
                # what should be heard, in playback order (a cold open plays its hook line first)
                spoken = [w for s in sents[c.first:c.last + 1] for w in s.words]
                expected = [w["word"] for w in reelplan.map_words(built.plan.segments, spoken)]
                entry["validation"] = verify_render(Path(res["final"]), built.plan, cfg, expected)
                bad = [v for v in entry["validation"] if v["status"] == "FAIL"]
                if bad:
                    entry["needs_review"] = True
                    entry["failures"] = entry["failures"] + [f"{v['name']}: {v['message']}" for v in bad]
            results.append(entry)
        progress("finished", done, total)
    return {"results": results, "capacity": cv.distinct_capacity(out["candidates"], sents),
            "sentences": len(sents), "speakers": sorted({s.speaker for s in sents if s.speaker})}

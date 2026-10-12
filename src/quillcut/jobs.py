"""Orchestration: ingest → analysis (cached) → plan → render → QC → export."""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from . import analysis, delegate, media, qc
from .config import load_config
from .formats import load_pipeline, load_style, resolve_format
from .gpu import GpuLock
from .paths import CACHE, OUTPUTS
from .render import render_plan
from .scripts import PlanRejected, SourceCtx, build_plan, load_templates
from .storage import Project, atomic_write_json, disk_free_gb, read_json, resolve_rights
from .types import EditPlan, SourceRef

log = logging.getLogger("quillcut")
ANALYSIS_VERSION = "1"


class SourceRejected(Exception):
    pass


@dataclass
class Prepared:
    project: Project
    ref: SourceRef
    fused: dict
    mode: str
    words: list[dict]


def _check_disk(cfg: dict, where: Path) -> None:
    free = disk_free_gb(where)
    if free < cfg["storage"]["min_free_gb"]:
        raise SourceRejected(f"only {free:.1f} GB free on {where.anchor}; need {cfg['storage']['min_free_gb']} GB")


def prepare(video: Path, fmt: str, resolved_by: str, rights_flag: str | None, cfg: dict,
            transcribe: bool = True) -> Prepared:
    video = video.resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    _check_disk(cfg, video)
    project = Project.for_video(video, fmt, resolved_by, resolve_rights(video, rights_flag))
    info = media.video_info(video)
    if info["duration"] < 20:
        raise SourceRejected(f"{video.name} is {info['duration']:.0f}s; need at least 20 s of source")
    box = tuple(project.data.get("content_box") or
                media.detect_content_box(video, info["duration"], info["width"], info["height"]))
    short = min(box[0], box[1])
    q = cfg["source_quality"]
    if short < q["min_content_short_side"]:
        raise SourceRejected(f"real picture is {box[0]}x{box[1]}; under {q['min_content_short_side']} px is too low-res")
    project.data.update({"probe": info, "content_box": list(box),
                         "quality_warning": short < q["warn_content_short_side"]})
    project.save()
    ref = SourceRef(path=str(video), sha1=project.data["source_sha1"], width=info["width"], height=info["height"],
                    fps=info["fps"], duration=info["duration"], has_audio=info["has_audio"], content_box=box)

    adir = project.sub("analysis")
    key = project.data["source_sha1"] + ANALYSIS_VERSION
    if not project.stage_complete("analysis", key):
        project.mark_stage("analysis", "running", key)
        audio = media.decode_audio(video)
        fps = cfg["analysis"]["video_fps"]
        fused = analysis.fuse(analysis.audio_features(audio, int(info["duration"] * analysis.HZ)),
                              analysis.video_features(video, box, fps), info["duration"], fps)
        from .models.asr import speech_coverage
        coverage = speech_coverage(audio)
        np.savez_compressed(adir / "fused.npz", t=fused["t"], intensity=fused["intensity"],
                            rms_db=fused["rms_db"], motion=fused["motion"])
        atomic_write_json(adir / "meta.json", {"cuts": fused["cuts"], "speech_coverage": coverage})
        project.mark_stage("analysis", "complete", key, speech_coverage=coverage)
    data = np.load(adir / "fused.npz")
    meta = read_json(adir / "meta.json")
    fused = {k: data[k] for k in data.files} | {"cuts": meta["cuts"]}

    mcfg = cfg.get("mode", {})
    mode = "commentary" if meta["speech_coverage"] >= mcfg.get("commentary_min_speech_coverage", 0.25) else "cinematic"
    words: list[dict] = []
    tpath = adir / "transcript.json"
    if mode == "commentary" and (transcribe or project.stage_complete("transcript", key)):
        if not project.stage_complete("transcript", key):
            from .models.asr import FasterWhisperASR
            a = cfg["asr"]
            lang = None if cfg.get("language", "auto") == "auto" else cfg["language"]
            transcript = FasterWhisperASR(a["model"], a["device"], a["compute_type"]).transcribe(
                media.decode_audio(video), language=lang)
            atomic_write_json(tpath, transcript)
            project.mark_stage("transcript", "complete", key)
        words = read_json(tpath, {}).get("words", [])
    project.data["mode"] = mode
    project.save()
    return Prepared(project, ref, fused, mode, words)


def source_ctx(p: Prepared, key: str, cfg: dict, rng: tuple[float, float] | None = None,
               peak: float | None = None, nth: int = 0) -> SourceCtx:
    lo, hi = rng or (0.0, p.ref.duration)
    lo, hi = max(0.0, lo), min(p.ref.duration, hi)
    if peak is None:
        peaks = analysis.find_peaks(p.fused["t"], p.fused["intensity"], top_n=nth + 1,
                                    min_sep=cfg["analysis"]["peak_min_separation"], lo=lo, hi=hi)
        if len(peaks) <= nth:
            raise PlanRejected(f"no peak #{nth + 1} found in {lo:.0f}-{hi:.0f}s")
        peak, value = peaks[nth]
    else:
        idx = min(int(peak * analysis.HZ), len(p.fused["intensity"]) - 1)
        value = float(p.fused["intensity"][idx])
    return SourceCtx(key=key, ref=p.ref, t_peak=peak, lo=lo, hi=hi, peak_value=round(value, 4), words=p.words)


def render_and_export(plan: EditPlan, project: Project, cfg: dict, *, fast: bool = False,
                      publish_dir: Path | None = None) -> dict:
    specs = project.sub("edit_specs")
    plan_path = specs / f"{plan.clip_id}.v{plan.plan_version}.editplan.json"
    atomic_write_json(plan_path, json.loads(plan.model_dump_json()))
    r = cfg["render"]
    out = project.sub("renders") / f"{plan.clip_id}_v{plan.plan_version}.mp4"
    work = project.sub("work") / plan.clip_id
    render_plan(plan, out, work, preset="veryfast" if fast else r["preset"], crf=r["crf"], threads=r["threads"])
    shutil.rmtree(work, ignore_errors=True)
    checks = qc.run_qc(out, plan, cfg)
    atomic_write_json(project.sub("qc") / f"{plan.clip_id}.json", qc.as_dicts(checks))
    result = {"clip_id": plan.clip_id, "plan": str(plan_path), "render": str(out),
              "duration": plan.duration, "qc": "PASS" if qc.passed(checks) else "FAIL",
              "failures": [f"{c.name}: {c.message}" for c in checks if c.status == "FAIL"],
              "warnings": [f"{c.name}: {c.message}" for c in checks if c.status == "WARN"]}
    if qc.passed(checks):
        final = project.sub("final") / f"{plan.clip_id}.mp4"
        shutil.move(str(out), final)
        published = (publish_dir / f"{plan.clip_id}.mp4" if publish_dir else
                     OUTPUTS / f"{date.today():%Y-%m-%d}" / plan.format_key / f"{project.id[:40]}_{plan.clip_id}.mp4")
        published.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, published)
        meta = {"clip_id": plan.clip_id, "script": plan.script, "duration": plan.duration,
                "hook_text": next((c.text for c in plan.text_cards if c.kind == "hook"), None),
                "source_ranges": [(s.source, s.src_in, s.src_out) for s in plan.segments],
                "rights": plan.rights, "placeholders_used": plan.placeholders_used,
                "topic": plan.moment.get("topic"), "music": plan.audio.music.file if plan.audio.music else None,
                "content_id_risk": "game_soundtrack" if plan.moment.get("type") == "narrative_beat" else None}
        atomic_write_json(project.sub("metadata") / f"{plan.clip_id}.json", meta)
        result.update(final=str(final), published=str(published))
    project.mark_clip(plan.clip_id, **{k: v for k, v in result.items() if k not in ("clip_id", "warnings")})
    return result


def run_auto(video: Path, fmt_flag: str | None, rights: str | None, max_clips: int | None = None,
             fast: bool = False) -> list[dict]:
    fmt, how = resolve_format(video, fmt_flag)
    if fmt is None:
        raise SourceRejected("format unresolved: pass --format, use formats/<key>/input/, or a sidecar")
    cfg = load_config(fmt)
    pipeline = load_pipeline(fmt)
    with GpuLock():
        if pipeline.delegated:
            return run_delegated(video, fmt, how, rights, cfg)
        p = prepare(video, fmt, how, rights, cfg)
        tpl = load_templates(fmt)[pipeline.auto_template]
        style = load_style(fmt, cfg["editing"]["caption_style"])
        limit = max_clips or cfg["moments"]["max_clips_per_project"]
        results = []
        for n in range(limit):
            clip_id = f"c_{n + 1:02d}"
            try:
                ctx = source_ctx(p, "a", cfg, nth=n)
                plan = build_plan(tpl, {"a": ctx}, cfg, style, clip_id=clip_id, project_id=p.project.id,
                                  mode=p.mode, rights=p.project.data["rights"])
            except PlanRejected as exc:
                log.info("stop at %s: %s", clip_id, exc)
                results.append({"clip_id": clip_id, "rejected": str(exc)})
                if "no peak" in str(exc):
                    break
                continue
            results.append(render_and_export(plan, p.project, cfg, fast=fast))
        return results


def run_script(script: str, fmt: str, sources: dict[str, dict], rights: str | None, text: str | None = None,
               layout: str | None = None, fast: bool = False) -> dict:
    """sources: {'a': {'video': Path, 'range': (lo, hi) | None, 'peak': float | None}, 'b': {...}}"""
    cfg = load_config(fmt)
    tpl = load_templates(fmt)[script]
    style = load_style(fmt, cfg["editing"]["caption_style"])
    with GpuLock():
        wants_speech = tpl.get("speech_captions") == "auto"
        prepared = {k: prepare(Path(v["video"]), fmt, "flag", rights, cfg, transcribe=wants_speech and k == "a")
                    for k, v in sources.items()}
        ctxs = {k: source_ctx(prepared[k], k, cfg, v.get("range"), v.get("peak")) for k, v in sources.items()}
        a = prepared["a"]
        rights_info = a.project.data["rights"]
        if any(p.project.data["rights"]["status"] == "unknown" for p in prepared.values()):
            rights_info = {"status": "unknown", "resolved_by": "a source has no rights record"}
        existing = [c for c in a.project.data.get("clips", {}) if c.startswith(f"s_{script}")]
        clip_id = f"s_{script}_{len(existing) + 1:02d}"
        plan = build_plan(tpl, ctxs, cfg, style, clip_id=clip_id, project_id=a.project.id,
                          mode=a.mode if wants_speech else "cinematic", rights=rights_info, text=text,
                          layout=layout)
        return render_and_export(plan, a.project, cfg, fast=fast)


def run_delegated(video: Path, fmt: str, how: str, rights: str | None, cfg: dict) -> list[dict]:
    project = Project.for_video(video, fmt, how, resolve_rights(video, rights))
    job = delegate.submit(video.resolve(), cfg)
    status = delegate.wait(job)
    if status.get("status") != "completed":
        raise SourceRejected(f"OpenShorts job {job} failed: {status.get('logs', [])[-2:]}")
    results = []
    for n, clip in enumerate(delegate.clips(job), start=1):
        clip_id = f"d_{n:02d}"
        final = project.sub("final") / f"{clip_id}.mp4"
        shutil.copy2(clip, final)
        entry = {"clip_id": clip_id, "final": str(final), "engine": "openshorts", "job_id": job}
        if project.data["rights"]["status"] != "unknown":
            published = OUTPUTS / f"{date.today():%Y-%m-%d}" / fmt / f"{project.id[:40]}_{clip_id}.mp4"
            published.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final, published)
            entry["published"] = str(published)
        project.mark_clip(clip_id, **entry)
        results.append(entry)
    return results


def ensure_shots(p: Prepared) -> list[tuple[float, float]]:
    """Frame-accurate source shot list (PySceneDetect on a small CPU proxy), cached per source."""
    path = p.project.sub("analysis") / "shots.json"
    cached = read_json(path)
    if cached:
        return [tuple(x) for x in cached]
    proxy = p.project.sub("analysis") / "shots_proxy.mp4"
    media.run(["ffmpeg", "-y", "-v", "error", "-i", p.ref.path, "-vf", "scale=640:-2", "-an",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", str(proxy)])
    from scenedetect import ContentDetector, SceneManager, open_video
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=22, min_scene_len=8))
    manager.detect_scenes(open_video(str(proxy)))
    shots = [(round(a.seconds, 3), round(b.seconds, 3)) for a, b in manager.get_scene_list()]
    proxy.unlink(missing_ok=True)
    atomic_write_json(path, shots)
    return shots


def first_visible_frame(path: str, t: float, span: float = 2.0) -> float:
    """If the source is black at t, return the first time it is not (within span)."""
    proc = media.run(["ffmpeg", "-hide_banner", "-ss", f"{t:.3f}", "-t", f"{span:.2f}", "-i", path, "-an",
                      "-vf", "blackdetect=d=0.03:pix_th=0.10", "-f", "null", "-"], check=False)
    import re as _re
    for start, end in _re.findall(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)", proc.stderr):
        if float(start) < 0.05:
            return round(t + float(end) + 0.05, 3)
    return t


def run_book(book_path: Path, only: set[str] | None = None, fast: bool = False, creative: bool = False,
             repair: bool = True) -> list[dict]:
    """creative=True swaps the final render/export for the creative pass (evaluate, repair, re-render, decide);
    it writes to new versioned paths only and never overwrites existing production files."""
    from .config import load_yaml
    from .reels import ShotSource, apply_fixes, apply_word_fixes, build_reel, load_music_library
    book = load_yaml(book_path)
    fmt = book.get("format", "gaming")
    cfg = load_config(fmt)
    style = load_style(fmt, book.get("caption_style", cfg["editing"]["caption_style"]))
    library = load_music_library()
    book["rights_record"] = {"status": book.get("rights", "unknown"), "resolved_by": f"reel book {book_path.name}",
                             "note": book.get("rights_note")}
    publish_dir = OUTPUTS / book["output"]
    results = []
    with GpuLock():
        sources, projects = {}, {}
        for key, path in book["sources"].items():
            if book.get("always_transcribe"):
                cfg["mode"]["commentary_min_speech_coverage"] = 0.0
            p = prepare(Path(path), fmt, "flag", None, cfg, transcribe=True)
            words = apply_fixes(p.words, [f for f in book.get("transcript_fixes", []) if f["source"] == key])
            words = apply_word_fixes(words, book.get("word_fixes", {}))
            ctx = SourceCtx(key=key, ref=p.ref, t_peak=0.0, lo=0.0, hi=p.ref.duration, words=words)
            extra = tuple(book.get("extra_cuts", {}).get(key, []))
            sources[key], projects[key] = ShotSource(ctx, ensure_shots(p), extra), p.project
        visible = lambda k, t: first_visible_frame(sources[k].ctx.ref.path, t)       # noqa: E731
        models = None
        if creative:
            from .creative.llm import Models
            models = Models.from_config(cfg, cache=CACHE / "creative" / "llm_cache.json")
        for reel in book["reels"]:
            if only and reel["id"] not in only:
                continue
            project = projects[reel["parts"][0][0]]
            if creative:
                from .creative.loop import run_reel
                results.append(run_reel(reel, book, sources, cfg, style, library, project, publish_dir=publish_dir,
                                        first_visible=visible, models=models, fast=fast, repair=repair))
                continue
            try:
                plan = build_reel(reel, book, sources, cfg, style, library, project.id, first_visible=visible)
            except PlanRejected as exc:
                results.append({"clip_id": reel["id"], "rejected": str(exc)})
                continue
            results.append(render_and_export(plan, project, cfg, fast=fast, publish_dir=publish_dir))
    return results

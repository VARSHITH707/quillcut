"""Web jobs for conversation-aware clipping: analyse a video, then make reels with per-reel durations."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import conversation_job, looks, reelplan
from .. import rules as rules_mod
from ..config import load_config, load_yaml
from ..paths import CLIPPING
from . import copywriter

CONTENT_STYLES = ("auto", "educational", "storytelling", "motivational", "podcast", "interview", "debate",
                  "business")
DURATION_PRESETS = [30, 35, 45, 60, 75, 90]
MIN_TARGET, MAX_TARGET, MAX_REELS = 15, 180, 20
MAX_MIX = 12
HEX = re.compile(r"^[0-9A-Fa-f]{6}$")
UPLOAD_ID = re.compile(r"^[0-9a-f]{12}$")


class OptionError(ValueError):
    pass


def music_tracks() -> list[dict]:
    """Licensed or generated beds found next to the data folder; empty when none are installed."""
    root = CLIPPING / "Sound Effects and Music"
    manifest = load_yaml(root / "manifest.yaml") if (root / "manifest.yaml").exists() else {}
    out = []
    for item in manifest.get("music", []) or []:
        path = root / item["file"]
        if path.exists():
            out.append({"file": str(path), "mood": item.get("mood", ""), "energy": item.get("energy", 0.5),
                        "licence": item.get("licence", "CC0 (original)")})
    return sorted(out, key=lambda t: t["energy"])


def _range(value, name: str) -> tuple[float, float]:
    try:
        a, b = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError):
        raise OptionError(f"{name} must be [start, end] in seconds.") from None
    if not 0 <= a < b:
        raise OptionError(f"{name} must be [start, end] in seconds.")
    return a, b


def validate(body: dict) -> dict:
    """Options that are specific to conversation reels (the caption style fields are validated by the server)."""
    reels = body.get("reels")
    if not isinstance(reels, list) or not 1 <= len(reels) <= MAX_REELS:
        raise OptionError(f"Choose between 1 and {MAX_REELS} reels.")
    targets: list[float] = []
    for i, r in enumerate(reels, start=1):
        try:
            t = float(r.get("target") if isinstance(r, dict) else r)
        except (TypeError, ValueError):
            raise OptionError(f"Reel {i}: the duration must be a number of seconds.") from None
        if not MIN_TARGET <= t <= MAX_TARGET:
            raise OptionError(f"Reel {i}: choose between {MIN_TARGET} and {MAX_TARGET} seconds.")
        targets.append(round(t, 1))
    intensity = body.get("intensity", "dynamic")
    if intensity not in reelplan.INTENSITY:
        raise OptionError("Editing intensity must be clean, dynamic or high energy.")
    aspect = body.get("aspect", "9:16")
    if aspect not in reelplan.ASPECTS:
        raise OptionError("Aspect ratio must be 9:16, 1:1 or 16:9.")
    include_question = body.get("include_question", "auto")
    if include_question not in ("auto", "yes", "no"):
        raise OptionError("Question setting must be auto, yes or no.")
    style = body.get("content_style", "auto")
    if style not in CONTENT_STYLES:
        raise OptionError("Unknown content style.")
    accent = str(body.get("accent", "F85830")).lstrip("#")
    if not HEX.match(accent):
        raise OptionError("The brand color must be a hex color like F85830.")
    cta = body.get("cta")
    if cta is not None:
        cta = str(cta).strip()[:60] or None
    caption_y = body.get("caption_y")
    if caption_y is not None:
        try:
            caption_y = float(caption_y)
        except (TypeError, ValueError):
            raise OptionError("Caption position must be a number.") from None
        if not 0.35 <= caption_y <= 0.65:
            raise OptionError("Caption position must stay in the centre third of the frame (0.35 to 0.65).")
    music = body.get("music", "none")
    if music not in ("none", "subtle"):
        raise OptionError("Music must be none or subtle.")
    exclude = [list(_range(r, "Excluded range")) for r in (body.get("exclude") or [])][:40]
    prefer = list(_range(body["prefer"], "Preferred range")) if body.get("prefer") else None
    text = body.get("rules")
    if text is not None and not isinstance(text, str):
        raise OptionError("Rules must be text.")
    text = (text or "").strip()
    if len(text) > rules_mod.MAX_RULE_CHARS:
        raise OptionError(f"Rules can be at most {rules_mod.MAX_RULE_CHARS} characters.")
    grade = body.get("grade") or "none"
    if grade not in looks.GRADES:
        raise OptionError("The look must be one of: " + ", ".join(sorted(looks.GRADES)) + ".")
    mix_amount = body.get("mix_amount", "light")
    if mix_amount not in reelplan.MIX_SHARE:
        raise OptionError("Mixed footage amount must be light or medium.")
    music_upload = body.get("music_upload")
    if music_upload is not None and not (isinstance(music_upload, str) and UPLOAD_ID.match(music_upload)):
        raise OptionError("Upload the music track again.")
    mix_uploads = body.get("mix_uploads") or []
    if not isinstance(mix_uploads, list) or len(mix_uploads) > MAX_MIX or \
            not all(isinstance(m, str) and UPLOAD_ID.match(m) for m in mix_uploads):
        raise OptionError(f"Add up to {MAX_MIX} mixed-footage files, then try again.")
    out = {"targets": targets, "intensity": intensity, "aspect": aspect, "include_question": include_question,
           "strict": bool(body.get("strict", False)), "content_style": style, "accent": accent.upper(),
           "cta": cta, "caption_y": caption_y, "music": music, "reframe": bool(body.get("reframe", True)),
           "exclude": exclude, "prefer": prefer, "logo_upload": body.get("logo_upload"), "hook_style": None,
           "rules": text, "grade": grade, "preset": body.get("preset"), "beat_cuts": bool(body.get("beat_cuts", False)),
           "music_upload": music_upload, "mix_uploads": list(dict.fromkeys(mix_uploads)), "mix_amount": mix_amount}
    parsed = rules_mod.parse(text)
    out["music_rule"] = parsed.music
    out["rule_notes"] = rules_mod.apply_to_options(parsed, out)       # rules win over the page, and say so
    out["rules_report"] = parsed.report()
    return out


def run_analysis(job: dict, update) -> None:
    """Read the whole video: transcript, speakers, conversation units, how many distinct reels it supports."""
    def progress(stage, done=0, total=0):
        update(stage=stage, done=done, total=total)
    result = conversation_job.analyse_only(Path(job["source"]), on_progress=progress)
    update(status="done", stage="finished", analysis=result)


def _music(opts: dict, music_path: Path | None, extra: list[str]) -> dict | None:
    """The bed for these reels: the user's own track first, then an installed one. Rules saying "no music"
    win over an upload."""
    if music_path is not None and opts.get("music_rule") == "none":
        extra.append("The rules say no music, so the uploaded track was left out.")
        music_path = None
    if music_path is not None:
        try:
            return conversation_job.music_with_beats(music_path, "supplied by the user")
        except Exception as exc:  # noqa: BLE001 - a bad track must not stop the reels
            extra.append(f"The music track could not be read ({exc}); the reels have no music.")
            return None
    if opts["music"] != "subtle":
        if opts.get("beat_cuts"):
            extra.append("Cut on the beat needs a music track; upload one to use it.")
        return None
    tracks = music_tracks()
    if not tracks:
        return {"file": ""}                     # reelplan says no track is installed
    if opts.get("beat_cuts"):
        return conversation_job.music_with_beats(Path(tracks[0]["file"]), tracks[0]["licence"])
    return {"file": tracks[0]["file"], "licence": tracks[0]["licence"], "level_db": -16.0}


def run_reels(job: dict, update, publish_dir: Path, logo_path: Path | None, music_path: Path | None = None,
              mix_paths: list[Path] | None = None) -> None:
    opts = job["options"]
    cfg = load_config("gaming")
    from_rules = opts.get("style_from_rules")        # the page's colours belong to the page's preset
    overrides = {} if from_rules else {"colors": opts["colors"], "speech": {"size": opts["caption_size"]},
                                       "hook": {"size": opts["hook_size"]}}
    extra: list[str] = []
    update(stage="preparing")
    music = _music(opts, music_path, extra)
    mix = conversation_job.mix_sources(mix_paths or [])
    if mix_paths and len(mix) < len(mix_paths):
        extra.append(f"{len(mix_paths) - len(mix)} mixed-footage file(s) could not be read and were left out.")
    plan_opts = reelplan.PlanOptions(
        intensity=opts["intensity"], aspect=opts["aspect"], reframe=opts["reframe"], cta=opts["cta"],
        accent=None if from_rules else opts["accent"], logo=str(logo_path) if logo_path else None,
        caption_y=opts.get("caption_y"), hook_style=opts.get("hook_style"), music=music,
        grade=None if opts.get("grade", "none") == "none" else opts["grade"],
        beat_cuts=bool(opts.get("beat_cuts")), mix=mix or None, mix_amount=opts.get("mix_amount", "light"))

    def progress(stage, done=0, total=0):
        update(stage=stage, done=done, total=total)

    profile = None if opts["content_style"] == "auto" else opts["content_style"]
    result = conversation_job.run(
        Path(job["source"]), rights=opts["rights"], targets=opts["targets"], caption_style=opts["preset"],
        style_overrides=overrides, options=plan_opts, strict=opts["strict"],
        include_question=opts["include_question"], fast=opts["quality"] == "draft", publish_dir=publish_dir,
        on_progress=progress, profile=profile, exclude=[tuple(r) for r in opts["exclude"]] or None,
        prefer=tuple(opts["prefer"]) if opts["prefer"] else None, rules=rules_mod.parse(opts.get("rules")))
    update(stage="writing_captions")
    llm = cfg.get("creative", {}).get("llm", {})
    clips, notes = [], []
    for r in result["results"]:
        if r.get("rejected"):
            clips.append({"reel": r["reel"], "target": r["target"], "rejected": r["rejected"]})
            continue
        entry = dict(r)
        entry["duration"] = r["actual"]
        if r["qc"] == "PASS" and r.get("published") and Path(r["published"]).exists():
            entry["file"] = Path(r["published"]).name
            text = _spoken_text(Path(r["final"]).parent.parent, r["source_start"], r["source_end"])
            entry["transcript"] = text[:600]
            entry.update(copywriter.social_post(
                r["hook"]["text"], r.get("topic") or None, text,
                content_style=None if opts["content_style"] == "auto" else opts["content_style"],
                url=llm.get("url", "http://localhost:11434"),
                model=llm.get("text_model", "qwen2.5:7b-instruct-q4_K_M")))
        clips.append(entry)
        notes += r.get("notes", [])
    good = [c for c in clips if c.get("file")]
    missing = [c for c in clips if c.get("rejected")]
    note = None
    if not good:
        note = (missing[0]["rejected"] if missing else "No reel passed the quality checks.")
    elif missing:
        note = (f"Made {len(good)} of {len(clips)} reels. The video has {result['capacity']} distinct complete "
                f"moment{'s' if result['capacity'] != 1 else ''}; the rest would have needed filler.")
    report = opts.get("rules_report") or {}
    messages = sorted(set(notes)) + extra + list(opts.get("rule_notes") or [])
    messages += [f"Not enforced automatically: {line}" for line in report.get("advisory", [])]
    update(status="done", stage="finished", clips=clips, note=note, capacity=result["capacity"],
           speakers=result["speakers"], messages=messages, rules_report=report,
           rule_notes=opts.get("rule_notes") or [])


def _spoken_text(project_dir: Path, start: float, end: float) -> str:
    try:
        words = json.loads((project_dir / "analysis" / "transcript.json").read_text(encoding="utf-8")).get("words", [])
    except (OSError, ValueError):
        return ""
    return " ".join(w["word"] for w in words if start - 0.05 <= w["start"] < end)

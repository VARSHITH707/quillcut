"""Quality control. QC never edits video; a failing clip simply never reaches final/."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import media
from .types import EditPlan


@dataclass
class Check:
    name: str
    status: str          # PASS | WARN | FAIL
    message: str


def _edge_frames(path: Path, duration: float, n: int = 6) -> np.ndarray:
    frames = []
    for k in range(n):
        t = duration * (k + 0.5) / n
        proc = media.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1",
                          "-vf", "scale=108:192,format=gray", "-f", "rawvideo", "-"], binary=True, check=False)
        if len(proc.stdout) == 108 * 192:
            frames.append(np.frombuffer(proc.stdout, dtype=np.uint8).reshape(192, 108))
    return np.array(frames)


def run_qc(video: Path, plan: EditPlan, cfg: dict) -> list[Check]:
    checks: list[Check] = []

    def add(name, ok, msg, warn=False):
        checks.append(Check(name, "PASS" if ok else ("WARN" if warn else "FAIL"), msg))

    decode = media.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=False)
    add("container_integrity", decode.returncode == 0 and not decode.stderr.strip(),
        decode.stderr.strip()[:300] or "clean full decode")

    info = media.video_info(video)
    want_size = (plan.output.width, plan.output.height)
    add("resolution", (info["width"], info["height"]) == want_size,
        f"{info['width']}x{info['height']} (plan {want_size[0]}x{want_size[1]})")
    want, got = Fraction(plan.output.fps), Fraction(info["fps"])
    add("frame_rate", want == got and not info["vfr"], f"{got} (plan {want})")
    tol = cfg["qc"]["duration_tolerance"]
    add("duration", abs(info["duration"] - plan.duration) <= tol, f"{info['duration']:.2f}s vs plan {plan.duration:.2f}s")
    add("duration_floor", info["duration"] >= plan.hard_floor_seconds - 0.05,
        f"{info['duration']:.2f}s, floor {plan.hard_floor_seconds}s")
    add("audio_presence", info["has_audio"], "audio track present" if info["has_audio"] else "no audio track")

    integrated, true_peak = media.measure_loudness(video)
    target = plan.audio.target_lufs
    add("loudness", abs(integrated - target) <= cfg["audio"]["loudness_tolerance"],
        f"{integrated:.1f} LUFS (target {target})")
    add("audio_clipping", true_peak <= -0.5, f"true peak {true_peak:.1f} dBTP")
    add("audio_level", integrated > -40, f"{integrated:.1f} LUFS", warn=False)

    black = media.run(["ffmpeg", "-hide_banner", "-t", "1.0", "-i", str(video), "-an",
                       "-vf", "blackdetect=d=0.05:pix_th=0.10", "-f", "null", "-"], check=False)
    starts = [float(s) for s in re.findall(r"black_start:\s*([\d.]+)", black.stderr)]
    add("black_frames_opening", not any(s < 0.5 for s in starts), f"black at {starts}" if starts else "none")

    edges = _edge_frames(video, info["duration"])
    bars = False
    if edges.size:
        top, bottom = edges[:, :6, :].mean(axis=(1, 2)), edges[:, -6:, :].mean(axis=(1, 2))
        left, right = edges[:, :, :4].mean(axis=(1, 2)), edges[:, :, -4:].mean(axis=(1, 2))
        bars = bool(((top < 3) & (bottom < 3)).all() or ((left < 3) & (right < 3)).all())
    add("letterbox", not bars, "black bars on every sampled frame" if bars else "no bars")

    # the platform safe area (and its like/comment rail) is a vertical-video concept; square and wide reels
    # are laid out against a plain margin, so judge them by the same rule they were built with
    vertical = plan.output.width < plan.output.height
    safe = cfg["safe_area"] if vertical else {"top": 0.06, "bottom": 0.10, "left": 0.06, "right": 0.06}
    rail = safe.get("right_rail")
    bad = []
    for card in plan.text_cards:
        x0, y0, x1, y1 = card.bbox_norm
        inside = x0 >= safe["left"] - 1e-3 and x1 <= 1 - safe["right"] + 1e-3 and \
            y0 >= safe["top"] - 1e-3 and y1 <= 1 - safe["bottom"] + 1e-3
        if rail and y1 > rail["y_from"] and x1 > rail["x_from"] + 1e-3:
            inside = False
        if not inside or len(card.lines) > 2:
            bad.append(card.text)
    add("caption_safe_area", not bad, f"outside safe area: {bad}" if bad else f"{len(plan.text_cards)} cards inside")

    planned_hold = plan.segments[-1].hold_after
    freeze = media.run(["ffmpeg", "-hide_banner", "-i", str(video), "-an", "-vf", "freezedetect=n=-60dB:d=0.6",
                        "-f", "null", "-"], check=False)
    freezes = [float(s) for s in re.findall(r"freeze_start:\s*([\d.]+)", freeze.stderr)]
    unplanned = [f for f in freezes if not (planned_hold and f >= plan.duration - planned_hold - 0.7)]
    add("frozen_frames", not unplanned, f"unplanned freezes at {unplanned}" if unplanned else "none unplanned",
        warn=True)

    status = plan.rights.get("status", "unknown")
    add("rights", status != "unknown", f"rights status '{status}'")
    return checks


def passed(checks: list[Check]) -> bool:
    return not any(c.status == "FAIL" for c in checks)


def as_dicts(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]

"""ffmpeg / ffprobe helpers. Paths always go in as argv items, never through a shell."""
from __future__ import annotations

import json
import re
import subprocess  # nosec B404
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from .paths import child_env

STANDARD_RATES = [Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
                  Fraction(30), Fraction(50), Fraction(60000, 1001), Fraction(60)]


class MediaError(RuntimeError):
    pass


def run(cmd, *, timeout: float = 7200, cwd=None, check: bool = True, binary: bool = False,
        stdin_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    binary = binary or stdin_bytes is not None
    kwargs = {} if binary else {"text": True, "encoding": "utf-8", "errors": "replace"}
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=child_env(), capture_output=True,  # nosec B603
                          timeout=timeout, input=stdin_bytes, **kwargs)
    if check and proc.returncode != 0:
        err = proc.stderr if not binary else proc.stderr.decode("utf-8", "replace")
        raise MediaError(f"{Path(str(cmd[0])).name} exited {proc.returncode}: {err[-2000:]}")
    return proc


def probe(path: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
               "-show_streams", str(path)], timeout=120).stdout
    return json.loads(out)


def output_rate(rate: Fraction) -> Fraction:
    """Keep standard rates exactly; send >60 fps to 60; snap odd rates to the nearest standard."""
    if rate > 61:
        return Fraction(60)
    if rate <= 0:
        return Fraction(30)
    return min(STANDARD_RATES, key=lambda s: abs(float(s) - float(rate)))


def video_info(path: Path) -> dict:
    data = probe(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream in {path}")
    width, height = int(video["width"]), int(video["height"])
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(side["rotation"])
    if abs(rotation) in (90, 270):
        width, height = height, width
    r_rate = Fraction(video.get("r_frame_rate", "0/1") or "0/1")
    avg = video.get("avg_frame_rate", "0/0")
    avg_rate = Fraction(avg) if avg not in ("0/0", "", None) else r_rate
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    vfr = bool(avg_rate) and abs(float(r_rate) - float(avg_rate)) > 0.5
    rate = output_rate(avg_rate if vfr else (r_rate or avg_rate))
    return {
        "width": width, "height": height, "duration": duration,
        "fps": f"{rate.numerator}/{rate.denominator}",
        "source_fps": float(avg_rate or r_rate), "vfr": vfr,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def detect_content_box(path: Path, duration: float, width: int, height: int) -> tuple[int, int, int, int]:
    """Majority cropdetect over five sampled windows. Returns (w, h, x, y); full frame if no bars."""
    votes: Counter = Counter()
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        start = max(0.0, duration * frac - 1.0)
        proc = run(["ffmpeg", "-hide_banner", "-ss", f"{start:.2f}", "-t", "2", "-i", str(path),
                    "-vf", "cropdetect=24:2:0", "-an", "-f", "null", "-"], timeout=300, check=False)
        for m in re.finditer(r"crop=(\d+):(\d+):(\d+):(\d+)", proc.stderr):
            votes[tuple(int(v) for v in m.groups())] += 1
    if not votes:
        return (width, height, 0, 0)
    w, h, x, y = votes.most_common(1)[0][0]
    if w * h >= 0.92 * width * height or w < 64 or h < 64:
        return (width, height, 0, 0)
    return (w - w % 2, h - h % 2, x, y)


def measure_loudness(path: Path, extra_input: list[str] | None = None, cwd=None) -> tuple[float, float]:
    """(integrated LUFS, true peak dBTP) via ebur128."""
    inputs = extra_input or ["-i", str(path)]
    proc = run(["ffmpeg", "-hide_banner", "-nostats", *inputs, "-vn", "-af", "ebur128=peak=true",
                "-f", "null", "-"], check=False, cwd=cwd)
    summary = proc.stderr.split("Summary:")[-1]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    peak = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary)
    integrated = float(i.group(1)) if i else -99.0
    true_peak = float(peak.group(1)) if peak and peak.group(1) != "-inf" else -99.0
    return integrated, true_peak


def decode_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    proc = run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
                "-f", "f32le", "-"], binary=True, timeout=7200, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def gray_frames(path: Path, fps: int, size: tuple[int, int], crop: tuple[int, int, int, int]):
    """Yield (index, frame) grayscale uint8 frames from one decode pass."""
    w, h = size
    cw, ch, cx, cy = crop
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-an",
           "-vf", f"crop={cw}:{ch}:{cx}:{cy},fps={fps},scale={w}:{h},format=gray",
           "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=child_env())  # nosec B603
    frame_bytes = w * h
    index = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield index, np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
            index += 1
    finally:
        proc.stdout.close()
        proc.wait(timeout=60)

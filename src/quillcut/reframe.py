"""Where to point a vertical crop: follow the person who is speaking.

Samples frames inside each shot, detects faces, and picks the active one. With one face the crop follows it;
with several, the face whose mouth region moves most is the speaker. No faces -> None (caller centres).
Detection is MediaPipe on CPU through models.faces; if it is not installed everything falls back to centred.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import media
from .models.faces import detect_faces

SAMPLE_FPS = 3
SAMPLE_W = 480


def sample_frames(video: Path, start: float, end: float, content_box: tuple[int, int, int, int],
                  fps: int = SAMPLE_FPS) -> tuple[np.ndarray, float]:
    """RGB frames of the content area between start and end. Returns (frames[n,h,w,3], seconds_per_frame)."""
    w, h, x, y = content_box
    sw = SAMPLE_W
    sh = max(2, int(round(sw * h / w / 2)) * 2)
    proc = media.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{max(0.2, end - start):.3f}",
                      "-i", str(video), "-vf", f"crop={w}:{h}:{x}:{y},fps={fps},scale={sw}:{sh}",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], binary=True, check=False)
    n = len(proc.stdout) // (sw * sh * 3)
    if n == 0:
        return np.zeros((0, sh, sw, 3), dtype=np.uint8), 1.0 / fps
    return np.frombuffer(proc.stdout[: n * sw * sh * 3], dtype=np.uint8).reshape(n, sh, sw, 3), 1.0 / fps


def _mouth_activity(frames: np.ndarray, track: list[tuple[int, tuple[float, float, float, float]]]) -> float:
    """Mean absolute change of the lower part of a face box between consecutive sampled frames."""
    diffs, prev = [], None
    for idx, (x0, y0, x1, y1) in track:
        h, w = frames.shape[1:3]
        ya = int(np.clip(y0 + 0.55 * (y1 - y0), 0, 1) * h)
        yb = int(np.clip(y1, 0, 1) * h)
        xa, xb = int(np.clip(x0 + 0.2 * (x1 - x0), 0, 1) * w), int(np.clip(x1 - 0.2 * (x1 - x0), 0, 1) * w)
        if yb - ya < 4 or xb - xa < 4:
            continue
        patch = frames[idx, ya:yb, xa:xb].astype(np.float32).mean(axis=2)
        if prev is not None and prev.shape == patch.shape:
            diffs.append(float(np.abs(patch - prev).mean()))
        prev = patch
    return float(np.mean(diffs)) if diffs else 0.0


def pick_active_center(frames: np.ndarray, detections: list[list[tuple[float, float, float, float]]]) -> float | None:
    """Normalised x of the face to keep in frame, from per-frame face boxes. None when nobody was found."""
    seen = [(i, b) for i, boxes in enumerate(detections) for b in boxes]
    if not seen:
        return None
    # group boxes into tracks by horizontal position
    tracks: list[list[tuple[int, tuple[float, float, float, float]]]] = []
    for i, box in seen:
        cx = (box[0] + box[2]) / 2
        for tr in tracks:
            last = tr[-1][1]
            if abs((last[0] + last[2]) / 2 - cx) < 0.12 and tr[-1][0] != i:
                tr.append((i, box))
                break
        else:
            tracks.append([(i, box)])
    tracks = [t for t in tracks if len(t) >= max(1, len(detections) // 4)] or tracks
    if len(tracks) == 1:
        t = tracks[0]
        return float(np.mean([(b[0] + b[2]) / 2 for _, b in t]))
    best = max(tracks, key=lambda t: (_mouth_activity(frames, t), len(t)))
    return float(np.mean([(b[0] + b[2]) / 2 for _, b in best]))


WINDOW_SECONDS = 4.0     # shots this close together share one face analysis


def windows(shots: list[tuple[float, float]], span: float = WINDOW_SECONDS) -> list[list[int]]:
    """Group shot indices into runs of contiguous shots covering at most `span` seconds."""
    groups: list[list[int]] = []
    for i, (a, b) in enumerate(shots):
        if groups:
            first = shots[groups[-1][0]]
            last = shots[groups[-1][-1]]
            if abs(a - last[1]) < 0.6 and b - first[0] <= span:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


def crop_centers(video: Path, shots: list[tuple[float, float]], content_box: tuple[int, int, int, int],
                 detector=detect_faces) -> list[float | None]:
    """One normalised crop centre per shot (None = centre it).

    Face analysis runs once per ~4 s window of contiguous shots, then each shot takes its window's speaker,
    so a reel cut every second or two does not pay for a detector per cut."""
    out: list[float | None] = [None] * len(shots)
    for group in windows(shots):
        start, end = shots[group[0]][0], shots[group[-1]][1]
        frames, _ = sample_frames(video, start, end, content_box)
        if len(frames) == 0:
            continue
        dets = detector(list(frames))
        center = pick_active_center(frames, dets) if dets is not None else None
        for i in group:
            out[i] = center
    return out


def crop_x_pixels(box_w: int, box_h: int, aspect: float, center: float | None) -> tuple[int, int, int]:
    """(crop_w, crop_h, x) in content-box pixels for an output aspect (w/h), centred on `center` (0-1)."""
    if box_w / box_h >= aspect:
        cw, ch = int(box_h * aspect) // 2 * 2, box_h
    else:
        cw, ch = box_w, int(box_w / aspect) // 2 * 2
    c = 0.5 if center is None else min(max(center, 0.0), 1.0)
    x = int(round(c * box_w - cw / 2))
    x = max(0, min(box_w - cw, x)) // 2 * 2
    return cw, ch, x

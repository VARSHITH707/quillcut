"""Cheap signals: audio loudness/flux, frame motion/flash/cuts, fused into one intensity curve."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import media

HZ = 20          # analysis grid
SR = 16000


def robust_z(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return np.clip((x - med) / (mad + 1e-6), -3, 8)


def smooth(x: np.ndarray, seconds: float, hz: float = HZ) -> np.ndarray:
    n = max(1, int(seconds * hz))
    if x.size < n:
        return x.copy()
    return np.convolve(x, np.ones(n) / n, mode="same")


def audio_features(audio: np.ndarray, n_frames: int) -> dict:
    hop = SR // HZ
    out = {"rms_db": np.full(n_frames, -90.0), "flux": np.zeros(n_frames)}
    n = min(n_frames, audio.size // hop)
    if n == 0:
        return out
    frames = audio[: n * hop].reshape(n, hop).astype(np.float64)
    out["rms_db"][:n] = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-5)
    window = np.hanning(hop)
    prev = None
    flux = np.zeros(n)
    for start in range(0, n, 20000):
        block = np.log1p(np.abs(np.fft.rfft(frames[start:start + 20000] * window, axis=1)))
        ref = np.vstack([block[:1] if prev is None else prev[None, :], block[:-1]])
        flux[start:start + len(block)] = np.maximum(0, block - ref).sum(axis=1)
        prev = block[-1]
    out["flux"][:n] = flux
    return out


def video_features(path: Path, crop: tuple[int, int, int, int], fps: int) -> dict:
    motion, luma, hist_diff = [], [], []
    prev, prev_hist = None, None
    for _, frame in media.gray_frames(path, fps, (160, 90), crop):
        f = frame.astype(np.float32)
        hist = np.histogram(frame, bins=32, range=(0, 256))[0].astype(np.float32)
        hist /= hist.sum() + 1e-6
        motion.append(0.0 if prev is None else float(np.abs(f - prev).mean()))
        hist_diff.append(0.0 if prev_hist is None else float(np.abs(hist - prev_hist).sum()))
        luma.append(float(f.mean()))
        prev, prev_hist = f, hist
    return {k: np.array(v, dtype=np.float64) for k, v in
            (("motion", motion), ("luma", luma), ("hist_diff", hist_diff))}


def fuse(audio_feat: dict, video_feat: dict, duration: float, video_fps: int) -> dict:
    n = int(duration * HZ)
    t = np.arange(n) / HZ
    tv = np.arange(len(video_feat["motion"])) / video_fps

    def on_grid(x):
        return np.interp(t, tv, x) if len(x) else np.zeros(n)

    motion = on_grid(video_feat["motion"])
    flash = on_grid(np.maximum(0, np.diff(video_feat["luma"], prepend=video_feat["luma"][:1])))
    rms = audio_feat["rms_db"][:n]
    loud_rise = np.maximum(0, rms - smooth(rms, 3.0))
    has_audio = bool(np.any(rms > -80))
    composite = (0.30 * robust_z(motion) + 0.15 * robust_z(flash)
                 + (0.35 * robust_z(loud_rise) + 0.20 * robust_z(audio_feat["flux"][:n]) if has_audio else 0))
    composite = smooth(composite, 1.0)
    lo, hi = np.percentile(composite, [5, 99.5]) if composite.size else (0, 1)
    intensity = np.clip((composite - lo) / (hi - lo + 1e-6), 0, 1)
    hd = video_feat["hist_diff"]
    cut_idx = np.where((robust_z(hd) > 5) & (hd > 0.3))[0] if hd.size else []
    return {"t": t, "intensity": intensity, "rms_db": rms, "motion": motion,
            "cuts": [round(i / video_fps, 2) for i in cut_idx]}


def find_peaks(t: np.ndarray, x: np.ndarray, *, top_n: int, min_sep: float,
               lo: float, hi: float, margin: float = 2.0) -> list[tuple[float, float]]:
    """Greedy non-max suppression on the smoothed curve inside [lo+margin, hi-margin]."""
    mask = (t >= lo + margin) & (t <= hi - margin)
    order = np.argsort(-x)
    picked: list[tuple[float, float]] = []
    for i in order:
        if not mask[i]:
            continue
        if all(abs(t[i] - p) >= min_sep for p, _ in picked):
            picked.append((round(float(t[i]), 2), round(float(x[i]), 4)))
            if len(picked) >= top_n:
                break
    return picked

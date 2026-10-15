"""Speaker separation from voice features (numpy only, no model download).

Each sentence gets a small voice fingerprint (average spectral shape of its voiced frames plus median pitch).
Two clusters are accepted only when they are clearly separated; otherwise the video is treated as one voice.
This is an estimate, not a speaker-identification model: it separates "the person asking" from "the person
answering" in an interview, which is what clipping needs. Labels are "A"/"B" in order of first appearance.
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 400          # 25 ms
HOP = 160            # 10 ms
BANDS = 24
MIN_VOICED_FRAMES = 12
SEPARATION_MIN = 2.2     # distance between centroids / average spread
MIN_CLUSTER_FRACTION = 0.05      # an interviewer may speak only a few sentences; the separation test guards false splits
PITCH_WEIGHT = 6.0


def _frames(x: np.ndarray) -> np.ndarray:
    if len(x) < FRAME:
        return np.zeros((0, FRAME))
    n = 1 + (len(x) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] * np.hanning(FRAME)


def _band_edges() -> np.ndarray:
    return np.unique(np.round(np.geomspace(2, FRAME // 2, BANDS + 1)).astype(int))


def _pitch(frame: np.ndarray) -> float | None:
    ac = np.correlate(frame, frame, mode="full")[FRAME - 1:]
    if ac[0] <= 1e-8:
        return None
    lo, hi = SR // 400, SR // 70
    seg = ac[lo:hi]
    k = int(np.argmax(seg))
    if seg[k] / ac[0] < 0.35:
        return None
    return SR / (lo + k)


def fingerprint(audio: np.ndarray) -> np.ndarray | None:
    """Voice vector for a stretch of audio, or None when there is too little voiced signal."""
    fr = _frames(audio.astype(np.float64))
    if len(fr) == 0:
        return None
    energy = np.sqrt((fr ** 2).mean(axis=1))
    voiced = fr[energy > max(0.01, 0.35 * np.median(energy[energy > 0]) if np.any(energy > 0) else 0.01)]
    if len(voiced) < MIN_VOICED_FRAMES:
        return None
    spec = np.abs(np.fft.rfft(voiced, axis=1)) ** 2
    edges = _band_edges()
    bands = np.stack([spec[:, a:b].mean(axis=1) if b > a else spec[:, a] for a, b in zip(edges[:-1], edges[1:])], axis=1)
    logb = np.log(bands + 1e-9)
    logb -= logb.mean(axis=1, keepdims=True)         # shape only; level depends on mic distance
    pitches = [p for p in (_pitch(f) for f in voiced[:: max(1, len(voiced) // 60)]) if p]
    f0 = np.median(pitches) if pitches else 0.0
    return np.concatenate([logb.mean(axis=0), [np.log(f0 + 1.0) * 1.5]])


def _kmeans2(x: np.ndarray, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    a = int(np.argmax(np.linalg.norm(x - x.mean(axis=0), axis=1)))
    b = int(np.argmax(np.linalg.norm(x - x[a], axis=1)))
    c = np.stack([x[a], x[b]])
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(x[:, None, :] - c[None, :, :], axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for k in (0, 1):
            if np.any(labels == k):
                c[k] = x[labels == k].mean(axis=0)
    return labels, c


def assign(audio: np.ndarray, spans: list[tuple[float, float]], protect: set[int] | None = None) -> list[str | None]:
    """Label each (start, end) span with 'A' or 'B'. All 'A' when the voices are not clearly different.

    `protect` holds span indices that keep their measured voice even when they sit alone between two spans of
    the other voice (an interviewer's question inside a stretch of answers)."""
    protect = protect or set()
    feats: list[np.ndarray | None] = []
    for s, e in spans:
        a, b = int(s * SR), int(e * SR)
        feats.append(fingerprint(audio[a:b]) if b - a >= FRAME * 3 else None)
    ok = [i for i, f in enumerate(feats) if f is not None]
    if len(ok) < 4:
        return ["A"] * len(spans)
    x = np.stack([feats[i] for i in ok])
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
    x[:, -1] *= PITCH_WEIGHT        # pitch is the most reliable cue between two different voices
    labels, c = _kmeans2(x)
    sizes = [(labels == k).sum() for k in (0, 1)]
    if min(sizes) < max(2, MIN_CLUSTER_FRACTION * len(ok)):
        return ["A"] * len(spans)
    spread = np.mean([np.linalg.norm(x[labels == k] - c[k], axis=1).mean() for k in (0, 1)])
    if np.linalg.norm(c[0] - c[1]) / (spread + 1e-6) < SEPARATION_MIN:
        return ["A"] * len(spans)

    raw: list[int | None] = [None] * len(spans)
    for i, lab in zip(ok, labels):
        raw[i] = int(lab)
    for i, v in enumerate(raw):                      # spans too short for a fingerprint take a neighbour's voice
        if v is None:
            near = next((raw[j] for d in range(1, len(raw)) for j in (i - d, i + d)
                         if 0 <= j < len(raw) and raw[j] is not None), 0)
            raw[i] = near
    # one isolated sentence between two sentences of the same voice is a misread, not a speaker change
    smooth = list(raw)
    for i in range(1, len(raw) - 1):
        if raw[i - 1] == raw[i + 1] != raw[i] and i not in protect:
            smooth[i] = raw[i - 1]
    first = smooth[0]
    return ["A" if v == first else "B" for v in smooth]

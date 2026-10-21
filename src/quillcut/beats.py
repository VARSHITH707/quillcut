"""Beat tracking and beat-aligned cutting, numpy only.

1. `analyse(path)` finds the tempo and the beat times of a music track (onset strength from spectral flux,
   tempo from its autocorrelation, beats by dynamic programming as in Ellis 2007).
2. `align(...)` slides the track against a reel so that as many beats as possible land in the gaps between
   spoken words, then picks cut times on those beats at the requested cadence.

A cut is only ever placed between two words: when a beat falls inside a word it is skipped, not forced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import media

SR = 22050
HOP = 512
N_FFT = 2048
FPS = SR / HOP                     # onset frames per second
TEMPO_RANGE = (60.0, 180.0)
TEMPO_PRIOR_BPM = 110.0
ON_BEAT_TOL = 0.06                 # a cut may sit this far (s) from its beat to fall inside a word gap
MIN_GAP = 0.03                     # gaps shorter than this between words are not cut points


@dataclass
class Track:
    bpm: float
    beats: list[float]             # seconds, track time
    duration: float
    start: float                   # where the music really starts (skips a quiet intro)

    def as_dict(self) -> dict:
        return {"bpm": round(self.bpm, 2), "beats": [round(b, 3) for b in self.beats],
                "duration": round(self.duration, 3), "start": round(self.start, 3)}

    @classmethod
    def from_dict(cls, d: dict) -> Track:
        return cls(bpm=d["bpm"], beats=list(d["beats"]), duration=d["duration"], start=d.get("start", 0.0))


# --------------------------------------------------------------------------------------------- analysis

def onset_envelope(y: np.ndarray) -> np.ndarray:
    if y.size < N_FFT:
        return np.zeros(0)
    n = 1 + (y.size - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n)[:, None]
    window = np.hanning(N_FFT)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    bands = [freqs < 180.0, (freqs >= 180.0) & (freqs < 2500.0), freqs >= 2500.0]
    flux = np.zeros((len(bands), n))
    prev = None
    for start in range(0, n, 4000):                 # chunked so long tracks stay light on memory
        frames = y[idx[start:start + 4000]] * window
        mag = np.log1p(10.0 * np.abs(np.fft.rfft(frames, axis=1)))
        ref = np.vstack([mag[:1] if prev is None else prev[None, :], mag[:-1]])
        rise = np.maximum(0.0, mag - ref)
        for b, mask in enumerate(bands):
            flux[b, start:start + len(mag)] = rise[:, mask].mean(axis=1)
        prev = mag[-1]
    # Each band is judged on its own scale, then the kick / 808 band counts most: it carries the beat in the
    # music this is cut to, and broadband hi-hats would otherwise pull the tracker onto the off-beat.
    k = int(FPS)
    env = np.zeros(n)
    for b, weight in enumerate((3.0, 1.0, 0.4)):
        trend = np.convolve(flux[b], np.ones(k) / k, mode="same")
        band = np.maximum(0.0, flux[b] - trend)
        env += weight * band / (band.std() + 1e-9)
    return env / (env.std() + 1e-9)


def estimate_period(env: np.ndarray) -> float:
    """Beat period in onset frames, from the autocorrelation with a mild preference for ~110 BPM."""
    n = env.size
    ac = np.fft.irfft(np.abs(np.fft.rfft(env, 2 * n)) ** 2)[:n]
    lo = int(FPS * 60.0 / TEMPO_RANGE[1])
    hi = min(n - 2, int(FPS * 60.0 / TEMPO_RANGE[0]))
    if hi <= lo + 2:
        return FPS * 60.0 / TEMPO_PRIOR_BPM
    lags = np.arange(lo, hi + 1)
    bpm = 60.0 * FPS / lags
    prior = np.exp(-0.5 * (np.log2(bpm / TEMPO_PRIOR_BPM) / 1.0) ** 2)
    score = ac[lo:hi + 1] * prior
    k = int(np.argmax(score))
    if 0 < k < len(score) - 1:                      # parabolic refinement between lags
        a, b, c = score[k - 1], score[k], score[k + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom else 0.0
        return float(lags[k] + shift)
    return float(lags[k])


def track_beats(env: np.ndarray, period: float, tightness: float = 100.0) -> list[int]:
    """Dynamic-programming beat tracker: strong onsets, spaced close to one period."""
    n = env.size
    if n == 0:
        return []
    score = np.zeros(n)
    back = -np.ones(n, dtype=int)
    p = period
    for t in range(n):
        lo, hi = int(t - 2 * p), int(t - p / 2)
        best, arg = 0.0, -1
        if hi > 0:
            prev = np.arange(max(0, lo), hi)
            if prev.size:
                cost = -tightness * np.log((t - prev) / p) ** 2
                cand = score[prev] + cost
                j = int(np.argmax(cand))
                best, arg = float(cand[j]), int(prev[j])
        score[t] = env[t] + max(0.0, best)
        back[t] = arg if best > 0 else -1
    tail = int(max(0, n - p))
    t = tail + int(np.argmax(score[tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return beats[::-1]


def analyse(path) -> Track:
    """Tempo and beat times of a music file."""
    y = media.decode_audio(path, sample_rate=SR).astype(np.float64)
    if y.size < SR * 2:
        raise ValueError("That track is too short to find a beat.")
    env = onset_envelope(y)
    period = estimate_period(env)
    frames = track_beats(env, period)
    # an onset shows up as the rise into the frame centred one hop after it starts: measured -24 ms without
    # the extra hop on synthetic kicks at 72-150 BPM, about 0 ms with it
    beats = [(f * HOP + N_FFT / 2 + HOP) / SR for f in frames]
    # skip a quiet intro: start at the first beat after the track reaches a quarter of its median loudness
    win = SR // 2                                   # half-second loudness, as a running mean (linear time)
    c = np.concatenate([[0.0], np.cumsum(y ** 2)])
    rms = np.sqrt(np.maximum(0.0, (c[win:] - c[:-win]) / win))
    voiced = rms[rms > 0]
    threshold = 0.25 * float(np.median(voiced)) if voiced.size else 0.0
    loud = np.where(rms > threshold)[0]
    first = loud[0] / SR if loud.size else 0.0
    start = next((b for b in beats if b >= first), beats[0] if beats else 0.0)
    return Track(bpm=60.0 * FPS / period, beats=beats, duration=y.size / SR, start=start)


# --------------------------------------------------------------------------------------------- alignment

def word_gaps(words: list[dict], min_gap: float = MIN_GAP) -> list[tuple[float, float]]:
    """Clip-time intervals between consecutive spoken words where a cut cannot clip a word."""
    out = []
    for a, b in zip(words, words[1:]):
        if b["start"] - a["end"] >= min_gap:
            out.append((a["end"], b["start"]))
    return out


def _snap_to_gap(t: float, gaps: list[tuple[float, float]], tol: float) -> float | None:
    """The nearest point inside a word gap, if it is within `tol` of the beat; None otherwise."""
    for g0, g1 in gaps:
        if g0 - tol <= t <= g1 + tol:
            snapped = min(max(t, g0 + 0.005), g1 - 0.005) if g1 - g0 > 0.01 else (g0 + g1) / 2
            return snapped if abs(snapped - t) <= tol else None
        if g0 - tol > t:
            break
    return None


def align(track: Track, words: list[dict], duration: float, cadence: tuple[float, float], *,
          no_cut_before: float = 0.0, tol: float = ON_BEAT_TOL, step: float = 0.01) -> dict:
    """Choose the music offset and the cut times.

    Returns {"offset": track time at clip t=0, "cuts": clip times, "on_beat": share of cuts within `tol` of a
    beat, "beats_used": beats that became cuts}. Cuts respect the cadence (shortest shot >= cadence[0]) and
    never fall inside a word."""
    gaps = word_gaps(words)
    lo, hi = cadence
    beats = np.array(track.beats)
    if beats.size < 4 or not gaps:
        return {"offset": track.start, "cuts": [], "on_beat": 0.0, "beats_used": 0}
    period = 60.0 / track.bpm
    usable_end = max(track.start, track.duration - duration)       # the music must last the whole reel
    best = None
    for phase in np.arange(0.0, period, step):
        offset = min(track.start + phase, usable_end)
        clip_beats = beats - offset
        clip_beats = clip_beats[(clip_beats > no_cut_before) & (clip_beats < duration - 0.3)]
        hits = sum(1 for b in clip_beats if _snap_to_gap(float(b), gaps, tol) is not None)
        if best is None or hits > best[0]:
            best = (hits, offset)
    offset = best[1]
    clip_beats = [float(b - offset) for b in beats if no_cut_before < b - offset < duration - 0.3]
    cuts: list[float] = []
    last = 0.0
    for b in clip_beats:
        snapped = _snap_to_gap(b, gaps, tol)
        if snapped is None or snapped - last < lo:      # spacing is measured on the real cut times
            continue
        if duration - snapped < 0.6 * lo:
            break
        cuts.append(round(snapped, 3))
        last = snapped
    on_beat = sum(1 for c in cuts if np.min(np.abs(np.array(clip_beats) - c)) <= tol + 1e-6) / len(cuts) if cuts else 0.0
    return {"offset": round(float(offset), 3), "cuts": cuts, "on_beat": round(on_beat, 3), "beats_used": len(cuts),
            "max_gap_between_cuts": round(max([b - a for a, b in zip([0.0, *cuts], [*cuts, duration])]), 2)}


def frame_snap(t: float, fps: float) -> float:
    """Round a time to the video frame grid so every shot is a whole number of frames (no audio gaps)."""
    return round(round(t * fps) / fps, 4)


def parse_fps(fps: str | float) -> float:
    if isinstance(fps, (int, float)):
        return float(fps)
    if "/" in fps:
        n, d = fps.split("/")
        return float(n) / float(d)
    return float(fps)

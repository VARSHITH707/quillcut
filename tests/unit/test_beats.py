"""Beat detection on synthetic music and beat-aligned cuts that never land inside a word."""
from __future__ import annotations

import wave

import numpy as np
import pytest

from quillcut import beats

SR = 22050


def drum_track(bpm: float, seconds: float, path, intro: float = 0.0, swing_noise: float = 0.0) -> list[float]:
    """Kick on every beat, hi-hat on the off-beat, a soft pad underneath. Returns the true beat times."""
    t = np.arange(int(seconds * SR)) / SR
    y = 0.05 * np.sin(2 * np.pi * 110 * t)
    period = 60.0 / bpm
    truth = []
    rng = np.random.default_rng(3)
    b = intro
    while b < seconds - 0.3:
        truth.append(round(b, 4))
        n = int(b * SR)
        k = np.arange(int(0.18 * SR))
        kick = np.sin(2 * np.pi * (60 + 90 * np.exp(-k / 400)) * k / SR) * np.exp(-k / 2500)
        y[n:n + k.size] += 0.9 * kick[: max(0, min(k.size, y.size - n))]
        h = int((b + period / 2) * SR)
        hat = rng.normal(0, 1, int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / 200)
        y[h:h + hat.size] += 0.25 * hat[: max(0, min(hat.size, y.size - h))]
        b += period + rng.normal(0, swing_noise)
    if intro:
        y[: int(intro * SR)] *= 0.02
    pcm = (np.clip(y / np.abs(y).max(), -1, 1) * 30000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return truth


@pytest.mark.parametrize("bpm", [72.0, 95.0, 128.0, 150.0])
def test_tempo_and_beats_are_found(tmp_path, bpm):
    truth = drum_track(bpm, 24.0, tmp_path / "t.wav")
    tr = beats.analyse(tmp_path / "t.wav")
    assert tr.bpm == pytest.approx(bpm, rel=0.03)
    found = np.array(tr.beats)
    errors = [np.min(np.abs(found - b)) for b in truth[2:-2]]
    assert np.median(errors) < 0.03 and np.mean(np.array(errors) < 0.05) > 0.9


def test_a_quiet_intro_is_skipped(tmp_path):
    drum_track(100.0, 20.0, tmp_path / "t.wav", intro=3.0)
    tr = beats.analyse(tmp_path / "t.wav")
    assert tr.start >= 2.8


def test_track_round_trips_through_json():
    tr = beats.Track(bpm=96.0, beats=[0.5, 1.125], duration=30.0, start=0.5)
    assert beats.Track.from_dict(tr.as_dict()) == tr


def speech(seconds: float, rng_seed: int = 1) -> list[dict]:
    """Words of 0.2-0.45 s with 0.04-0.25 s gaps, like natural speech."""
    rng = np.random.default_rng(rng_seed)
    words, t = [], 0.3
    while t < seconds - 0.5:
        d = rng.uniform(0.2, 0.45)
        words.append({"word": "w", "start": round(t, 3), "end": round(t + d, 3)})
        t += d + rng.uniform(0.04, 0.25)
    return words


def test_cuts_land_on_beats_and_never_inside_a_word():
    words = speech(40.0)
    track = beats.Track(bpm=90.0, beats=[i * (60 / 90) for i in range(200)], duration=140.0, start=0.0)
    out = beats.align(track, words, 40.0, (2.0, 2.5), no_cut_before=3.0)
    cuts = out["cuts"]
    assert len(cuts) >= 8
    for c in cuts:
        assert not any(w["start"] + 1e-6 < c < w["end"] - 1e-6 for w in words), c        # never inside a word
    clip_beats = np.array(track.beats) - out["offset"]
    assert all(np.min(np.abs(clip_beats - c)) <= beats.ON_BEAT_TOL + 1e-6 for c in cuts)  # on the beat
    assert out["on_beat"] == 1.0
    assert min(b - a for a, b in zip(cuts, cuts[1:])) >= 2.0 - 1e-6                     # cadence respected
    assert cuts[0] > 3.0                                                                  # hook left alone


def test_the_music_is_long_enough_for_the_reel():
    words = speech(30.0)
    track = beats.Track(bpm=100.0, beats=[i * 0.6 for i in range(60)], duration=36.0, start=0.0)
    out = beats.align(track, words, 30.0, (2.0, 2.5))
    assert out["offset"] + 30.0 <= track.duration + 1e-6


def test_no_beats_or_no_gaps_means_no_beat_cuts():
    track = beats.Track(bpm=100.0, beats=[0.6], duration=10.0, start=0.0)
    assert beats.align(track, speech(10.0), 10.0, (2.0, 2.5))["cuts"] == []
    continuous = [{"word": "w", "start": i * 0.3, "end": i * 0.3 + 0.3} for i in range(30)]
    full = beats.Track(bpm=100.0, beats=[i * 0.6 for i in range(30)], duration=20.0, start=0.0)
    assert beats.align(full, continuous, 9.0, (2.0, 2.5))["cuts"] == []


def test_frame_snap_lands_on_whole_frames():
    for fps in (24.0, 25.0, 30000 / 1001):
        t = beats.frame_snap(12.3456, fps)
        assert abs(t * fps - round(t * fps)) < 1e-3
    assert beats.parse_fps("30000/1001") == pytest.approx(29.97, abs=0.01)

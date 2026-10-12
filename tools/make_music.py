"""Original, royalty-free music beds and SFX, synthesised with numpy (licence: CC0, ours).

Writes WAVs + manifest.yaml into data/Sound Effects and Music. Re-running overwrites.
Each track has an ambient intro, a riser, and a drop at a known time (manifest `drop`).
"""
from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import yaml

SR = 48000
OUT = Path(os.environ.get("QUILLCUT_HOME") or Path(__file__).resolve().parents[1] / "data") / "Sound Effects and Music"
RNG = np.random.default_rng(7)


def t_axis(sec: float) -> np.ndarray:
    return np.arange(int(sec * SR)) / SR


def fft_filter(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(x))


# ------------------------------------------------------------------ instruments

def kick(punch: float = 1.0) -> np.ndarray:
    t = t_axis(0.5)
    freq = 42 + 110 * np.exp(-t * 28)
    body = np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 6.5)
    click = fft_filter(RNG.standard_normal(len(t)), 2000, 8000) * np.exp(-t * 300) * 0.3
    return np.tanh((body + click) * 1.6 * punch) * 0.9


def clap() -> np.ndarray:
    t = t_axis(0.35)
    noise = fft_filter(RNG.standard_normal(len(t)), 900, 7000)
    env = np.zeros_like(t)
    for off in (0.0, 0.012, 0.024):
        env += np.where(t >= off, np.exp(-(t - off) * 35), 0)
    env += np.exp(-t * 9) * 0.5
    return noise / np.max(np.abs(noise)) * env * 0.55


def hat(open_: bool = False) -> np.ndarray:
    t = t_axis(0.25 if open_ else 0.06)
    noise = fft_filter(RNG.standard_normal(len(t)), 7000, 16000)
    return noise / np.max(np.abs(noise)) * np.exp(-t * (14 if open_ else 70)) * 0.22


def bass808(freq: float, length: float, glide_to: float | None = None) -> np.ndarray:
    t = t_axis(length)
    f = np.full_like(t, freq) if glide_to is None else freq + (glide_to - freq) * np.clip(t / 0.12, 0, 1)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR)
    env = np.minimum(1, t / 0.005) * np.exp(-t * 1.6)
    return np.tanh(tone * env * 2.2) * 0.7


def cowbell(freq: float) -> np.ndarray:
    t = t_axis(0.35)
    sq = np.sign(np.sin(2 * np.pi * freq * t)) + np.sign(np.sin(2 * np.pi * freq * 1.48 * t))
    return fft_filter(sq, 350, 4500) * np.exp(-t * 11) * 0.16


def pad(freqs: list[float], length: float, bright: float = 900) -> np.ndarray:
    t = t_axis(length)
    x = np.zeros_like(t)
    for f in freqs:
        for det in (-0.4, 0.0, 0.45):
            ph = RNG.uniform(0, 1)
            x += 2 * ((t * (f + det) + ph) % 1) - 1          # saw
    x = fft_filter(x, 40, bright)
    env = np.minimum(1, t / 1.2) * np.minimum(1, (length - t) / 1.2)
    return x / (np.max(np.abs(x)) + 1e-9) * env * 0.22


def drone(freq: float, length: float) -> np.ndarray:
    t = t_axis(length)
    x = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * freq * 2.01 * t)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t)
    return x * np.minimum(1, t / 2) * 0.18


def riser(length: float) -> np.ndarray:
    t = t_axis(length)
    noise = RNG.standard_normal(len(t))
    out = np.zeros_like(t)
    steps = 12
    for k in range(steps):                      # stepped band rising in pitch
        a, b = int(k * len(t) / steps), int((k + 1) * len(t) / steps)
        lo = 300 * 2 ** (k * 4 / steps)
        out[a:b] = fft_filter(noise[a:b], lo, lo * 3)
    sweep = np.sin(2 * np.pi * np.cumsum(200 + 1800 * (t / length) ** 2) / SR) * 0.25
    return (out / (np.max(np.abs(out)) + 1e-9) * 0.6 + sweep) * (t / length) ** 1.8 * 0.6


def impact() -> np.ndarray:
    t = t_axis(2.2)
    sub = np.sin(2 * np.pi * np.cumsum(30 + 60 * np.exp(-t * 4)) / SR) * np.exp(-t * 1.8)
    crash = fft_filter(RNG.standard_normal(len(t)), 60, 5000) * np.exp(-t * 5) * 0.25
    return np.tanh((sub + crash) * 1.8) * 0.85


def whoosh(length: float = 0.6) -> np.ndarray:
    t = t_axis(length)
    noise = fft_filter(RNG.standard_normal(len(t)), 400, 6000)
    env = np.sin(np.pi * t / length) ** 2
    return noise / np.max(np.abs(noise)) * env * 0.45


def tick() -> np.ndarray:
    t = t_axis(0.03)
    return np.sin(2 * np.pi * 2400 * t) * np.exp(-t * 220) * 0.25


# ------------------------------------------------------------------ arrangement

def place(buf: np.ndarray, sound: np.ndarray, at: float, gain: float = 1.0, pan: float = 0.0) -> None:
    i = int(at * SR)
    if i >= len(buf):
        return
    n = min(len(sound), len(buf) - i)
    buf[i:i + n, 0] += sound[:n] * gain * (1 - max(0, pan))
    buf[i:i + n, 1] += sound[:n] * gain * (1 + min(0, pan))


def phonk_track(bpm: float, length: float, root: float, drop_bar: int, cowbell_notes: list[float],
                half_time: bool = False) -> tuple[np.ndarray, float]:
    beat = 60 / bpm
    bar = 4 * beat
    buf = np.zeros((int(length * SR), 2))
    drop = drop_bar * bar
    d = drone(root * 2, length)
    buf[:, 0] += d
    buf[:, 1] += d * 0.95
    for b in range(int(drop / beat)):                       # intro: ticking clock
        place(buf, tick(), b * beat, 0.8, pan=0.3 if b % 2 else -0.3)
    place(buf, riser(bar), drop - bar, 0.9)
    place(buf, impact(), drop, 0.8)
    n_bars = int((length - drop) / bar)
    prog = [root, root, root * 1.189, root * 0.891]          # i, i, iii, VII-ish
    for k in range(n_bars):
        t0 = drop + k * bar
        note = prog[(k // 2) % len(prog)]
        kicks = [0, 2.5] if half_time else [0, 1.5, 2, 3.5]
        for kb in kicks:
            place(buf, kick(), t0 + kb * beat, 0.95)
        claps = [2] if half_time else [1, 3]
        for cb in claps:
            place(buf, clap(), t0 + cb * beat, 0.8)
        steps = 16 if not half_time else 8
        for h in range(steps):
            swing = 0.08 * beat * (h % 2)
            place(buf, hat(open_=(h % 8 == 6)), t0 + h * (4 * beat / steps) + swing, 0.55,
                  pan=0.25 if h % 2 else -0.25)
        place(buf, bass808(note, bar * 0.95, glide_to=note * 0.5 if k % 4 == 3 else None), t0, 0.9)
        if cowbell_notes:
            for i, cn in enumerate(cowbell_notes):
                place(buf, cowbell(cn), t0 + i * (beat / 2), 0.9, pan=-0.15)
        if k % 8 == 7:
            place(buf, riser(bar * 0.5), t0 + bar * 0.5, 0.4)
    return buf, drop


def ambient_track(bpm: float, length: float, root: float) -> tuple[np.ndarray, float]:
    beat = 60 / bpm
    buf = np.zeros((int(length * SR), 2))
    chords = [[root, root * 1.5, root * 2.38], [root * 0.89, root * 1.335, root * 2.0],
              [root * 1.19, root * 1.78, root * 2.67], [root, root * 1.5, root * 2.0]]
    seg = 8 * beat
    for k in range(int(length / seg) + 1):
        p = pad(chords[k % 4], seg + 1.5)
        place(buf, p, k * seg, 1.0, pan=0.1 if k % 2 else -0.1)
    for b in range(int(length / beat)):
        if b % 4 == 0:
            place(buf, kick(punch=0.5), b * beat, 0.35)
        if b % 2 == 1:
            place(buf, hat(), b * beat, 0.35)
    return buf, 0.0


def tension_track(bpm: float, length: float, root: float, drop_bar: int) -> tuple[np.ndarray, float]:
    beat = 60 / bpm
    bar = 4 * beat
    buf = np.zeros((int(length * SR), 2))
    drop = drop_bar * bar
    d = drone(root, length)
    buf[:, 0] += d
    buf[:, 1] += d
    for b in range(int(length / beat)):
        t0 = b * beat
        place(buf, kick(punch=0.6), t0, 0.45 if t0 < drop else 0.9)
        place(buf, tick(), t0 + beat / 2, 0.2)
        if t0 >= drop and b % 2 == 1:
            place(buf, clap(), t0, 0.6)
    place(buf, riser(bar * 2), drop - 2 * bar, 0.8)
    place(buf, impact(), drop, 0.9)
    for k in range(int(length / (8 * beat)) + 1):
        place(buf, pad([root * 2, root * 3], 8 * beat + 1), k * 8 * beat, 0.6)
    return buf, drop


def upbeat_track(bpm: float, length: float, root: float, drop_bar: int) -> tuple[np.ndarray, float]:
    """Bright, positive four-on-the-floor bed for product / lifestyle reels."""
    beat = 60 / bpm
    bar = 4 * beat
    buf = np.zeros((int(length * SR), 2))
    drop = drop_bar * bar
    major = [[root, root * 1.26, root * 1.5], [root * 1.335, root * 1.68, root * 2.0],
             [root * 1.5, root * 1.89, root * 2.25], [root * 1.12, root * 1.41, root * 1.68]]
    for k in range(int(length / bar) + 1):
        place(buf, pad([f * 2 for f in major[k % 4]], bar + 0.8, bright=2600), k * bar, 1.1,
              pan=0.15 if k % 2 else -0.15)
    for b in range(int(length / beat)):
        t0 = b * beat
        building = t0 < drop
        place(buf, kick(punch=0.8), t0, 0.45 if building else 0.85)
        place(buf, hat(open_=True), t0 + beat / 2, 0.25 if building else 0.45, pan=0.2)
        if b % 2 == 1 and not building:
            place(buf, clap(), t0, 0.6)
        if not building:
            note = major[(b // 4) % 4][0] / 2
            place(buf, bass808(note, beat * 0.45), t0 + beat / 2, 0.55)
    place(buf, riser(bar), drop - bar, 0.6)
    return buf, drop


def finish(buf: np.ndarray, target_rms_db: float = -17.0) -> np.ndarray:
    if buf.ndim == 1:
        buf = np.stack([buf, buf], axis=1)
    rms = np.sqrt(np.mean(buf ** 2)) + 1e-9
    buf = buf * (10 ** (target_rms_db / 20) / rms)
    return np.tanh(buf * 1.1) / 1.1


def write(path: Path, buf: np.ndarray) -> None:
    data = (np.clip(buf, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())


def main() -> None:
    (OUT / "music").mkdir(parents=True, exist_ok=True)
    (OUT / "sfx").mkdir(parents=True, exist_ok=True)
    E1 = 41.2
    tracks = {
        "qc_phonk_150": (phonk_track(150, 100, E1, 4, [659.3, 659.3, 783.99, 659.3, 587.3, 659.3, 0, 523.3]),
                         150, 0.95, ["aggressive", "fast", "drift", "phonk"]),
        "qc_dark_trap_140": (phonk_track(140, 100, 36.7, 4, [], half_time=True),
                             140, 0.8, ["dark", "tense", "heavy"]),
        "qc_tension_120": (tension_track(120, 100, 43.65, 8), 120, 0.6, ["tense", "cinematic", "build"]),
        "qc_ambient_96": (ambient_track(96, 100, 110.0), 96, 0.3, ["calm", "atmospheric", "world"]),
        "qc_upbeat_118": (upbeat_track(118, 100, 130.81, 4), 118, 0.7, ["upbeat", "positive", "bright", "lifestyle"]),
    }
    manifest = {"music": [], "sfx": []}
    for name, ((buf, drop), bpm, energy, mood) in tracks.items():
        write(OUT / "music" / f"{name}.wav", finish(buf))
        manifest["music"].append({
            "file": f"music/{name}.wav", "bpm": bpm, "drop": round(drop, 3), "energy": energy, "mood": mood,
            "duration": round(len(buf) / SR, 2), "licence": "CC0 (original, generated by Quillcut tools/make_music.py)",
            "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "platforms_allowed": ["youtube_shorts", "instagram_reels", "x"]})
    sfx = {"whoosh": whoosh(), "impact": impact(), "riser_2s": riser(2.0), "tick": tick()}
    for name, sound in sfx.items():
        write(OUT / "sfx" / f"{name}.wav", finish(sound, -14))
        manifest["sfx"].append({"file": f"sfx/{name}.wav", "licence": "CC0 (original)",
                                "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                                "platforms_allowed": ["youtube_shorts", "instagram_reels", "x"]})
    (OUT / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print("wrote", len(manifest["music"]), "tracks and", len(manifest["sfx"]), "sfx to", OUT)


if __name__ == "__main__":
    main()

"""Deterministic EditPlan executor. No creative decisions live here.

Each segment renders to a short x264 + PCM intermediate (PCM avoids AAC priming gaps at the
joins); the concat then gets captions, two-pass loudnorm and the final encode.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from . import media
from .captions import build_ass
from .types import EditPlan, Segment, SourceRef

OUT_W, OUT_H = 1080, 1920


def _crop_box(ref: SourceRef) -> str:
    w, h, x, y = ref.content_box
    return f"crop={w}:{h}:{x}:{y}"


def _layout_chain(layout: str, label_in: str, label_out: str) -> str:
    if layout == "center_crop":
        return (f"[{label_in}]crop='if(gte(iw/ih,9/16),trunc(ih*9/16/2)*2,iw)':"
                f"'if(gte(iw/ih,9/16),ih,trunc(iw*16/9/2)*2)',"
                f"scale={OUT_W}:{OUT_H}:flags=lanczos[{label_out}]")
    if layout == "wide_fill":
        return (f"[{label_in}]split=2[bg0][fg0];"
                f"[bg0]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
                f"boxblur=24:2,eq=brightness=-0.18[bg];"
                f"[fg0]crop='min(iw,trunc(ih*4/5/2)*2)':ih,scale={OUT_W}:-2:flags=lanczos[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[{label_out}]")
    if layout == "fit_blur":
        return (f"[{label_in}]split=2[bg0][fg0];"
                f"[bg0]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
                f"boxblur=30:2,eq=brightness=-0.22[bg];"
                f"[fg0]scale={OUT_W}:-2:flags=lanczos[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[{label_out}]")
    raise ValueError(layout)


def _half(label_in: str, label_out: str) -> str:
    return (f"[{label_in}]crop='if(gte(iw/ih,9/8),trunc(ih*9/8/2)*2,iw)':"
            f"'if(gte(iw/ih,9/8),ih,trunc(iw*8/9/2)*2)',scale={OUT_W}:{OUT_H // 2}:flags=lanczos[{label_out}]")


def segment_command(seg: Segment, plan: EditPlan, out: Path) -> list[str]:
    src = plan.sources[seg.source]
    span = seg.src_out - seg.src_in
    vsrc = plan.sources[seg.video_source] if seg.video_source else src
    v_in = seg.video_in if seg.video_source else seg.src_in
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{v_in:.3f}", "-t", f"{span:.3f}", "-i", vsrc.path]
    inputs = 1
    if seg.layout == "split":
        second = plan.sources[seg.second_source]
        cmd += ["-ss", f"{seg.second_in:.3f}", "-t", f"{span:.3f}", "-i", second.path]
        inputs = 2
    audio_in = "0:a"
    if seg.video_source and src.has_audio:
        cmd += ["-ss", f"{seg.src_in:.3f}", "-t", f"{span:.3f}", "-i", src.path]
        audio_in = f"{inputs}:a"
        inputs += 1
    elif not src.has_audio:
        cmd += ["-f", "lavfi", "-t", f"{span:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_in = f"{inputs}:a"

    speed = f"setpts=(PTS-STARTPTS)/{seg.speed}"
    tail = f"fps={plan.output.fps},format=yuv420p,setsar=1"
    parts = [f"[0:v]{_crop_box(vsrc)},{speed}[v0]"]
    if seg.layout == "split":
        parts.append(f"[1:v]{_crop_box(plan.sources[seg.second_source])},{speed}[v1]")
        parts += [_half("v0", "top"), _half("v1", "bot"), "[top][bot]vstack=inputs=2[lay]"]
    else:
        parts.append(_layout_chain(seg.layout, "v0", "lay"))
    chain = "lay"
    if seg.push_in:
        dur = seg.out_duration
        parts.append(f"[lay]scale=w='trunc({OUT_W}*(1+{seg.push_in}*t/{dur:.3f})/2)*2':"
                     f"h='trunc({OUT_H}*(1+{seg.push_in}*t/{dur:.3f})/2)*2':eval=frame,"
                     f"crop={OUT_W}:{OUT_H}[pz]")
        chain = "pz"
    hold = f",tpad=stop_mode=clone:stop_duration={seg.hold_after:.3f}" if seg.hold_after else ""
    parts.append(f"[{chain}]{tail}{hold}[vout]")
    pad = f",apad=pad_dur={seg.hold_after:.3f}" if seg.hold_after else ""
    parts.append(f"[{audio_in}]asetpts=PTS-STARTPTS,atempo={seg.speed},volume={seg.audio_gain},"
                 f"aformat=sample_rates=48000:channel_layouts=stereo{pad}[aout]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]", "-map", "[aout]",
            "-t", f"{seg.out_duration:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
            "-threads", "16", "-c:a", "pcm_s16le", str(out)]
    return cmd


LIMIT_DBFS = -2.5        # sample-peak ceiling; leaves room for inter-sample peaks under -1 dBTP
LOUDNESS_STEPS = 5
LOUDNESS_TOLERANCE = 0.3


TRUE_PEAK_STEPS = 4
SR = 48000
DUCK_SMOOTH = 0.3          # seconds; gain changes ramp instead of clicking
BED_FADE_OUT = 0.8
SILENT_SOURCE_REF = -16.0  # reference loudness when the source has no audio


def _decode_stereo(inputs: list[str], cwd: Path) -> np.ndarray:
    proc = media.run(["ffmpeg", "-v", "error", *inputs, "-vn", "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                     cwd=cwd, binary=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).astype(np.float64)


def _loudness_of(x: np.ndarray, work: Path, name: str) -> float:
    path = work / name
    media.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
               "-c:a", "pcm_f32le", path.name], cwd=work, stdin_bytes=x.astype(np.float32).tobytes())
    return media.measure_loudness(path)[0]


def _mix_bed(plan: EditPlan, work: Path, concat: list[str]) -> list[str]:
    """Source + licensed music bed (ducked under speech) + SFX. Returns ffmpeg input args."""
    audio = plan.audio
    if audio.music is None and not audio.sfx:
        return concat
    src = _decode_stereo(concat, work)
    n = len(src)
    ref = media.measure_loudness(work, extra_input=concat, cwd=work)[0]
    if ref <= -70:
        ref = SILENT_SOURCE_REF
    bed = np.zeros_like(src)
    if audio.music is not None:
        track = _decode_stereo(["-i", audio.music.file], work)
        off = int(round(audio.music.start_offset * SR))
        lo, hi = max(0, -off), min(n, len(track) - off)
        if hi > lo:
            bed[lo:hi] = track[lo + off:hi + off]
            level = _loudness_of(track[max(0, off):max(0, off) + (hi - lo)], work, "bed_probe.wav")
            if level > -70:
                gain = np.full(n, 10 ** ((ref + audio.music.level_db - level) / 20))
                ducked = 10 ** ((ref + audio.music.duck_db - level) / 20)
                for a, b in audio.music.duck:
                    gain[max(0, int(a * SR)):min(n, int(b * SR))] = ducked
                win = np.hanning(int(DUCK_SMOOTH * SR))
                gain = np.convolve(gain, win / win.sum(), mode="same")
                fade = np.ones(n)
                k = min(n, int(BED_FADE_OUT * SR))
                fade[n - k:] = np.linspace(1, 0, k)
                bed *= (gain * fade)[:, None]
    for i, hit in enumerate(audio.sfx):
        sound = _decode_stereo(["-i", hit.file], work)
        level = _loudness_of(sound, work, f"sfx_probe_{i}.wav")
        if level <= -70:
            continue
        start = int(hit.t * SR)
        m = min(len(sound), n - start)
        if m > 0:
            bed[start:start + m] += sound[:m] * 10 ** ((ref + hit.gain_db - level) / 20)
    mix = src + bed
    media.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
               "-c:a", "pcm_f32le", "mix.wav"], cwd=work, stdin_bytes=mix.astype(np.float32).tobytes())
    return ["-i", "mix.wav"]


def _master_audio(concat: list[str], work: Path, target: float, tp_max: float, fade: str) -> Path:
    """Gain → peak limiter → AAC → measure the *encoded* audio → correct.

    loudnorm's linear mode cannot raise loud-peaked game audio to target without breaking true
    peak, so a limiter does that job. The AAC encoder then adds up to ~2.5 dB of overshoot on
    dense mixes, so true peak is verified on the encoded file and that exact file is muxed."""
    integrated, _ = media.measure_loudness(work, extra_input=concat, cwd=work)
    silent = integrated <= -70                 # silent source: leave it; QC rejects the clip
    gain = 0.0 if silent else target - integrated
    limit_db = LIMIT_DBFS
    wav, aac = work / "master.wav", work / "master.m4a"
    for _ in range(TRUE_PEAK_STEPS):
        for _ in range(LOUDNESS_STEPS):
            af = (f"volume={gain:.2f}dB,alimiter=limit={10 ** (limit_db / 20):.4f}"
                  f":attack=2:release=60:level=0{fade}")
            media.run(["ffmpeg", "-y", "-v", "error", *concat, "-vn", "-af", af, "-ar", "48000", "-ac", "2",
                       "-c:a", "pcm_s16le", wav.name], cwd=work)
            got, _ = media.measure_loudness(wav)
            if silent or abs(got - target) <= LOUDNESS_TOLERANCE:
                break
            gain += target - got
        media.run(["ffmpeg", "-y", "-v", "error", "-i", wav.name, "-c:a", "aac", "-b:a", "192k", "-aac_pns", "0",
                   "-flags:a", "+bitexact", aac.name], cwd=work)
        _, true_peak = media.measure_loudness(aac)
        if silent or true_peak <= tp_max - 0.3:
            break
        limit_db -= true_peak - (tp_max - 0.5)
    return aac


def render_plan(plan: EditPlan, out_path: Path, work: Path, *, preset: str = "slow", crf: int = 18,
                threads: int = 8) -> Path:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    names = []
    for i, seg in enumerate(plan.segments):
        name = f"seg_{i:03d}.mkv"
        media.run(segment_command(seg, plan, work / name), cwd=work)
        names.append(name)
    (work / "list.txt").write_text("".join(f"file '{n}'\n" for n in names), encoding="utf-8")
    (work / "captions.ass").write_text(build_ass(plan.text_cards, plan.caption_style), encoding="utf-8")

    concat = ["-f", "concat", "-safe", "0", "-i", "list.txt"]
    dur = plan.duration
    vf = []
    if plan.look:
        vf.append("eq=" + ":".join(f"{k}={v}" for k, v in sorted(plan.look.items())))
    vf.append("subtitles=captions.ass")
    if plan.progress_bar:
        vf.append(f"drawbox=x=0:y=ih-10:w='iw*t/{dur:.3f}':h=10:color=0x{plan.progress_bar}@0.95:t=fill")
    fade = ""
    if plan.audio.fade_out:
        f = plan.audio.fade_out
        vf.append(f"fade=t=out:st={dur - f:.3f}:d={f:.3f}")
        fade = f",afade=t=out:st={dur - f:.3f}:d={f:.3f}"
    mixed = _mix_bed(plan, work, concat)
    master = _master_audio(mixed, work, plan.audio.target_lufs, plan.audio.true_peak_max, fade)
    tmp_out = work / "final.mp4"
    logo_input, video_filter, video_map = [], ["-vf", ",".join(vf)], "0:v"
    if plan.logo:                    # persistent brand logo, composited over captions for the whole video
        shutil.copy(plan.logo["file"], work / "logo.png")
        w = int(round(plan.output.width * plan.logo.get("width", 0.30) / 2) * 2)
        top = int(round(plan.output.height * plan.logo.get("top", 0.145)))
        logo_input = ["-loop", "1", "-i", "logo.png"]
        video_filter = ["-filter_complex", f"[0:v]{','.join(vf)}[base];[2:v]scale={w}:-2,format=rgba[logo];"
                                           f"[base][logo]overlay=x=(W-w)/2:y={top}:format=auto[vout]"]
        video_map = "[vout]"
    media.run(["ffmpeg", "-y", "-v", "error", *concat, "-i", master.name, *logo_input,
               "-map", video_map, "-map", "1:a", *video_filter,
               "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-threads", str(threads),
               "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-r", plan.output.fps, "-c:a", "copy",
               "-t", f"{dur:.3f}", "-movflags", "+faststart", "-map_metadata", "-1",
               "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact", "final.mp4"],
              cwd=work)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_out), out_path)
    return out_path

"""Synthetic test media generated with ffmpeg. No copyrighted material."""
from __future__ import annotations

import subprocess
from pathlib import Path

MEDIA = Path(__file__).parent / "media"

BURST = "if(between(t,{a},{b})+between(t,{c},{d}),14,1)"


def _ff(args: list[str], out: Path) -> Path:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", *args, str(out)], check=True)
    return out


def synthetic(name: str, size: str, dur: int, bursts: tuple[float, float], src: str = "testsrc2") -> Path:
    b1, b2 = bursts
    flash = f"drawbox=c=white@1:t=fill:enable='between(t,{b1},{b1 + 0.4})+between(t,{b2},{b2 + 0.4})'"
    vol = BURST.format(a=b1 - 0.2, b=b1 + 3, c=b2 - 0.2, d=b2 + 3)
    return _ff(["-f", "lavfi", "-i", f"{src}=s={size}:r=30:d={dur}",
                "-f", "lavfi", "-i", f"anoisesrc=d={dur}:c=pink:a=0.03",
                "-vf", flash, "-af", f"volume='{vol}':eval=frame",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest"], MEDIA / name)


def make_all() -> dict[str, Path]:
    return {
        "a_1080": synthetic("synthetic_a_1080p_90s.mp4", "1920x1080", 90, (40.0, 70.0)),
        "b_1080": synthetic("synthetic_b_1080p_70s.mp4", "1920x1080", 70, (35.0, 55.0), src="testsrc"),
        "a_720": synthetic("synthetic_720p_70s.mp4", "1280x720", 70, (38.0, 58.0)),
        "phone": _ff(["-f", "lavfi", "-i", "testsrc2=s=360x150:r=30:d=40",
                      "-f", "lavfi", "-i", "anoisesrc=d=40:c=pink:a=0.05",
                      "-vf", "pad=360:640:0:245:black", "-c:v", "libx264", "-preset", "ultrafast",
                      "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"], MEDIA / "letterboxed_phone_40s.mp4"),
        "no_audio": _ff(["-f", "lavfi", "-i", "testsrc2=s=1920x1080:r=30:d=60", "-c:v", "libx264",
                         "-preset", "ultrafast", "-pix_fmt", "yuv420p"], MEDIA / "no_audio_60s.mp4"),
    }


if __name__ == "__main__":
    for key, path in make_all().items():
        print(key, path)

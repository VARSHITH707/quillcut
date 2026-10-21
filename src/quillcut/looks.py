"""Colour grades applied to the whole reel, footage and cutaways alike, before captions are drawn.

Each grade is an ffmpeg filter-graph fragment that reads the previous step's output and hands on its own, so it
can sit between "[0:v]" and ",subtitles=..." in the final render. Captions come after the grade so text stays
crisp and readable.
"""
from __future__ import annotations

GRADES = {
    "none": "No grade",
    "hazy": "Hazy: lifted blacks, muted colour, soft glow, grain, vignette",
    "vhs": "VHS: soft resolution, colour bleed, scanlines, grain",
    "film": "Film: warm vintage curve, grain, vignette",
}


def grade_graph(name: str | None, width: int, height: int) -> str:
    """Filter fragment for a grade ('' for none). Labels are prefixed so they never clash with others."""
    if not name or name == "none":
        return ""
    if name == "hazy":
        return ("curves=all='0/0.07 0.25/0.23 0.5/0.48 0.75/0.72 1/0.92',"
                "eq=saturation=0.78:contrast=0.95:gamma=0.97,"
                "colorbalance=rs=0.03:bs=-0.02:rh=0.03:bh=-0.03,"
                "split=2[qz_a][qz_b];[qz_b]gblur=sigma=18,curves=all='0/0 0.55/0.32 1/0.62'[qz_g];"
                "[qz_a][qz_g]blend=all_mode=screen:all_opacity=0.28,"
                "noise=alls=5:allf=t,vignette=angle=PI/5")
    if name == "vhs":
        lw, lh = max(2, width * 2 // 5 // 2 * 2), max(2, height * 2 // 5 // 2 * 2)
        return (f"scale={lw}:{lh}:flags=bilinear,scale={width}:{height}:flags=bilinear,"
                "rgbashift=rh=4:bh=-4,eq=saturation=1.12:contrast=1.04:brightness=0.01,"
                "noise=alls=13:allf=t,"
                f"drawgrid=w={width}:h=4:t=1:c=black@0.16,vignette=angle=PI/6")
    if name == "film":
        return ("curves=preset=vintage,eq=saturation=0.9:contrast=1.03,"
                "noise=alls=10:allf=t,vignette=angle=PI/5")
    raise ValueError(f"unknown grade '{name}'")

"""Caption layout (planner side) and ASS generation (renderer side).

Layout measures text with the same font files libass uses and keeps every box inside the
strict common safe area. The ASS writer only serialises what the plan already decided.
"""
from __future__ import annotations

import re
from functools import lru_cache

from PIL import ImageFont

from .types import TextCard, Word

W, H = 1080, 1920
DEVANAGARI = re.compile("[ऀ-ॿ]")
PAUSE_BREAK = 0.25
MIN_WORD_PROB = 0.5


def script_of(text: str) -> str:
    return "devanagari" if DEVANAGARI.search(text) else "latin"


def apply_case(text: str, case: str) -> str:
    return {"upper": text.upper(), "lower": text.lower()}.get(case, text)


@lru_cache(maxsize=64)
def _pil_font(file: str, index: int, size: int):
    return ImageFont.truetype(file, size, index=index)


def _width(text: str, font: dict, size: int, stroke: int) -> float:
    left, _, right, _ = _pil_font(font["file"], font.get("index", 0), size).getbbox(text)
    synth_bold = 1.04 if font.get("bold") else 1.0
    return (right - left) * synth_bold + 2 * stroke


def _allowed_width(top: float, bottom: float, safe: dict) -> float:
    left = safe["left"] * W
    right = (1 - safe["right"]) * W
    rail = safe.get("right_rail")
    if rail and bottom > rail["y_from"] * H:
        right = min(right, rail["x_from"] * W)
    return 2 * min(W / 2 - left, right - W / 2)


def _wrap(words: list[str], font: dict, size: int, stroke: int, max_w: float, max_lines: int):
    lines: list[list[str]] = [[]]
    for word in words:
        trial = " ".join(lines[-1] + [word])
        if lines[-1] and _width(trial, font, size, stroke) > max_w:
            lines.append([word])
        else:
            lines[-1].append(word)
        if _width(" ".join(lines[-1]), font, size, stroke) > max_w or len(lines) > max_lines:
            return None
    return lines


def layout_text(text: str, block: dict, style: dict, safe: dict):
    """Return (lines, font_spec, size, pos, bbox_norm) or None when it cannot fit."""
    font = style["fonts"][script_of(text)]
    stroke = style["stroke_px"]
    words = text.split()
    y = block["y"] * H
    for size in range(block["size"], block["min_size"] - 1, -2):
        ascent, descent = _pil_font(font["file"], font.get("index", 0), size).getmetrics()
        line_h = ascent + descent
        worst_bottom = y + block["max_lines"] * line_h / 2
        max_w = _allowed_width(y - block["max_lines"] * line_h / 2, worst_bottom, safe)
        lines = _wrap(words, font, size, stroke, max_w, block["max_lines"])
        if not lines:
            continue
        height = len(lines) * line_h
        top, bottom = y - height / 2, y + height / 2
        if top < safe["top"] * H or bottom > (1 - safe["bottom"]) * H:
            continue
        width = max(_width(" ".join(line), font, size, stroke) for line in lines)
        bbox = (round((W - width) / 2 / W, 4), round(top / H, 4),
                round((W + width) / 2 / W, 4), round(bottom / H, 4))
        return [" ".join(line) for line in lines], font, size, (W // 2, int(y)), bbox
    return None


def hook_card(text: str, start: float, end: float, style: dict, safe: dict, kind: str = "hook",
              block: str = "hook") -> TextCard:
    text = apply_case(text.strip(), style.get("case", "natural")) if block == "hook" else text.strip()
    fitted = layout_text(text, style[block], style, safe)
    if fitted is None:
        raise ValueError(f"hook text does not fit the safe area: {text!r}")
    lines, font, size, pos, bbox = fitted
    return TextCard(kind=kind, text=text, start=start, end=end, lines=lines, font=font["family"],
                    font_size=size, pos=pos, bbox_norm=bbox)


def speech_cards(words: list[dict], clip_duration: float, style: dict, safe: dict) -> list[TextCard]:
    """Group clip-time words into cards: break on pauses, punctuation and the word cap."""
    block = style["speech"]
    words = [w for w in words if w.get("probability", 1) >= MIN_WORD_PROB and w["word"].strip()]
    joined: list[dict] = []
    for w in words:               # whisper splits "out-of-the-box" and "160,000" into fragments
        if joined and re.match(r"^(-\w|,\d)", w["word"]):
            joined[-1] = {**joined[-1], "word": joined[-1]["word"] + w["word"], "end": w["end"]}
        else:
            joined.append(w)
    words = joined
    groups: list[list[dict]] = []
    for w in words:
        if (not groups or len(groups[-1]) >= block["words_per_card"]
                or w["start"] - groups[-1][-1]["end"] > PAUSE_BREAK
                or re.search(r"[.!?,;:]$", groups[-1][-1]["word"])):
            groups.append([])
        groups[-1].append(w)

    def clean_text(group: list[dict]) -> str:
        return " ".join(apply_case(re.sub(r"[{}\\]", "", w["word"]), style.get("case", "natural")) for w in group)

    def fit(group: list[dict]) -> list[tuple[list[dict], tuple | None]]:
        """A group that cannot fit the safe area is split in half, never dropped (a lone word that still
        does not fit is the only thing left out)."""
        fitted = layout_text(clean_text(group), block, style, safe)
        if fitted or len(group) == 1:
            return [(group, fitted)]
        mid = len(group) // 2
        return fit(group[:mid]) + fit(group[mid:])

    placed = [item for g in groups for item in fit(g)]
    cards = []
    for i, (group, fitted) in enumerate(placed):
        start = group[0]["start"]
        nxt = placed[i + 1][0][0]["start"] if i + 1 < len(placed) else clip_duration
        end = min(max(group[-1]["end"] + 0.15, start + 0.4), nxt, clip_duration)
        if end - start < 0.1 or fitted is None:
            continue
        clean = clean_text(group).split(" ")
        lines, font, size, pos, bbox = fitted
        cards.append(TextCard(kind="speech", text=" ".join(clean), start=round(start, 3), end=round(end, 3),
                              lines=lines, font=font["family"], font_size=size, pos=pos, bbox_norm=bbox,
                              words=[Word(text=c, start=w["start"], end=w["end"]) for c, w in zip(clean, group)]))
    return cards


# --------------------------------------------------------------------------- ASS output

def _ass_color(hex_rgb: str) -> str:
    rr, gg, bb = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"&H00{bb}{gg}{rr}&".upper()


def _ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def _ass_size(card: TextCard, style: dict) -> int:
    """libass sizes text by line height (ascent + descent); layout measured em size with PIL."""
    font = next((f for f in style["fonts"].values() if f["family"] == card.font), None)
    if font is None:
        return card.font_size
    ascent, descent = _pil_font(font["file"], font.get("index", 0), card.font_size).getmetrics()
    return round(ascent + descent)


def build_ass(cards: list[TextCard], style: dict) -> str:
    base, active = _ass_color(style["colors"]["base"]), _ass_color(style["colors"]["active"])
    stroke = _ass_color(style["colors"]["stroke"])
    latin = style["fonts"]["latin"]["family"]
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\nWrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Q,{latin},64,{base},{base},{stroke},&H80000000&,0,0,0,0,100,100,0,0,1,"
        f"{style['stroke_px']},{style['shadow_px']},5,0,0,0,1\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    events = []
    pop = "\\fscx92\\fscy92\\t(0,120,\\fscx100\\fscy100)"
    for card in cards:
        bold = "\\b1" if card.font != latin else ""
        prefix = f"\\pos({card.pos[0]},{card.pos[1]})\\fn{card.font}\\fs{_ass_size(card, style)}{bold}"
        if card.kind != "speech" or not card.words:
            text = "\\N".join(_escape(line) for line in card.lines)
            events.append(f"Dialogue: 0,{_ass_time(card.start)},{_ass_time(card.end)},Q,,0,0,0,,"
                          f"{{{prefix}{pop}}}{text}")
            continue
        per_line = [len(line.split()) for line in card.lines]
        for i, word in enumerate(card.words):
            w_start = card.start if i == 0 else word.start
            w_end = card.words[i + 1].start if i + 1 < len(card.words) else card.end
            if w_end - w_start < 0.01:
                continue
            tokens, k = [], 0
            for count in per_line:
                line_tokens = []
                for _ in range(count):
                    token = _escape(card.words[k].text)
                    line_tokens.append(f"{{\\c{active}}}{token}{{\\c{base}}}" if k == i else token)
                    k += 1
                tokens.append(" ".join(line_tokens))
            anim = pop if i == 0 else ""
            joined = "\\N".join(tokens)
            events.append(f"Dialogue: 0,{_ass_time(w_start)},{_ass_time(w_end)},Q,,0,0,0,,"
                          f"{{{prefix}{anim}}}{joined}")
    return header + "\n".join(events) + "\n"

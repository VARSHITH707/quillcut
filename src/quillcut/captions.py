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


def clean_word(text: str) -> str:
    return re.sub(r"[^a-z0-9']", "", text.lower())


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


def canvas_of(style: dict) -> tuple[int, int]:
    """Output canvas for layout: a style may carry {"canvas": [w, h]}; default is vertical 1080x1920."""
    c = style.get("canvas")
    return (int(c[0]), int(c[1])) if c else (W, H)


def _allowed_width(top: float, bottom: float, safe: dict, canvas: tuple[int, int] = (W, H)) -> float:
    cw, ch = canvas
    left = safe["left"] * cw
    right = (1 - safe["right"]) * cw
    rail = safe.get("right_rail")
    if rail and bottom > rail["y_from"] * ch:
        right = min(right, rail["x_from"] * cw)
    return 2 * min(cw / 2 - left, right - cw / 2)


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


def _balance(lines: list[list[str]], font: dict, size: int, stroke: int, max_w: float) -> list[list[str]]:
    """Avoid a lone word on the second line ("in the afternoon / i"): move one word down when it still fits."""
    if len(lines) == 2 and len(lines[1]) == 1 and len(lines[0]) >= 3:
        trial = [lines[0][:-1], [lines[0][-1], *lines[1]]]
        if all(_width(" ".join(t), font, size, stroke) <= max_w for t in trial):
            return trial
    return lines


def layout_text(text: str, block: dict, style: dict, safe: dict):
    """Return (lines, font_spec, size, pos, bbox_norm) or None when it cannot fit."""
    script = script_of(text)
    font = block.get("font") if script == "latin" and block.get("font") else style["fonts"][script]
    stroke = style["stroke_px"]
    words = text.split()
    cw, ch = canvas_of(style)
    y = block["y"] * ch
    for size in range(block["size"], block["min_size"] - 1, -2):
        ascent, descent = _pil_font(font["file"], font.get("index", 0), size).getmetrics()
        line_h = ascent + descent
        worst_bottom = y + block["max_lines"] * line_h / 2
        max_w = _allowed_width(y - block["max_lines"] * line_h / 2, worst_bottom, safe, (cw, ch))
        lines = _wrap(words, font, size, stroke, max_w, block["max_lines"])
        if not lines:
            continue
        lines = _balance(lines, font, size, stroke, max_w)
        height = len(lines) * line_h
        top, bottom = y - height / 2, y + height / 2
        if top < safe["top"] * ch or bottom > (1 - safe["bottom"]) * ch:
            continue
        width = max(_width(" ".join(line), font, size, stroke) for line in lines)
        bbox = (round((cw - width) / 2 / cw, 4), round(top / ch, 4),
                round((cw + width) / 2 / cw, 4), round(bottom / ch, 4))
        return [" ".join(line) for line in lines], font, size, (cw // 2, int(y)), bbox
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


def lower_third_card(text: str, start: float, end: float, style: dict, safe: dict, y: float = 0.80,
                     size: int = 46) -> TextCard:
    """Left-aligned label bar. `pos` is the text's left edge and vertical centre."""
    cw, ch = canvas_of(style)
    font = style["fonts"][script_of(text)]
    stroke = 0
    max_w = cw * (1 - safe["left"] - safe["right"]) - 80
    use = size
    while use > 28 and _width(text, font, use, stroke) > max_w:
        use -= 2
    width = _width(text, font, use, stroke)
    if width > max_w:                              # last resort: trim at a word boundary
        words = text.split()
        while len(words) > 1 and _width(" ".join(words) + "…", font, use, stroke) > max_w:
            words.pop()
        text = " ".join(words) + "…"
        width = _width(text, font, use, stroke)
    ascent, descent = _pil_font(font["file"], font.get("index", 0), use).getmetrics()
    height = ascent + descent
    x_left = safe["left"] * cw + 40
    cy = y * ch
    bbox = (round((x_left - 40) / cw, 4), round((cy - height / 2 - 14) / ch, 4),
            round((x_left + width + 28) / cw, 4), round((cy + height / 2 + 14) / ch, 4))
    return TextCard(kind="lower_third", text=text, start=start, end=end, lines=[text], font=font["family"],
                    font_size=use, pos=(int(x_left), int(cy)), bbox_norm=bbox)


def callout_card(text: str, start: float, end: float, style: dict, safe: dict, y: float = 0.56) -> TextCard:
    """Key-phrase pill, centred, placed above the speech captions."""
    block = {"y": y, "size": 60, "min_size": 40, "max_lines": 1}
    fitted = layout_text(text, block, style, safe)
    if fitted is None:
        raise ValueError(f"callout does not fit: {text!r}")
    lines, font, size, pos, bbox = fitted
    return TextCard(kind="callout", text=text, start=start, end=end, lines=lines, font=font["family"],
                    font_size=size, pos=pos, bbox_norm=bbox)


def cta_card(text: str, start: float, end: float, style: dict, safe: dict, y: float = 0.62) -> TextCard:
    block = {"y": y, "size": 72, "min_size": 44, "max_lines": 2}
    fitted = layout_text(text, block, style, safe)
    if fitted is None:
        raise ValueError(f"cta does not fit: {text!r}")
    lines, font, size, pos, bbox = fitted
    return TextCard(kind="cta", text=text, start=start, end=end, lines=lines, font=font["family"],
                    font_size=size, pos=pos, bbox_norm=bbox)


def speech_cards(words: list[dict], clip_duration: float, style: dict, safe: dict,
                 emphasis: set[str] | None = None) -> list[TextCard]:
    """Group clip-time words into cards: break on pauses, punctuation and the word cap."""
    block = style["speech"]
    emphasis = emphasis or set()
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
                              words=[Word(text=c, start=w["start"], end=w["end"],
                                          emphasis=clean_word(c) in emphasis and len(clean_word(c)) > 3)
                                     for c, w in zip(clean, group)]))
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


def _font_spec(family: str, style: dict) -> dict | None:
    """The font entry for a family: style-wide fonts first, then per-block overrides (e.g. a serif title)."""
    specs = list(style["fonts"].values()) + [b["font"] for b in style.values() if isinstance(b, dict) and "font" in b]
    return next((f for f in specs if f.get("family") == family), None)


def _ass_size(card: TextCard, style: dict) -> int:
    """libass sizes text by line height (ascent + descent); layout measured em size with PIL."""
    font = _font_spec(card.font, style)
    if font is None:
        return card.font_size
    ascent, descent = _pil_font(font["file"], font.get("index", 0), card.font_size).getmetrics()
    return round(ascent + descent)


DEFAULT_ACCENT = "F85830"          # the Quillcut orange; a style can override with colors.accent


def _rect(x0: float, y0: float, x1: float, y1: float) -> str:
    return f"m {x0:.0f} {y0:.0f} l {x1:.0f} {y0:.0f} {x1:.0f} {y1:.0f} {x0:.0f} {y1:.0f}"


def _drawing(layer: int, start: float, end: float, x: float, y: float, shape: str, colour: str,
             alpha: str = "00", anim: str = "") -> str:
    return (f"Dialogue: {layer},{_ass_time(start)},{_ass_time(end)},Q,,0,0,0,,"
            f"{{\\an7\\pos({x:.0f},{y:.0f})\\bord0\\shad0\\1c{colour}\\1a&H{alpha}&{anim}\\p1}}{shape}{{\\p0}}")


def build_ass(cards: list[TextCard], style: dict) -> str:
    base, active = _ass_color(style["colors"]["base"]), _ass_color(style["colors"]["active"])
    stroke = _ass_color(style["colors"]["stroke"])
    accent = _ass_color(style["colors"].get("accent", DEFAULT_ACCENT))
    ink = _ass_color("101010")
    latin = style["fonts"]["latin"]["family"]
    cw, ch = canvas_of(style)
    header = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {cw}\nPlayResY: {ch}\nWrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Q,{latin},64,{base},{base},{stroke},&H80000000&,0,0,0,0,100,100,0,0,1,"
        f"{style['stroke_px']},{style['shadow_px']},5,0,0,0,1\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    events = []
    subtle = style.get("motion") == "subtle"           # catalog look: fades, no bounce
    pop = "" if subtle else "\\fscx92\\fscy92\\t(0,120,\\fscx100\\fscy100)"
    title_pop = ("\\fad(320,320)" if subtle else
                 "\\fad(110,180)\\fscx72\\fscy72\\t(0,200,\\fscx106\\fscy106)\\t(200,330,\\fscx100\\fscy100)")
    bar_h = 3 if subtle else 12
    for card in cards:
        spec = _font_spec(card.font, style)
        bold = "\\b1" if (spec or {}).get("bold") else ""
        size = _ass_size(card, style)
        prefix = f"\\pos({card.pos[0]},{card.pos[1]})\\fn{card.font}\\fs{size}{bold}"
        t0, t1 = _ass_time(card.start), _ass_time(card.end)
        x0, y0, x1, y1 = (card.bbox_norm[0] * cw, card.bbox_norm[1] * ch, card.bbox_norm[2] * cw,
                          card.bbox_norm[3] * ch)
        text = "\\N".join(_escape(line) for line in card.lines)

        if card.kind == "hook":
            bar_w = x1 - x0
            events.append(_drawing(0, card.start, card.end, x0, y1 + 12, _rect(0, 0, bar_w, bar_h), accent,
                                   anim="\\fad(320,320)" if subtle else "\\fscx0\\fad(0,180)\\t(120,520,\\fscx100)"))
            events.append(f"Dialogue: 1,{t0},{t1},Q,,0,0,0,,{{{prefix}{title_pop}}}{text}")
            continue
        if card.kind == "lower_third":
            pad, bar = 22, 10
            events.append(_drawing(0, card.start, card.end, x0, y0, _rect(0, 0, x1 - x0, y1 - y0), ink, alpha="55",
                                   anim="\\fad(140,160)"))
            events.append(_drawing(1, card.start, card.end, x0, y0, _rect(0, 0, bar, y1 - y0), accent,
                                   anim="\\fad(140,160)"))
            events.append(f"Dialogue: 2,{t0},{t1},Q,,0,0,0,,{{\\an4\\pos({x0 + bar + pad:.0f},{card.pos[1]})"
                          f"\\fn{card.font}\\fs{size}{bold}\\bord0\\shad0\\fad(140,160)}}{text}")
            continue
        if card.kind == "callout":
            events.append(f"Dialogue: 1,{t0},{t1},Q,,0,0,0,,{{{prefix}\\c{ink}\\3c{accent}\\bord16\\shad0"
                          f"\\fad(100,160)\\fscx60\\fscy60\\t(0,180,\\fscx106\\fscy106)\\t(180,300,\\fscx100\\fscy100)}}{text}")
            continue
        if card.kind == "cta":
            events.append(f"Dialogue: 1,{t0},{t1},Q,,0,0,0,,{{{prefix}\\fad(150,0){pop}}}{text}")
            continue
        if card.kind != "speech" or not card.words:
            events.append(f"Dialogue: 0,{t0},{t1},Q,,0,0,0,,{{{prefix}{pop}}}{text}")
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
                    w = card.words[k]
                    token = _escape(w.text)
                    if k == i:
                        line_tokens.append(f"{{\\c{active}}}{token}{{\\c{base}}}")
                    elif w.emphasis:
                        line_tokens.append(f"{{\\c{accent}\\fscx112\\fscy112}}{token}{{\\c{base}\\fscx100\\fscy100}}")
                    else:
                        line_tokens.append(token)
                    k += 1
                tokens.append(" ".join(line_tokens))
            anim = pop if i == 0 else ""
            joined = "\\N".join(tokens)
            events.append(f"Dialogue: 0,{_ass_time(w_start)},{_ass_time(w_end)},Q,,0,0,0,,"
                          f"{{{prefix}{anim}}}{joined}")
    return header + "\n".join(events) + "\n"

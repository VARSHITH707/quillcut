"""User rules for a run: free text -> constraints Quillcut can enforce, plus an honest list of what it cannot.

Rules are optional. Each line of text is read by a fixed set of patterns. A line that matches becomes either a
hard filter on which moments may be chosen (topics, banned words, profanity, numbers, length) or a setting
(editing intensity, shape, music, closing line, hook style, the interviewer's question). A line that matches
nothing is reported back as "not enforced automatically" and is never silently ignored or silently trusted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_RULE_CHARS = 4000
MIN_RULE_SECONDS, MAX_RULE_SECONDS = 5.0, 180.0

PROFANITY = {"fuck", "fucking", "fucked", "shit", "bullshit", "shitty", "bitch", "asshole", "bastard", "damn",
             "dammit", "goddamn", "crap", "piss", "pissed", "dick", "cunt", "motherfucker", "wtf", "bollocks"}
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                "twenty", "thirty", "forty", "fifty", "hundred", "thousand", "million", "billion", "percent", "dozen"}
FILLER_WORDS = {"the", "a", "an", "any", "some", "my", "our", "your", "their", "this", "that", "these", "those",
                "of", "to", "about", "on", "in", "topic", "topics", "subject", "word", "words", "term", "terms"}
UNIT = r"(seconds?|secs?|s|minutes?|mins?|m)\b"


def _seconds(value: str, unit: str) -> float:
    n = float(value)
    return n * 60 if unit.lower().startswith("m") else n


def _terms(text: str, limit: int = 8) -> list[str]:
    """'pricing, our competitors and Acme Corp' -> ['pricing', 'competitors', 'acme corp']."""
    text = re.sub(r"[\"“”‘’]", "", text)
    text = re.sub(r"[.!?]+$", "", text.strip())
    out: list[str] = []
    for part in re.split(r",|/|;|\band\b|\bor\b|\bplus\b", text, flags=re.I):
        words = [w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", part) if w.lower() not in FILLER_WORDS]
        term = " ".join(words[:4]).lower()
        if term and term not in out:
            out.append(term)
    return out[:limit]


@dataclass
class Rules:
    text: str = ""
    include_any: list[str] = field(default_factory=list)
    include_all: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    profanity: bool = False
    require_number: bool = False
    min_seconds: float | None = None
    max_seconds: float | None = None
    target_seconds: float | None = None
    strict: bool | None = None
    include_question: str | None = None
    intensity: str | None = None
    aspect: str | None = None
    music: str | None = None
    cta: str | None = None
    hook_style: str | None = None          # "question" | "cold_open" | "opening"
    applied: list[dict] = field(default_factory=list)      # {"rule", "meaning", "kind": "filter" | "setting"}
    advisory: list[str] = field(default_factory=list)      # lines that cannot be enforced automatically

    # ---- what the rules constrain -----------------------------------------------------------------------
    @property
    def empty(self) -> bool:
        return not (self.applied or self.advisory)

    @property
    def filters(self) -> bool:
        return bool(self.include_any or self.include_all or self.exclude or self.profanity or self.require_number
                    or self.min_seconds or self.max_seconds)

    def allows(self, text: str, duration: float | None = None) -> bool:
        return all(c["ok"] for c in self.check(text, duration))

    def check(self, text: str, duration: float | None = None) -> list[dict]:
        """One result per enforceable rule: {"rule", "ok", "detail"}. Used both to filter and to report."""
        out: list[dict] = []
        low = text.lower()

        def has(term: str) -> bool:
            return re.search(r"\b" + re.escape(term) + r"\w{0,3}\b", low) is not None

        if self.include_all:
            missing = [t for t in self.include_all if not has(t)]
            out.append({"rule": "must mention " + ", ".join(self.include_all), "ok": not missing,
                        "detail": "mentions all of them" if not missing else "missing: " + ", ".join(missing)})
        if self.include_any:
            found = [t for t in self.include_any if has(t)]
            out.append({"rule": "only about " + " / ".join(self.include_any), "ok": bool(found),
                        "detail": ("mentions " + ", ".join(found)) if found else "does not mention any of them"})
        if self.exclude:
            hit = [t for t in self.exclude if has(t)]
            out.append({"rule": "never mention " + ", ".join(self.exclude), "ok": not hit,
                        "detail": "none mentioned" if not hit else "mentions " + ", ".join(hit)})
        if self.profanity:
            bad = sorted({w for w in re.findall(r"[a-z']+", low) if w in PROFANITY})
            out.append({"rule": "no profanity", "ok": not bad, "detail": "clean" if not bad else "contains " + ", ".join(bad)})
        if self.require_number:
            ok = bool(re.search(r"\d", text)) or any(w in NUMBER_WORDS for w in re.findall(r"[a-z]+", low))
            out.append({"rule": "includes a number", "ok": ok, "detail": "has a number" if ok else "no number said"})
        if duration is not None and self.max_seconds:
            out.append({"rule": f"no longer than {self.max_seconds:g}s", "ok": duration <= self.max_seconds + 0.5,
                        "detail": f"{duration:.0f}s"})
        if duration is not None and self.min_seconds:
            out.append({"rule": f"at least {self.min_seconds:g}s", "ok": duration >= self.min_seconds - 0.5,
                        "detail": f"{duration:.0f}s"})
        return out

    def report(self) -> dict:
        """What the page shows: what was understood and enforced, and what was not."""
        return {"applied": self.applied, "advisory": self.advisory, "filters": self.filters}


# --------------------------------------------------------------------------------------------- parsing

def _statements(text: str) -> list[str]:
    parts = re.split(r"[\r\n]+|;|(?<=[.!?])\s+(?=[A-Z0-9])", text)
    out = []
    for p in parts:
        p = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s*", "", p).strip()
        if p:
            out.append(p)
    return out


def parse(text: str | None) -> Rules:
    """Read free-text rules. Never raises; anything it cannot enforce lands in `advisory`."""
    rules = Rules(text=(text or "").strip()[:MAX_RULE_CHARS])
    for raw in _statements(rules.text):
        s = raw.strip()
        low = s.lower()
        done = False

        def note(meaning: str, kind: str) -> None:
            nonlocal done
            rules.applied.append({"rule": raw, "meaning": meaning, "kind": kind})
            done = True

        # --- language and safety
        if re.search(r"\b(no|without|avoid|never|don'?t|do not)\b.*\b(profanity|swear\w*|curs\w+|bad words|expletives?|"
                     r"explicit|offensive language)\b|\b(family[- ]friendly|keep it clean|clean language)\b", low):
            rules.profanity = True
            note("Moments containing profanity are never chosen.", "filter")

        # --- settings
        if re.search(r"\b(no|without|don'?t use|do not use)\b.{0,12}\bmusic\b", low):
            rules.music = "none"
            note("No background music.", "setting")
        elif re.search(r"\b(with|add|include|use|put)\b.{0,40}\bmusic\b", low):
            rules.music = "subtle"
            note("Subtle background music.", "setting")
        opens_on = re.search(r"\b(?:start|open|begin)\w*\b", low) is not None        # a hook rule, not this one
        if re.search(r"\b(without|skip|exclude|drop|remove|no|don'?t include|do not include|don'?t keep)\b.{0,30}"
                     r"\bquestions?\b|\banswers? only\b", low):
            rules.include_question = "no"
            note("The interviewer's question is not included in the video.", "setting")
        elif not opens_on and re.search(r"\b(include|keep|show|with|add)\b.{0,30}\bquestions?\b", low):
            rules.include_question = "yes"
            note("The interviewer's question is always included.", "setting")
        m = re.search(r"\b(square|1:1)\b|\b(16:9|landscape|widescreen|horizontal)\b|\b(9:16|vertical|portrait)\b", low)
        if m:
            rules.aspect = "1:1" if m.group(1) else "16:9" if m.group(2) else "9:16"
            note(f"Shape set to {rules.aspect}.", "setting")
        m = re.search(r"\b(high[- ]energy|energetic|fast[- ]paced|punchy)\b|\b(dynamic)\b|"
                      r"\b(minimal(?:ist)?|calm|simple edit|clean edit|clean look|low[- ]key)\b", low)
        if m:
            rules.intensity = "high_energy" if m.group(1) else "dynamic" if m.group(2) else "clean"
            note(f"Editing intensity set to {rules.intensity.replace('_', ' ')}.", "setting")
        m = re.search(r"(?:end|close|finish)\s+(?:each\s+\w+\s+)?with\s+[\"“'](.{2,60}?)[\"”']|"
                      r"\b(?:cta|call to action)\s*[:\-]\s*(.{2,60})$", s, flags=re.I)
        if m:
            rules.cta = (m.group(1) or m.group(2)).strip().strip("\"'“”.")
            note(f"Closing line: “{rules.cta}”.", "setting")
        if re.search(r"\b(?:start|open|begin)\b.{0,10}\b(?:with|on)\b.{0,10}\bquestion\b", low):
            rules.hook_style = "question"
            note("Reels open on the question when there is one.", "setting")
        elif re.search(r"\b(?:start|open|begin)\b.{0,10}\b(?:with|on)\b.{0,15}\b(strongest|best|punchiest|boldest)\b", low):
            rules.hook_style = "cold_open"
            note("Reels may open on their strongest line.", "setting")
        elif re.search(r"\b(?:no|without|skip)\b.{0,6}\bcold open\b", low):
            rules.hook_style = "opening"
            note("Reels open on their own first line (no cold open).", "setting")
        if re.search(r"\b(?:must|should|each|every)\b.{0,25}\b(?:number|statistic|stat|figure|data point)\b", low):
            rules.require_number = True
            note("Every reel must include a spoken number.", "filter")
        if re.search(r"\bstrict(?:ly)?\b.{0,15}\b(?:duration|length|time|seconds)\b|\b(?:never|don'?t|do not)\b.{0,10}"
                     r"\b(?:go|run|exceed)\b.{0,8}\b(?:over|past|the target|the limit)\b", low):
            rules.strict = True
            note("Strict duration: a reel never runs past its target.", "setting")

        # --- length
        m = re.search(rf"\b(?:no|not)\s+(?:longer|more|over)\s+than\s+(\d+(?:\.\d+)?)\s*{UNIT}|"
                      rf"\b(?:under|below|max(?:imum)?(?:\s+of)?|at most|up to|within)\s+(\d+(?:\.\d+)?)\s*{UNIT}", low)
        if m:
            val, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            rules.max_seconds = min(max(_seconds(val, unit), MIN_RULE_SECONDS), MAX_RULE_SECONDS)
            note(f"No reel longer than {rules.max_seconds:g} seconds.", "filter")
        m = re.search(rf"\bat least\s+(\d+(?:\.\d+)?)\s*{UNIT}|\bno\s+(?:shorter|less)\s+than\s+(\d+(?:\.\d+)?)\s*{UNIT}|"
                      rf"\bmin(?:imum)?(?:\s+of)?\s+(\d+(?:\.\d+)?)\s*{UNIT}", low)
        if m:
            g = [x for x in m.groups() if x is not None]
            rules.min_seconds = min(max(_seconds(g[0], g[1]), MIN_RULE_SECONDS), MAX_RULE_SECONDS)
            note(f"No reel shorter than {rules.min_seconds:g} seconds.", "filter")
        m = re.search(rf"\b(\d+)\s*[- ]?{UNIT}\s+(?:reels?|clips?|videos?|shorts?)\b|"
                      rf"\b(?:reels?|clips?)\s+(?:of|around|about)\s+(\d+)\s*{UNIT}", low)
        if m:
            g = [x for x in m.groups() if x is not None]
            rules.target_seconds = min(max(_seconds(g[0], g[1]), 15.0), MAX_RULE_SECONDS)
            note(f"Every reel targets {rules.target_seconds:g} seconds.", "setting")

        # --- topics
        m = re.search(r"\b(?:only|just)\b.*?\b(?:about|on|covering|discussing|mentioning|mentions?|discusses|talks? about)"
                      r"\s+(.+)$", s, flags=re.I) or re.search(r"\b(?:focus on|stick to|centered on)\s+(.+)$", s, flags=re.I)
        if m:
            terms = _terms(m.group(1))
            if terms:
                rules.include_any += [t for t in terms if t not in rules.include_any]
                note("Only moments that mention " + " or ".join(f"“{t}”" for t in terms) + ".", "filter")
        m = re.search(r"\bmust\s+(?:mention|include|contain|cover|discuss|talk about|be about)\s+(.+)$", s, flags=re.I)
        if m and not re.search(r"\b(number|statistic|stat|figure|data point|question)\b", m.group(1), flags=re.I):
            terms = _terms(m.group(1))
            if terms:
                rules.include_all += [t for t in terms if t not in rules.include_all]
                note("Moments must mention " + " and ".join(f"“{t}”" for t in terms) + ".", "filter")
        m = re.search(r"\b(?:avoid|exclude|skip)\s+(?!.*\b(?:profanity|music|question)\b)(.+)$|"
                      r"\b(?:don'?t|do not|never)\s+(?:mention|talk about|say|use|include|discuss|show)\w*\s+(.+)$",
                      s, flags=re.I)
        if m and not re.search(r"profan|swear|curs|music|question", s, flags=re.I):
            terms = _terms(m.group(1) or m.group(2) or "")
            terms = [t for t in terms if t not in ("music", "profanity")]
            if terms:
                rules.exclude += [t for t in terms if t not in rules.exclude]
                note("Moments that mention " + " or ".join(f"“{t}”" for t in terms) + " are never chosen.",
                     "filter")

        if not done:
            rules.advisory.append(raw)
    return rules


def apply_to_options(rules: Rules, options: dict) -> list[str]:
    """Rules win over the page's settings. Returns plain-language notes for every setting a rule changed."""
    notes: list[str] = []
    mapping = {"intensity": "intensity", "aspect": "aspect", "music": "music", "cta": "cta",
               "include_question": "include_question", "hook_style": "hook_style", "strict": "strict"}
    for attr, key in mapping.items():
        value = getattr(rules, attr)
        if value is not None and options.get(key) != value:
            notes.append(f"Rule changed {key.replace('_', ' ')}: {options.get(key)} → {value}.")
            options[key] = value
    if rules.target_seconds is not None:
        options["targets"] = [rules.target_seconds for _ in options["targets"]]
        notes.append(f"Rule set every reel's target to {rules.target_seconds:g}s.")
    if rules.max_seconds is not None:
        options["targets"] = [min(t, rules.max_seconds) for t in options["targets"]]
    return notes

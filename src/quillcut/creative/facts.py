"""Everything an evaluator may look at, built once per clip version and shared read-only.

Heavy measurements (audio envelope, frames, faces) are lazy and cached on the instance, so a
category that does not need them never pays for them. Timeline helpers convert between clip time
and source time; repairs and findings both rely on them, so they live in one place.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .. import media
from ..scripts import map_words
from ..types import EditPlan, Segment

TERMINAL = re.compile(r"[.!?…][\"')\]”’]*$")
DANGLING = {"the", "a", "an", "and", "or", "but", "of", "to", "that", "which", "with", "for", "is", "are",
            "was", "were", "because", "so", "in", "on", "at", "by", "as", "if", "than", "then", "from",
            "about", "into", "my", "your", "his", "her", "its", "our", "their", "this", "these", "those"}
LEADING_CONNECTIVE = {"and", "but", "because", "which", "or"}      # "so"/"then" are ordinary openers, not continuations
STOP = DANGLING | {"you", "your", "have", "has", "had", "not", "can", "will", "just", "they", "them", "there",
                   "what", "when", "where", "who", "how", "all", "out", "get", "got", "like", "one", "some",
                   "more", "very", "much", "can't", "don't", "it's", "i'm", "he's", "she's", "says", "say"}
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS |= {"thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
                 "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000"}
HOP_S = 0.05
STOP_GAP = 0.30           # a pause this long is a natural stop even when the transcript has no punctuation
PUNCT_RELIABLE = 0.03     # share of words with terminal punctuation above which sentence ends can be trusted


def normalize(text: str) -> list[str]:
    """Lower-case word tokens; number words become digits so 'eight-foot' matches '8-foot'."""
    text = text.lower().replace("-", " ").replace("’", "'")
    tokens = re.findall(r"[a-z0-9']+", text.replace(",", ""))
    return [NUMBER_WORDS.get(t, t) for t in tokens]


def content_tokens(text: str) -> set[str]:
    return {t.rstrip("s") if len(t) > 3 else t for t in normalize(text) if t not in STOP and len(t) > 2}


@dataclass(frozen=True)
class Span:
    seg: Segment
    start: float            # clip time
    end: float
    part: int | None        # reel part index parsed from the segment id; None for the end card / scripts


def timeline(plan: EditPlan) -> list[Span]:
    t, out = 0.0, []
    for seg in plan.segments:
        m = re.match(r"p(\d+)_", seg.id)
        out.append(Span(seg, round(t, 3), round(t + seg.out_duration, 3), int(m.group(1)) if m else None))
        t += seg.out_duration
    return out


@dataclass
class Sentence:
    words: list[dict]

    @property
    def text(self) -> str:
        return " ".join(w["word"] for w in self.words)

    @property
    def start(self) -> float:
        return self.words[0]["start"]

    @property
    def end(self) -> float:
        return self.words[-1]["end"]


def split_sentences(words: list[dict], gap: float = STOP_GAP) -> list[Sentence]:
    out: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > gap:
            out.append(cur)
            cur = []
        cur.append(w)
        if TERMINAL.search(w["word"]):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [Sentence(s) for s in out]


def stop_times(words: list[dict]) -> list[float]:
    """Where a sentence can end. A well-punctuated transcript is read from its punctuation alone (a
    hesitation is not a sentence end); a sparsely punctuated one falls back to pauses of STOP_GAP."""
    pauses = punctuation_share(words) < PUNCT_RELIABLE
    out = []
    for i, w in enumerate(words):
        nxt = words[i + 1]["start"] if i + 1 < len(words) else None
        if TERMINAL.search(w["word"]) or nxt is None or (pauses and nxt - w["end"] >= STOP_GAP):
            out.append(w["end"])
    return out


def gap_at(words: list[dict], t: float) -> tuple[float, dict | None, dict | None]:
    """(pause, word before, word after) for a cut at source time t. A cut through a word has pause 0."""
    through = next((w for w in words if w["start"] < t - 0.02 and w["end"] > t + 0.02), None)
    if through:
        return 0.0, through, through
    idx = max((i for i, w in enumerate(words) if w["end"] <= t + 0.05), default=None)
    prev = words[idx] if idx is not None else None
    nxt = words[idx + 1] if idx is not None and idx + 1 < len(words) else (words[0] if idx is None and words else None)
    if nxt is None:
        return float("inf"), prev, None
    return round(nxt["start"] - (prev["end"] if prev else t), 3), prev, nxt


def punctuation_share(words: list[dict]) -> float:
    return sum(bool(TERMINAL.search(w["word"])) for w in words) / max(1, len(words))


def has_capitals(words: list[dict]) -> bool:
    """Sentence-cased text, not just proper nouns scattered through an all-lowercase transcript."""
    caps = sum(w["word"][:1].isupper() for w in words if w["word"].lower() != "i") / max(1, len(words))
    return punctuation_share(words) >= PUNCT_RELIABLE and caps >= 0.04


def end_candidates(words: list[dict], src_out: float, floor: float, ceiling: float,
                   cuts: list[float] | None = None, *, trim_max: float = 12.0, extend_max: float = 8.0) -> dict:
    """Nearest complete-sentence ends around a cut at `src_out` (source time).

    `trim` is the last sentence end before the cut, `extend` the first one after it. A boundary that
    also lands within 0.5 s of a shot cut is preferred (a scene-complete ending beats a bare one).
    """
    closes = stop_times(words)
    cuts = cuts or []
    before = [c for c in closes if floor <= c < src_out - 0.3 and src_out - c <= trim_max]
    after = [c for c in closes if src_out - 0.05 <= c <= ceiling and c - src_out <= extend_max]

    def cost(c: float) -> float:
        return abs(c - src_out) - (2.0 if any(abs(c - k) <= 0.5 for k in cuts) else 0.0)

    return {"trim": min(before, key=cost, default=None), "extend": min(after, key=cost, default=None)}


def decode_frames(video: Path, fps: float, w: int, h: int, *, t0: float = 0.0, dur: float | None = None,
                  gray: bool = False) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.3f}"] + (["-t", f"{dur:.3f}"] if dur else []) + [
        "-i", str(video), "-an", "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo",
        "-pix_fmt", "gray" if gray else "rgb24", "-"]
    data = media.run(cmd, binary=True, check=False).stdout
    ch = 1 if gray else 3
    n = len(data) // (w * h * ch)
    arr = np.frombuffer(data[:n * w * h * ch], dtype=np.uint8)
    return arr.reshape(n, h, w) if gray else arr.reshape(n, h, w, ch)


def audio_envelope(video: Path) -> np.ndarray:
    """Short-term RMS in dBFS, one value per HOP_S, from the rendered file's own audio."""
    audio = media.decode_audio(video, 16000)
    hop = int(16000 * HOP_S)
    n = len(audio) // hop
    if n == 0:
        return np.zeros(0)
    x = audio[:n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt((x ** 2).mean(axis=1)) + 1e-9)


@dataclass
class ClipFacts:
    plan: EditPlan
    src_words: dict[str, list[dict]]                        # source-time words, transcript fixes applied
    shots: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    video: Path | None = None                                # rendered file; audio/visual need it
    policy: dict = field(default_factory=dict)               # book `creative:` block merged with reel's
    cfg: dict = field(default_factory=dict)                  # full quillcut config (creative + durations)

    @property
    def cc(self) -> dict:
        return self.cfg["creative"]

    @cached_property
    def spans(self) -> list[Span]:
        return timeline(self.plan)

    @cached_property
    def body_spans(self) -> list[Span]:
        return [s for s in self.spans if s.seg.id != "end"]

    @cached_property
    def body_end(self) -> float:
        return self.body_spans[-1].end

    @cached_property
    def words(self) -> list[dict]:
        ctx = {k: SimpleNamespace(words=v) for k, v in self.src_words.items()}
        return [w for w in map_words(self.plan.segments, ctx) if w["end"] <= self.body_end + 0.05]

    @cached_property
    def sentences(self) -> list[Sentence]:
        return split_sentences(self.words)

    @cached_property
    def transcript(self) -> str:
        return " ".join(w["word"] for w in self.words)

    @cached_property
    def hook_text(self) -> str:
        return next((c.text for c in self.plan.text_cards if c.kind == "hook"), "")

    def span_at(self, t: float) -> Span:
        return next((s for s in self.spans if s.start <= t < s.end), self.spans[-1])

    def clip_to_source(self, t: float) -> tuple[str, float]:
        sp = self.span_at(t)
        return sp.seg.source, round(sp.seg.src_in + (t - sp.start) * sp.seg.speed, 3)

    def source_to_clip(self, key: str, t: float) -> list[float]:
        """Every clip time at which source `key` time `t` is on screen (audio source)."""
        return [round(s.start + (t - s.seg.src_in) / s.seg.speed, 3) for s in self.spans
                if s.seg.source == key and s.seg.src_in - 1e-6 <= t <= s.seg.src_out + 1e-6]

    def visual_changes(self) -> list[float]:
        """Clip times where the picture changes: segment joins, b-roll swaps and source shot cuts."""
        times = {0.0}
        for prev, nxt in zip(self.spans, self.spans[1:]):
            continuous = (prev.seg.source == nxt.seg.source and prev.seg.video_source is None
                          and nxt.seg.video_source is None and abs(prev.seg.src_out - nxt.seg.src_in) < 0.02)
            if not continuous:
                times.add(nxt.start)
        for sp in self.body_spans:
            key = sp.seg.video_source or sp.seg.source
            base = sp.seg.video_in if sp.seg.video_source else sp.seg.src_in
            for a, _ in self.shots.get(key, []):
                if base + 0.05 < a < base + (sp.end - sp.start) * sp.seg.speed - 0.05:
                    times.add(round(sp.start + (a - base) / sp.seg.speed, 3))
        return sorted(t for t in times if t <= self.body_end)

    # ---- measurements from the rendered file (None when there is no file) -------------------
    @cached_property
    def env(self) -> np.ndarray | None:
        return audio_envelope(self.video) if self.video else None

    def env_slice(self, a: float, b: float) -> np.ndarray:
        e = self.env
        return e[int(a / HOP_S):max(int(a / HOP_S) + 1, int(b / HOP_S))] if e is not None else np.zeros(0)

    @cached_property
    def frames(self) -> np.ndarray | None:
        return decode_frames(self.video, 2, 270, 480) if self.video else None

    @cached_property
    def faces(self):
        if self.frames is None or not len(self.frames):
            return None
        from ..models.faces import detect_faces
        return detect_faces(self.frames)

    @cached_property
    def opening_motion(self) -> float | None:
        if not self.video:
            return None
        g = decode_frames(self.video, 10, 108, 192, dur=3.0, gray=True).astype(np.float32)
        return float(np.abs(np.diff(g, axis=0)).mean() / 255) if len(g) > 1 else None

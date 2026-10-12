"""Local model access (Ollama). Fails closed: unreachable or invalid -> ModelUnavailable.

Nothing here invents an answer. Model output is validated against a schema and every claim that
points at the clip (a quote, a span, a headline term) is checked against the real transcript; what
cannot be verified is dropped and counted, never passed on as a finding.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
from pydantic import BaseModel, Field, ValidationError, model_validator

from .facts import ClipFacts, content_tokens, normalize


class ModelUnavailable(RuntimeError):
    pass


class Check(BaseModel):
    """A yes/no PROBLEM question. A 'yes' is only believed when `quote` can be found in the material;
    a 'no' needs no evidence and is the safe default, so the model can miss a problem but not invent one."""
    yes: bool = False
    quote: str = ""
    reason: str = ""


class OffTopic(BaseModel):
    start: float
    end: float
    quote: str
    reason: str = ""


class Claim(BaseModel):
    text: str
    where: str = "hook"
    reason: str = ""


CHECKS = ("context_missing", "ends_mid_thought", "meaning_distorted")


class TextJudgement(BaseModel):
    """Evidence-bearing observations from the text model. It is NOT asked to score or to praise: a 7B model
    answered 'no specific subject' for 'Why BOXABL Builds Homes in a Factory' and rated a false headline above
    the real one, so subjective positives are measured by deterministic proxies instead (see checks.py)."""
    context_missing: Check = Field(default_factory=Check)
    ends_mid_thought: Check = Field(default_factory=Check)
    meaning_distorted: Check = Field(default_factory=Check)
    suggested_question: str = ""
    off_topic: list[OffTopic] = Field(default_factory=list)
    unsupported_claims: list[Claim] = Field(default_factory=list)
    rejected: int = 0                       # model statements dropped because they could not be verified

    @model_validator(mode="before")
    @classmethod
    def _bare_booleans(cls, data):
        if isinstance(data, dict):
            data = {k: ({"yes": v} if k in CHECKS and isinstance(v, bool) else v) for k, v in data.items()}
        return data


class FrameJudgement(BaseModel):
    """What the vision model SEES. It is given no narration: with narration in the prompt it described the
    narration instead of the picture (it called a black video 'a person speaking about cost')."""
    frame: int
    describe: str = ""                       # grounded one-line description; empty answers are not trusted
    caption_covers_face: bool = False        # advisory only: MediaPipe is the objective check for this


class VisionJudgement(BaseModel):
    frames: list[FrameJudgement]
    times: list[float] = Field(default_factory=list)     # clip seconds of each frame, set by judge_frames

    @model_validator(mode="before")
    @classmethod
    def _wrap(cls, data):
        """A small model sometimes answers a batch with one bare frame object or a bare list."""
        if isinstance(data, list):
            return {"frames": data}
        if isinstance(data, dict) and "frames" not in data and "describe" in data:
            return {"frames": [data]}
        return data


class Relation(BaseModel):
    frame: int
    unrelated: bool = False                  # a PROBLEM report: needs a quote of the narration it contradicts
    quote: str = ""
    reason: str = ""


class RelationJudgement(BaseModel):
    frames: list[Relation] = Field(default_factory=list)
    rejected: int = 0


QUESTION_WORDS = {"would", "could", "should", "what", "which", "live", "want", "think", "ever", "try", "love",
                  "like", "own", "buy", "choose", "pick", "prefer", "rather", "favorite", "need", "imagine",
                  "visit", "stay", "thought", "agree"}      # question stems carry no claim about the clip

SYSTEM = ("You are a strict short-form video editor reviewing a finished vertical clip. Judge ONLY from the "
          "material given. Never invent facts, names or numbers. Scores are integers 1-5 (5 = excellent, "
          "3 = mediocre, 1 = broken). Reply with one JSON object and nothing else.")


class Ollama:
    def __init__(self, cfg: dict, client: httpx.Client | None = None, cache: Path | None = None):
        self.cfg, self.cache_path = cfg, cache
        self.http = client or httpx.Client(timeout=cfg.get("timeout_s", 240))
        self._cache = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.exists() else {}

    def status(self) -> dict:
        try:
            tags = self.http.get(self.cfg["url"] + "/api/tags", timeout=5).json()
        except Exception as exc:
            return {"up": False, "models": [], "text": None, "vision": None,
                    "reason": f"ollama not reachable at {self.cfg['url']} ({type(exc).__name__})"}
        names = [m["name"] for m in tags.get("models", [])]

        def find(want: str) -> str | None:
            """An exact tag, or any tag of the family when `want` names no tag (e.g. "llava")."""
            family_ok = ":" not in want
            return next((n for n in names if n == want or (family_ok and n.split(":")[0] == want)), None)

        text = find(self.cfg["text_model"])
        vision = next((m for m in map(find, self.cfg["vision_models"]) if m), None)
        return {"up": True, "models": names, "text": text, "vision": vision,
                "reason": "" if text else f"text model {self.cfg['text_model']} not installed"}

    def chat(self, model: str, user: str, images: list[bytes] | None = None, num_ctx: int = 6144,
             timeout: float | None = None) -> dict:
        key = hashlib.sha1((model + SYSTEM + user + str(num_ctx)).encode() + b"".join(images or []),
                           usedforsecurity=False).hexdigest()  # reply-cache key, not a security control
        if key in self._cache:
            return self._cache[key]
        msg = {"role": "user", "content": user}
        if images:
            msg["images"] = [base64.b64encode(i).decode() for i in images]
        body = {"model": model, "stream": False, "format": "json",
                # unload at once: an 8 GB card cannot hold the text and vision models together, and the
                # second would silently fall back to CPU (minutes per call)
                "keep_alive": 0,
                "options": {"temperature": 0, "seed": 7, "num_ctx": num_ctx},     # deterministic: a retry would repeat it
                "messages": [{"role": "system", "content": SYSTEM}, msg]}
        try:
            r = self.http.post(self.cfg["url"] + "/api/chat", json=body, timeout=timeout or self.cfg["timeout_s"])
            r.raise_for_status()
            data = json.loads(r.json()["message"]["content"])
        except (httpx.HTTPError, KeyError) as exc:
            raise ModelUnavailable(f"{model}: {type(exc).__name__}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ModelUnavailable(f"{model}: reply was not valid JSON ({exc})") from exc
        self._cache[key] = data
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        return data


# ------------------------------------------------------------------ prompts and verification

def transcript_lines(f: ClipFacts) -> str:
    return "\n".join(f"[{s.start:.1f}-{s.end:.1f}s] {s.text}" for s in f.sentences) or "(no speech)"


def narration_between(f: ClipFacts, a: float, b: float) -> str:
    return " ".join(w["word"] for w in f.words if a <= w["start"] < b)


def text_prompt(f: ClipFacts) -> str:
    joins = "; ".join(f"{s.start:.1f}s: {s.seg.source} {s.seg.src_in:.1f}-{s.seg.src_out:.1f}"
                      + (f" (picture from {s.seg.video_source})" if s.seg.video_source else "")
                      for s in f.spans)
    return (
        f'On-screen headline: "{f.hook_text}". Clip length {f.plan.duration:.0f}s. '
        f"Topic: {f.plan.moment.get('topic') or 'n/a'}. Closing question allowed: {f.policy.get('cta', 'off')}.\n"
        f"Timeline of source ranges: {joins}\nNarration with clip times:\n{transcript_lines(f)}\n\n"
        "Look ONLY for problems. Answer each as {\"yes\": true|false, \"quote\": \"\", \"reason\": \"\"}. A yes is "
        "only accepted if \"quote\" holds the exact words, copied from the narration, that show the problem; "
        "otherwise answer false.\n"
        "context_missing: a viewer with no background could not understand something the narration refers to "
        "(quote the unexplained words).\n"
        "ends_mid_thought: the narration stops in the middle of a sentence (quote the last words).\n"
        "meaning_distorted: the cuts reverse or change what the speaker meant (quote the passage).\n"
        "Also return \"suggested_question\" (a short question using only subjects from the narration, or \"\"), "
        "\"off_topic\" (a list of {\"start\": 0.0, \"end\": 0.0, \"quote\": \"exact narration words\", \"reason\": \"\"} "
        "for stretches that do not serve the headline, or []) and \"unsupported_claims\" (a list of "
        "{\"text\": \"exact headline words\", \"where\": \"hook\", \"reason\": \"\"} for headline statements the "
        "narration never supports, or []).\n"
        "Reply with exactly this JSON shape (never a bare true/false): "
        + json.dumps({k: {"yes": False, "quote": "", "reason": ""} for k in CHECKS}
                     | {"suggested_question": "", "off_topic": [], "unsupported_claims": []}))


def quote_ok(quote: str, text: str, min_words: int = 2) -> bool:
    q = " ".join(normalize(quote))
    return len(q.split()) >= min_words and q in " ".join(normalize(text))


def safe_question(q: str, f: ClipFacts, max_words: int) -> str | None:
    """A closing question is allowed only if it is short and every subject in it is already in the clip."""
    q = q.strip()
    if not q.endswith("?") or len(q.split()) > max_words or any(ch.isdigit() for ch in q):
        return None
    known = content_tokens(f.transcript + " " + f.hook_text + " " + (f.plan.moment.get("topic") or "")
                           + " " + " ".join(f.policy.get("brand_terms", [])) + " "
                           + " ".join(f.policy.get("cta_questions", [])))
    return q if not (content_tokens(q) - known - QUESTION_WORDS) else None


EDGE_S = 8.0          # "last sentence" evidence must come from this far in from the end


def judge_text(ollama: Ollama, model: str, f: ClipFacts) -> TextJudgement:
    raw = ollama.chat(model, text_prompt(f))
    try:
        j = TextJudgement.model_validate(raw)
    except ValidationError as exc:
        raise ModelUnavailable(f"{model}: reply did not match the schema ({exc.error_count()} errors)") from exc
    rejected = 0
    tail = narration_between(f, f.body_end - EDGE_S, f.body_end + 1)
    grounded = {}
    for name, text in (("context_missing", f.transcript), ("ends_mid_thought", tail),
                       ("meaning_distorted", f.transcript)):
        c = getattr(j, name)
        if c.yes and not quote_ok(c.quote, text):              # a problem report without evidence is not believed
            rejected += 1
            c = c.model_copy(update={"yes": False, "reason": f"(unverified) {c.reason}".strip()})
        grounded[name] = c
    spans = []
    for s in j.off_topic:
        if (quote_ok(s.quote, f.transcript) and 0 <= s.start < s.end <= f.body_end + 0.1
                and s.end - s.start <= 0.4 * f.body_end):
            spans.append(s)
        else:
            rejected += 1
    support = content_tokens(f.transcript + " " + (f.plan.moment.get("topic") or "") + " "
                             + " ".join(f.policy.get("brand_terms", []) + f.policy.get("facts_allow", [])))
    claims = []
    for c in j.unsupported_claims:
        in_text = " ".join(normalize(c.text)) in " ".join(normalize(f.hook_text + " " + " ".join(
            x.text for x in f.plan.text_cards if x.kind == "label")))
        if in_text and (content_tokens(c.text) - support):
            claims.append(c)
        else:
            rejected += 1
    question = safe_question(j.suggested_question, f, f.cc["cta"]["max_words"]) or ""
    if j.suggested_question and not question:
        rejected += 1
    return j.model_copy(update={**grounded, "off_topic": spans, "unsupported_claims": claims,
                                "suggested_question": question, "rejected": rejected})


def vision_prompt(n: int) -> str:
    return (
        f"These are {n} frames, in order, from a vertical short video. White and yellow burned-in captions and a "
        "headline are normal. For EACH frame describe only what you can actually see, in one plain sentence "
        "(people, place, objects, activity). If a frame is black, blank or unreadable, say so. Also say whether "
        "burned-in caption text overlaps a person's face (caption_covers_face).\n"
        'Return JSON: {"frames":[{"frame":1,"describe":"","caption_covers_face":false}]} with one entry per frame.')


def judge_frames(ollama: Ollama, model: str, f: ClipFacts) -> VisionJudgement:
    import cv2
    from .facts import decode_frames
    if f.video is None:
        raise ModelUnavailable("no rendered file to show the vision model")
    n = f.cc["llm"]["frames"]
    w, h = f.cc["llm"]["frame_size"]
    step = f.plan.duration / n
    frames = decode_frames(f.video, 1 / step, w, h, t0=step / 2)[:n]           # n frames spread over the clip
    if not len(frames):
        raise ModelUnavailable("could not decode frames for the vision model")
    times = [round(step / 2 + i * step, 1) for i in range(len(frames))]
    jpgs = [cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))[1].tobytes() for fr in frames]
    llm_cfg = f.cc["llm"]

    def ask(imgs: list[bytes]) -> VisionJudgement:
        raw = ollama.chat(model, vision_prompt(len(imgs)), imgs, num_ctx=llm_cfg["vision_ctx"],
                          timeout=llm_cfg["vision_timeout_s"])
        return VisionJudgement.model_validate(raw)

    try:
        v = ask(jpgs)
        complete = len(v.frames) == len(jpgs)
    except ValidationError:
        complete = False
    if not complete:                       # the batch answer was malformed or short: ask frame by frame instead
        singles = []
        for k, jpg in enumerate(jpgs, 1):
            try:
                one = ask([jpg])
            except ValidationError as exc:
                raise ModelUnavailable(f"{model}: reply did not match the schema ({exc.error_count()} errors)") from exc
            if len(one.frames) != 1:
                raise ModelUnavailable(f"{model}: asked about one frame, got {len(one.frames)} answers")
            singles.append(one.frames[0].model_copy(update={"frame": k}))
        v = VisionJudgement(frames=singles)
    if len(v.frames) != len(frames) or sum(bool(x.describe.strip()) for x in v.frames) < len(frames) / 2:
        raise ModelUnavailable(f"{model}: reply is not grounded in the frames (descriptions missing or wrong count)")
    return v.model_copy(update={"times": times})


def relation_prompt(f: ClipFacts, v: VisionJudgement) -> str:
    rows = []
    for i, (t, fr) in enumerate(zip(v.times, v.frames), 1):
        near = narration_between(f, t - 2.0, t + 2.0)
        rows.append(f'frame {i} at {t:.1f}s: picture = "{fr.describe}" | narration = "{near or "(none)"}"')
    return (
        "For each frame below, a vision model described the PICTURE and the NARRATION is what is being said "
        "at that moment. Report a problem only when the picture is clearly unrelated to or contradicts the narration "
        "(for example narration about a kitchen over a picture of a car park). A person simply talking, or b-roll "
        "of the product while it is discussed, is NOT a problem. If you report one, \"quote\" must be the exact "
        "narration words.\n" + "\n".join(rows) + "\n"
        'Return JSON: {"frames":[{"frame":1,"unrelated":false,"quote":"","reason":""}]} with one entry per frame.')


def judge_relation(ollama: Ollama, model: str, f: ClipFacts, v: VisionJudgement) -> RelationJudgement:
    """Text-only: compares the vision model's description with the narration, so neither model sees both."""
    raw = ollama.chat(model, relation_prompt(f, v))
    try:
        r = RelationJudgement.model_validate(raw)
    except ValidationError as exc:
        raise ModelUnavailable(f"{model}: relation reply did not match the schema") from exc
    rejected, kept = 0, []
    for x in r.frames:
        i = x.frame - 1
        if not x.unrelated:
            continue
        near = narration_between(f, v.times[i] - 2.0, v.times[i] + 2.0) if 0 <= i < len(v.times) else ""
        if near and quote_ok(x.quote, near):
            kept.append(x)
        else:
            rejected += 1                                # an unquoted complaint is not believed
    return RelationJudgement(frames=kept, rejected=rejected)


def judge_visual(ollama: Ollama, vision_model: str, text_model: str | None, f: ClipFacts):
    """(what the frames show, quoted picture-vs-narration problems or None when no text model is installed)."""
    v = judge_frames(ollama, vision_model, f)
    return v, (judge_relation(ollama, text_model, f, v) if text_model else None)


@dataclass
class Models:
    """What the evaluators may ask. A None judge means that service is not available (and why)."""
    text: object | None = None                  # callable(ClipFacts) -> TextJudgement
    vision: object | None = None                # callable(ClipFacts) -> VisionJudgement
    names: dict = field(default_factory=dict)   # {"text_model": ..., "vision_model": ...}
    reasons: dict = field(default_factory=dict)  # {"text": why not, "vision": why not}

    @classmethod
    def from_config(cls, cfg: dict, cache: Path | None = None) -> "Models":
        o = Ollama(cfg["creative"]["llm"], cache=cache)
        st = o.status()
        m = cls(names={"text_model": st["text"], "vision_model": st["vision"]},
                reasons={"text": st["reason"] or None,
                         "vision": None if st["vision"] else
                         (st["reason"] or "no vision model installed (qwen2.5vl:7b is the planned P1 model)")})
        if st["text"]:
            m.text = lambda f, model=st["text"]: judge_text(o, model, f)
        if st["vision"]:
            m.vision = lambda f, model=st["vision"], text=st["text"]: judge_visual(o, model, text, f)
        return m

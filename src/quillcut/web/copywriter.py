"""Caption and hashtag text for a finished reel.

Uses the local Ollama text model with the reel's own transcript. If Ollama is not running, or its answer
is unusable, a plain template built from the same transcript is returned and labelled `source="template"`
so the page never presents a template as model output.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

import httpx

log = logging.getLogger("quillcut.web")

MAX_CAPTION = 180
MAX_TAGS = 8
BASE_TAGS = ["#shorts", "#reels"]
STOPWORDS = set("""a about after again all also am an and any are as at be because been before being but by can could did do
does doing down for from further had has have he her here him his how i if in into is it its just like me more most my no
nor not now of off on once only or other our out over own same she should so some such than that the their them then there
these they this those through to too under until up us very was we were what when where which while who why will with would
you your yeah okay really gonna got get one two going know right think thing things""".split())

SYSTEM = ("You write social media captions for short vertical videos. Use only facts that appear in the transcript. "
          "Never invent names, numbers or claims. Reply with JSON only.")


def keywords(text: str, limit: int = 5) -> list[str]:
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z']{3,}", text.lower()) if w not in STOPWORDS]
    return [w.replace("'", "") for w, _ in Counter(words).most_common(limit)]


def clean_tags(raw: list, extra: list[str] = ()) -> list[str]:
    """Lowercase, alphanumeric, '#'-prefixed, de-duplicated, capped."""
    tags: list[str] = []
    for item in [*raw, *extra]:
        tag = "#" + re.sub(r"[^0-9a-zA-Z]", "", str(item)).lower()
        if len(tag) > 2 and len(tag) <= 30 and tag not in tags:
            tags.append(tag)
    return tags[:MAX_TAGS]


def template_copy(hook: str | None, topic: str | None, transcript: str) -> dict:
    first = re.split(r"(?<=[.!?])\s+", transcript.strip())[0] if transcript.strip() else ""
    caption = (hook or topic or first or "Watch this").strip()
    if first and first.lower() not in caption.lower() and len(caption) + len(first) < MAX_CAPTION - 3:
        caption = f"{caption}. {first}"
    return {"caption": caption[:MAX_CAPTION].rstrip(), "hashtags": clean_tags(keywords(transcript), BASE_TAGS),
            "source": "template"}


def _ask_ollama(url: str, model: str, hook: str | None, topic: str | None, transcript: str,
                timeout: float) -> dict | None:
    prompt = (f"Video topic: {topic or 'unknown'}\nOn-screen hook: {hook or 'none'}\nTranscript:\n{transcript[:1800]}\n\n"
              f'Write one caption (max {MAX_CAPTION} characters, no hashtags in it) and {MAX_TAGS - 2} to {MAX_TAGS} '
              'relevant hashtags. Return {"caption": "...", "hashtags": ["#one", "#two"]}.')
    body = {"model": model, "stream": False, "format": "json", "keep_alive": 0,
            "options": {"temperature": 0.4, "num_ctx": 4096},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    try:
        reply = httpx.post(f"{url}/api/chat", json=body, timeout=timeout)
        reply.raise_for_status()
        data = json.loads(reply.json()["message"]["content"])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.info("caption model unavailable (%s); using template", exc)
        return None
    caption = str(data.get("caption", "")).strip()
    tags = data.get("hashtags")
    if not caption or not isinstance(tags, list):
        return None
    return {"caption": caption[:MAX_CAPTION].rstrip(), "hashtags": clean_tags(tags, BASE_TAGS[:1]), "source": "ollama"}


def describe(hook: str | None, topic: str | None, transcript: str, *, url: str, model: str,
             timeout: float = 90.0) -> dict:
    """Caption + hashtags for one reel. Never raises."""
    if transcript.strip() or hook or topic:
        for _ in range(2):  # small local models occasionally return unusable JSON; one retry is cheap
            answer = _ask_ollama(url, model, hook, topic, transcript, timeout)
            if answer:
                return answer
    return template_copy(hook, topic, transcript)

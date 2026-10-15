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


# --------------------------------------------------------------------------------------------- social post

CTA_SAVE = "Save this guide for later."
CTA_COMMENT = "Comment your thoughts below."
CTA_SEND = "Send this to a teammate."
BUSINESS_WORDS = {"revenue", "customers", "customer", "clients", "client", "startup", "market", "pricing", "sales",
                  "founder", "founders", "company", "business", "profit", "investors", "team", "hiring", "brand",
                  "growth", "industry", "strategy", "product"}
OPINION_WORDS = {"think", "believe", "opinion", "should", "wrong", "disagree", "honestly", "overrated", "myth",
                 "unpopular", "controversial", "actually", "debate"}
REHOOK_WORDS = 9          # "< 10 words"
BULLET_WORDS = 18         # keeps each bullet to two short lines


def clip_type(content_style: str | None, transcript: str) -> str:
    """educational | opinion | business, from the chosen content style or, failing that, the words used."""
    mapped = {"educational": "educational", "debate": "opinion", "motivational": "opinion",
              "storytelling": "opinion", "business": "business"}
    if content_style in mapped:
        return mapped[content_style]
    words = [w.strip(".,!?;:\"'").lower() for w in transcript.split()]
    biz = sum(w in BUSINESS_WORDS for w in words)
    opinion = sum(w in OPINION_WORDS for w in words)
    if biz >= 3 and biz >= opinion:
        return "business"
    if opinion >= 3:
        return "opinion"
    return "educational"


def cta_for(kind: str) -> str:
    return {"educational": CTA_SAVE, "opinion": CTA_COMMENT, "business": CTA_SEND}[kind]


def _shorten(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    cut = limit
    for k in range(limit, max(5, limit // 2), -1):        # end on a clause boundary when there is one
        if re.search(r"[,;:]$", words[k - 1]):
            cut = k
            break
    return " ".join(words[:cut]).rstrip(",;:") + "…"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 4]


def template_post(hook: str | None, topic: str | None, transcript: str, kind: str) -> dict:
    """Re-hook + bullets built only from the speaker's own sentences."""
    sents = _sentences(transcript)
    scored = sorted(range(len(sents)), key=lambda i: -len(keywords(sents[i], 20)))[:4]
    bullets = [_shorten(sents[i], BULLET_WORDS) for i in sorted(scored)][:4]
    base = (topic or hook or (sents[0] if sents else "Watch this")).strip()
    rehook = _shorten(re.sub(r"\s+", " ", base), REHOOK_WORDS).rstrip(".")
    if len(bullets) < 3 and sents:
        bullets = [_shorten(s, BULLET_WORDS) for s in sents[:3]]
    return {"rehook": rehook, "bullets": bullets[:4], "source": "template", "cta": cta_for(kind)}


def _grounded(bullet: str, transcript: str) -> bool:
    mine = set(keywords(bullet, 30))
    pool = set(keywords(transcript, 400))
    return not mine or len(mine & pool) / len(mine) >= 0.6


def _ask_post(url: str, model: str, hook: str | None, topic: str | None, transcript: str, timeout: float):
    prompt = (f"Transcript of a short video:\n{transcript[:2400]}\n\nTopic: {topic or 'unknown'}\n"
              f"Opening line: {hook or 'none'}\n\n"
              'Write a social media post body as JSON: {"rehook": "core premise in fewer than 10 words", '
              '"bullets": ["3 or 4 scannable bullets, each at most 18 words, each a point the speaker actually '
              'made"], "hashtags": ["#tag", "#tag"]}. Use only what the transcript says.')
    body = {"model": model, "stream": False, "format": "json", "keep_alive": 0,
            "options": {"temperature": 0.3, "num_ctx": 4096},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    try:
        reply = httpx.post(f"{url}/api/chat", json=body, timeout=timeout)
        reply.raise_for_status()
        data = json.loads(reply.json()["message"]["content"])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.info("social post model unavailable (%s); using template", exc)
        return None
    rehook = str(data.get("rehook", "")).strip().rstrip(".")
    bullets = [str(b).strip().lstrip("-•* ") for b in (data.get("bullets") or []) if str(b).strip()]
    tags = data.get("hashtags")
    if not rehook or len(rehook.split()) > REHOOK_WORDS + 1 or not 3 <= len(bullets) <= 4:
        return None
    if any(len(b.split()) > BULLET_WORDS + 4 or not _grounded(b, transcript) for b in bullets):
        return None                                        # a bullet the speaker never said: refuse it
    return {"rehook": rehook, "bullets": bullets, "hashtags": tags if isinstance(tags, list) else [],
            "source": "ollama"}


def social_post(hook: str | None, topic: str | None, transcript: str, *, content_style: str | None, url: str,
                model: str, timeout: float = 90.0) -> dict:
    """Post copy for a reel: re-hook, 3-4 bullets, call to action by clip type, hashtags. Never raises."""
    kind = clip_type(content_style, transcript)
    post = None
    if transcript.strip():
        for _ in range(2):
            post = _ask_post(url, model, hook, topic, transcript, timeout)
            if post:
                break
    if post is None:
        post = template_post(hook, topic, transcript, kind)
    cta = cta_for(kind)
    body = post["rehook"] + "\n\n" + "\n".join(f"• {b}" for b in post["bullets"]) + "\n\n" + cta
    tags = clean_tags(post.get("hashtags") or keywords(transcript), BASE_TAGS)
    return {"caption": body, "hashtags": tags, "source": post["source"], "rehook": post["rehook"],
            "bullets": post["bullets"], "cta": cta, "clip_type": kind}


def describe(hook: str | None, topic: str | None, transcript: str, *, url: str, model: str,
             timeout: float = 90.0) -> dict:
    """Caption + hashtags for one reel. Never raises."""
    if transcript.strip() or hook or topic:
        for _ in range(2):  # small local models occasionally return unusable JSON; one retry is cheap
            answer = _ask_ollama(url, model, hook, topic, transcript, timeout)
            if answer:
                return answer
    return template_copy(hook, topic, transcript)

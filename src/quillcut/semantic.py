"""Optional second opinion from the local language model on a selected moment.

The deterministic engine decides what is *eligible* (complete, self-contained). The model is asked one narrow
question about the finished excerpt and its neighbours: does the thought finish here? A confident "no" vetoes
the candidate. If Ollama is not running this returns None and nothing changes.
"""
from __future__ import annotations

import json
import logging

import httpx

log = logging.getLogger("quillcut.semantic")

SYSTEM = ("You judge whether a transcript excerpt is a complete, self-contained piece of conversation. "
          "Use only the text given. Reply with JSON only.")


def review(excerpt: str, before: str, after: str, *, url: str, model: str, timeout: float = 90.0) -> dict | None:
    prompt = (f"Text just BEFORE the excerpt:\n{before[-400:] or '(start of video)'}\n\n"
              f"EXCERPT:\n{excerpt[:2600]}\n\nText just AFTER the excerpt:\n{after[:400] or '(end of video)'}\n\n"
              'Answer as JSON: {"complete": true or false, "standalone": true or false, '
              '"topic": "3 to 6 words", "reason": "one short sentence"}. '
              '"complete" is false only if the speaker is clearly still making the same point after the excerpt '
              'ends. "standalone" is false only if the excerpt cannot be understood without the text before it.')
    body = {"model": model, "stream": False, "format": "json", "keep_alive": 0,
            "options": {"temperature": 0.1, "num_ctx": 4096},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    try:
        reply = httpx.post(f"{url}/api/chat", json=body, timeout=timeout)
        reply.raise_for_status()
        data = json.loads(reply.json()["message"]["content"])
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        log.info("semantic review unavailable (%s)", exc)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("complete"), bool):
        return None
    return {"complete": data["complete"], "standalone": bool(data.get("standalone", True)),
            "topic": str(data.get("topic", ""))[:60], "reason": str(data.get("reason", ""))[:160]}

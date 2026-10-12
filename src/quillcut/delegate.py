"""Delegated formats: OpenShorts makes the clips, Quillcut files them like its own."""
from __future__ import annotations

import time
from pathlib import Path

import httpx

from .paths import CLIPPING

API = "http://127.0.0.1:8000"
OPENSHORTS_OUTPUT = CLIPPING / "repos" / "openshorts" / "output"


class DelegateError(RuntimeError):
    pass


def submit(video: Path, cfg: dict) -> str:
    d = cfg["delegate"]
    data = {"output_format": "9:16", "acknowledged": "true", "target_clips": str(d.get("target_clips", 8)),
            "clip_min_seconds": str(cfg["duration"]["hard_floor_seconds"]),
            "clip_max_seconds": str(cfg["duration"]["max_seconds"])}
    if d.get("layouts"):
        data["layouts"] = d["layouts"]
    with video.open("rb") as fh:
        r = httpx.post(f"{API}/api/process", files={"file": (video.name, fh, "video/mp4")}, data=data,
                       timeout=httpx.Timeout(600))
    if r.status_code != 200:
        raise DelegateError(f"OpenShorts rejected {video.name}: {r.status_code} {r.text[:300]}")
    return r.json()["job_id"]


def wait(job_id: str, timeout: float = 3600, poll: float = 15) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = httpx.get(f"{API}/api/status/{job_id}", timeout=30).json()
        if status.get("status") in ("completed", "failed"):
            return status
        time.sleep(poll)
    raise DelegateError(f"OpenShorts job {job_id} timed out")


def clips(job_id: str) -> list[Path]:
    return sorted((OPENSHORTS_OUTPUT / job_id).glob("subtitled_*_clip_*.mp4"))

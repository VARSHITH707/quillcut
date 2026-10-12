"""Project folders, atomic writes, stage status, rights resolution, retention."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import date, datetime
from pathlib import Path

from .config import load_yaml
from .paths import FORMATS

# user_accepted: the user chose to publish without a licence record (flagged in metadata)
RIGHTS_STATUSES = {"own_footage", "client_owned", "client_licensed", "cc_verified", "user_accepted", "unknown"}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, data) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, default=str))


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def sha1_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha1(usedforsecurity=False)  # content fingerprint for caching, not a security control
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sanitize(stem: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_-")
    return (clean or "video")[:40]


def resolve_rights(video: Path, flag: str | None) -> dict:
    """Flag, then sidecar `<video>.yaml`, then RIGHTS.yaml in the input folder, else unknown."""
    if flag:
        if flag not in RIGHTS_STATUSES:
            raise ValueError(f"rights must be one of {sorted(RIGHTS_STATUSES)}")
        return {"status": flag, "resolved_by": "flag"}
    sidecar = load_yaml(video.with_name(video.name + ".yaml"))
    if sidecar.get("rights") in RIGHTS_STATUSES:
        return {"status": sidecar["rights"], "resolved_by": "sidecar"}
    folder = load_yaml(video.parent / "RIGHTS.yaml")
    if folder.get("status") in RIGHTS_STATUSES:
        return {"status": folder["status"], "resolved_by": "RIGHTS.yaml",
                "client": folder.get("client"), "agreement": folder.get("agreement")}
    return {"status": "unknown", "resolved_by": "none"}


class Project:
    def __init__(self, path: Path):
        self.path = path
        self.json_path = path / "project.json"
        self.data = read_json(self.json_path, {})

    @property
    def id(self) -> str:
        return self.path.name

    @classmethod
    def for_video(cls, video: Path, format_key: str, resolved_by: str, rights: dict) -> "Project":
        video = video.resolve()
        sha = sha1_file(video)
        projects = FORMATS / format_key / "projects"
        existing = [p for p in projects.glob(f"*__{sha[:8]}") if p.is_dir()] if projects.exists() else []
        path = existing[0] if existing else projects / f"{sanitize(video.stem)}__{date.today():%Y%m%d}__{sha[:8]}"
        project = cls(path)
        if not project.data:
            project.data = {"created": datetime.now().isoformat(timespec="seconds"), "stages": {}, "clips": {}}
        project.data.update({
            "source_path": str(video), "source_sha1": sha, "source_size": video.stat().st_size,
            "format_key": format_key, "format_resolved_by": resolved_by, "rights": rights,
        })
        project.save()
        return project

    def save(self) -> None:
        atomic_write_json(self.json_path, self.data)

    def sub(self, name: str) -> Path:
        path = self.path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_complete(self, name: str, input_hash: str) -> bool:
        stage = self.data.get("stages", {}).get(name, {})
        return stage.get("status") == "complete" and stage.get("input_hash") == input_hash

    def mark_stage(self, name: str, status: str, input_hash: str = "", **extra) -> None:
        self.data.setdefault("stages", {})[name] = {
            "status": status, "input_hash": input_hash,
            "at": datetime.now().isoformat(timespec="seconds"), **extra}
        self.save()

    def mark_clip(self, clip_id: str, **fields) -> None:
        self.data.setdefault("clips", {}).setdefault(clip_id, {}).update(fields)
        self.save()


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path.anchor or path).free / 1e9

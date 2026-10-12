"""Format registry and routing. Formats are discovered from folders; nothing is hard-coded."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .config import load_yaml
from .paths import FORMATS


def list_formats() -> list[str]:
    return sorted(p.name for p in FORMATS.iterdir() if p.is_dir() and not p.name.startswith("_")
                  and (p / "config.yaml").exists())


def resolve_format(video: Path, flag: str | None) -> tuple[str | None, str]:
    """flag → input folder → sidecar. Returns (None, 'unresolved') instead of guessing."""
    if flag:
        if flag not in list_formats():
            raise KeyError(f"unknown format '{flag}'; known: {list_formats()}")
        return flag, "flag"
    video = video.resolve()
    for key in list_formats():
        if video.parent == (FORMATS / key / "input").resolve():
            return key, "input_folder"
    sidecar = load_yaml(video.with_name(video.name + ".yaml")).get("format")
    if sidecar in list_formats():
        return sidecar, "sidecar"
    return None, "unresolved"


def load_pipeline(key: str):
    path = FORMATS / key / "code" / "pipeline.py"
    spec = importlib.util.spec_from_file_location(f"quillcut_format_{key}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PIPELINE


def load_style(key: str, name: str) -> dict:
    style = load_yaml(FORMATS / key / "styles" / f"{name}.yaml")
    if not style:
        raise KeyError(f"style '{name}' not found for format '{key}'")
    return style

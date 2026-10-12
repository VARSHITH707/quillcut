from copy import deepcopy
from pathlib import Path

import yaml

from .paths import CONFIG, FORMATS


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(format_key: str) -> dict:
    """config/default.yaml merged with formats/<key>/config.yaml; the format wins."""
    fmt_path = FORMATS / format_key / "config.yaml"
    if not fmt_path.exists():
        raise KeyError(f"unknown format '{format_key}' (no {fmt_path})")
    return deep_merge(load_yaml(CONFIG / "default.yaml"), load_yaml(fmt_path))

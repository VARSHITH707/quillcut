import os
from pathlib import Path

ROOT = Path(os.environ.get("QUILLCUT_ROOT") or Path(__file__).resolve().parents[2])
# Data lives in ./data by default; set QUILLCUT_HOME to keep it on another drive.
CLIPPING = Path(os.environ.get("QUILLCUT_HOME") or ROOT / "data")
CACHE = CLIPPING / "cache"
OUTPUTS = CLIPPING / "outputs"
FORMATS = ROOT / "formats"
CONFIG = ROOT / "config"


def child_env() -> dict:
    """Environment for every subprocess: caches stay under the data folder, UTF-8 I/O."""
    env = dict(os.environ)
    env.update({
        "HF_HOME": str(CACHE / "hf"),
        "TORCH_HOME": str(CACHE / "torch"),
        "PIP_CACHE_DIR": str(CACHE / "pip"),
        "TMP": str(CACHE / "tmp"),
        "TEMP": str(CACHE / "tmp"),
        "PYTHONUTF8": "1",
    })
    return env


def apply_cache_env() -> None:
    (CACHE / "tmp").mkdir(parents=True, exist_ok=True)
    for key in ("HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR", "TMP", "TEMP", "PYTHONUTF8"):
        os.environ[key] = child_env()[key]

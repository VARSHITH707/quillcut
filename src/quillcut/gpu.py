"""Cross-process GPU lock shared with the OpenShorts runner. Windows has no flock."""
from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path

from .paths import CACHE

LOCK_PATH = CACHE / "gpu.lock"
_STILL_ACTIVE = 259
_QUERY_LIMITED = 0x1000


def pid_alive(pid: int) -> bool:
    # os.kill(pid, 0) would terminate the process on Windows, so ask the kernel instead.
    if pid <= 0:
        return False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(_QUERY_LIMITED, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


class GpuLock:
    def __init__(self, path: Path = LOCK_PATH, poll: float = 5.0, timeout: float | None = None):
        self.path, self.poll, self.timeout = path, poll, timeout

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    holder = int(self.path.read_text(encoding="utf-8").strip() or 0)
                except (OSError, ValueError):
                    holder = 0
                if holder == os.getpid():
                    return self
                if not pid_alive(holder):
                    self.path.unlink(missing_ok=True)
                    continue
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError(f"GPU lock held by PID {holder}")
                time.sleep(self.poll)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False

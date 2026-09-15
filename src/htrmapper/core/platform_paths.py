"""Cross-platform path translation for the Windows <-> WSL2 workflow.

Fase 4 (nuvem densa) requires a CUDA-enabled `pycolmap`, which does not
exist as a pip wheel for Windows (`pycolmap-cuda12` is Linux-only) -- see
ARCHITECTURE.md's Fase 4/9 findings. The practical path for a Windows user
with a real NVIDIA GPU is: run Fase 1-3 (import/align/adjust) in the
Windows GUI as normal, then run Fase 4-6 (`htrmapper dense`/`dem`/`ortho`)
from a CLI inside WSL2, which WSL2's own filesystem interop exposes the
Windows disk to at `/mnt/<drive>/...`.

That workflow only works end-to-end if the *same* project file can be
read from both sides. `Project.save()` on Windows writes native Windows
paths (`C:\\Users\\...`) into `ImageRecord.path` and every phase summary's
path fields; loading that file unmodified from inside WSL2 would look for
files at a path that does not exist there. This module normalizes any
stored path to whatever the *current* runtime actually is, at load time,
so a project file created on one side keeps working on the other -- the
user never has to hand-edit the JSON or re-import.

This is pure, deterministic string translation, fully unit-testable
without a real WSL2 environment (which this development sandbox does not
have) -- what could not be tested here, and must be validated by the user
on real hardware, is that WSL2's `/mnt/<drive>` mount itself behaves as
documented; that part is not this module's concern.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

_WINDOWS_PATH_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_WSL_MOUNT_RE = re.compile(r"^/mnt/([A-Za-z])/(.*)$")


@lru_cache(maxsize=1)
def is_running_under_wsl() -> bool:
    """True when the current Python process is running inside WSL
    (WSL1 or WSL2) rather than native Linux or native Windows.

    Checked two ways, either is sufficient: the `WSL_DISTRO_NAME`
    environment variable WSL itself sets, and the "microsoft" marker WSL's
    kernel puts in `/proc/version` -- the same two signals commonly used
    to detect WSL, since neither requires calling out to an external tool.
    """
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def _windows_to_wsl(path_str: str) -> str:
    match = _WINDOWS_PATH_RE.match(path_str)
    if not match:
        return path_str
    drive, rest = match.groups()
    return f"/mnt/{drive.lower()}/{rest.replace(chr(92), '/')}"


def _wsl_to_windows(path_str: str) -> str:
    match = _WSL_MOUNT_RE.match(path_str)
    if not match:
        return path_str
    drive, rest = match.groups()
    return f"{drive.upper()}:\\{rest.replace('/', chr(92))}"


def normalize_path_for_current_platform(path_str: str) -> str:
    """Translate `path_str` to the convention the *current* process needs,
    if (and only if) it looks like it was written by the other side of
    the Windows/WSL2 pair. Anything else (a relative path, a path already
    in the current platform's own convention, an empty string) passes
    through unchanged -- this never guesses at a path it doesn't
    recognize as belonging to the other platform.
    """
    if not path_str:
        return path_str
    if is_running_under_wsl():
        return _windows_to_wsl(path_str)
    if os.name == "nt":
        return _wsl_to_windows(path_str)
    return path_str

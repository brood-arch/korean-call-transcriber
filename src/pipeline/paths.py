"""Path compatibility helpers and WSL/Windows conversion utilities.

New code should prefer ``src.config`` for runtime paths and ``src.config.WORKSPACE``
for workspace resolution.  This module retains WSL/Windows path conversion helpers
and re-exports legacy path constants so that existing ``from src.pipeline.paths
import …`` statements keep working without changes.
"""

from __future__ import annotations

import os
from pathlib import Path

# Re-export canonical path constants from src.config so that existing imports
# continue to resolve correctly.
from src.config import (  # noqa: F401 – re-exports
    AUDIO_DIR,
    LOG_DIR,
    OBSIDIAN_VAULT,
    STATE_DIR,
    TRANSCRIPT_DIR,
    WORKSPACE,
)


def is_wsl() -> bool:
    """Return True if running inside Windows Subsystem for Linux."""
    try:
        return os.name != "nt" and "microsoft" in Path("/proc/version").read_text(errors="ignore").lower()
    except (OSError, UnicodeDecodeError):  # noqa: BLE001
        return False


def win_to_wsl(path: str) -> str:
    """Convert a Windows drive path to a /mnt/<drive>/ path on WSL."""
    if len(path) >= 3 and path[1:3] == ":\\":
        drive = path[0].lower()
        rest = path[3:].replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return path


def wsl_to_win(path: str) -> str:
    """Convert /mnt/<drive>/ paths to Windows drive paths for subprocesses."""
    p = str(path)
    if p.startswith("/mnt/") and len(p) >= 7 and p[6] == "/":
        drive = p[5].upper()
        rest = p[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return p


def native_path(path: str | Path) -> Path:
    """Return *path* converted to the native platform convention."""
    s = str(path)
    if is_wsl():
        s = win_to_wsl(s)
    return Path(s)


def windows_path(path: str | Path) -> str:
    """Return *path* as a Windows-style string (useful inside WSL)."""
    s = str(path)
    if is_wsl():
        return wsl_to_win(s)
    return s

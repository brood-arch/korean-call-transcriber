"""Central path configuration for the transcription pipeline.

Loads paths.json from the workspace and returns OS-native Path values.
Works on Windows and WSL. Environment variables override JSON values:
  KCT_WORKSPACE, TRANSCRIPT_DIR, AUDIO_DIR, OBSIDIAN_VAULT,
  CHROMA_INDEX_DIR, WINDOWS_PYTHON, WHISPERX_PYTHON, HF_TOKEN_FILE

This module is a compatibility shim. New code should use paths.py directly:
    from paths import Paths
    p = Paths()
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def is_wsl() -> bool:
    try:
        return os.name != "nt" and "microsoft" in Path("/proc/version").read_text(errors="ignore").lower()
    except Exception:
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
    s = str(path)
    if is_wsl():
        s = win_to_wsl(s)
    return Path(s)


def windows_path(path: str | Path) -> str:
    s = str(path)
    if is_wsl():
        return wsl_to_win(s)
    return s


def _discover_workspace() -> Path:
    env = os.environ.get("KCT_WORKSPACE")
    if env:
        return native_path(env)

    here = Path(__file__).resolve().parent.parent
    if (here / "config.json").exists() or (here / "paths.json").exists():
        return here

    candidates = [r"."]
    for c in candidates:
        p = native_path(c)
        if (p / "config.json").exists() or (p / "paths.json").exists():
            return p
    return here


WORKSPACE = _discover_workspace()
PATHS_JSON = WORKSPACE / "paths.json"


def _load_paths() -> dict:
    if PATHS_JSON.exists():
        try:
            return json.loads(PATHS_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# ── Resolve current-platform and Windows paths from paths.json ────────────
_CFG = _load_paths()
_PLATFORM = "wsl" if is_wsl() else ("windows" if os.name == "nt" else "linux")
_PLATFORM_CFG = _CFG.get(_PLATFORM, {})
_WIN_CFG = _CFG.get("windows", {})


def get_path(key: str, default: str | Path | None = None, *, env: str | None = None) -> Path:
    raw = os.environ.get(env or key.upper()) or _PLATFORM_CFG.get(key) or default
    if raw is None:
        raise KeyError(f"missing path config: {key}")
    return native_path(raw)


def get_windows_path(key: str, default: str | Path | None = None, *, env: str | None = None) -> str:
    raw = os.environ.get(env or key.upper()) or _WIN_CFG.get(key) or default
    if raw is None:
        raise KeyError(f"missing path config: {key}")
    return windows_path(raw)


# ── Exported constants (backward-compatible with prior pipeline_paths API) ─
TRANSCRIPT_DIR = get_path("transcript_dir", os.environ.get("KCT_TRANSCRIPT_DIR", "output/transcripts"), env="TRANSCRIPT_DIR")
AUDIO_DIR = get_path("audio_dir", os.environ.get("KCT_AUDIO_DIR", "data/audio"), env="AUDIO_DIR")
OBSIDIAN_VAULT = get_path("obsidian_vault", "output/obsidian", env="OBSIDIAN_VAULT")
CHROMA_INDEX_DIR = get_path("chroma_index_dir", TRANSCRIPT_DIR / "chroma_index", env="CHROMA_INDEX_DIR")
STATE_DIR = get_path("state_dir", os.environ.get("KCT_STATE_DIR", WORKSPACE / "state"), env="STATE_DIR")
LOG_DIR = get_path("log_dir", os.environ.get("KCT_LOG_DIR", WORKSPACE / "logs"), env="LOG_DIR")
WINDOWS_PYTHON = get_windows_path("python", "python", env="WINDOWS_PYTHON")
WHISPERX_PYTHON = get_windows_path("whisperx_python", r".\tools\whisperx-venv\Scripts\python.exe", env="WHISPERX_PYTHON")
HF_TOKEN_FILE = get_windows_path("hf_token_file", "", env="HF_TOKEN_FILE")



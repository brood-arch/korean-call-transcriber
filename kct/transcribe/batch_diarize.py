#!/usr/bin/env python3
"""Batch diarize: find .segments.json files without _result.json and run align+diarize.

Designed to run from WSL, invoking whisperx-venv Python on Windows.
Uses file-based JSON input (avoids CLI escaping issues on WSL→Windows).

Usage:
    python batch_diarize.py [--transcript-dir DIR] [--limit N] [--skip-existing]
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from kct.config import (
    HF_TOKEN_FILE,
)
from kct.config import (
    KCT_ALIGN_WORKER as WIN_ALIGN_WORKER,
)
from kct.config import (
    TRANSCRIPT_DIR as WIN_TRANSCRIPT_DIR,
)
from kct.config import (
    WHISPERX_PYTHON as WIN_PYTHON,
)
from kct.pipeline.paths import is_wsl
from kct.pipeline.paths import wsl_to_win as wsl_path_to_windows

log = logging.getLogger(__name__)

CMD_EXE = "/mnt/c/Windows/System32/cmd.exe"


def find_missing(transcript_dir: Path, limit: int = 0) -> list:
    """Find .segments.json files without corresponding _result.json."""
    missing = []
    for segs in sorted(transcript_dir.glob("*.segments.json")):
        # Skip if _result.json already exists
        result_path = transcript_dir / segs.name.replace(".segments.json", ".segments_result.json")
        if result_path.exists():
            continue
        # Check that corresponding audio exists
        stem = segs.stem.replace(".segments", "")
        # Look for audio with matching stem
        for ext in (".m4a", ".wav", ".mp3"):
            audio_path = transcript_dir.parent / "통화녹음" / (stem + ext)
            if audio_path.exists():
                missing.append((stem, segs, audio_path))
                break
    if limit > 0:
        missing = missing[:limit]
    return missing


def find_retry_failed(transcript_dir: Path, limit: int = 0) -> list:
    """Find diarization failures with .segments.json that have a failed _result.json.

    Looks for _result.json files whose metadata indicates diarize failure,
    or .segments.json files where the corresponding _result.json has failed metadata.
    """
    retry_candidates = []
    for result_file in sorted(transcript_dir.glob("*.segments_result.json")):
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
            meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
            if not meta.get("diarize_ok", True):
                stem = result_file.stem.replace(".segments", "")
                segs_path = transcript_dir / (stem + ".segments.json")
                if not segs_path.exists():
                    continue
                for ext in (".m4a", ".wav", ".mp3"):
                    audio_path = transcript_dir.parent / "통화녹음" / (stem + ext)
                    if audio_path.exists():
                        retry_candidates.append((stem, segs_path, audio_path))
                        break
        except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
            log.debug("Failed to inspect result %s: %s", result_file, exc)
    if limit > 0:
        retry_candidates = retry_candidates[:limit]
    return retry_candidates


def _build_diarize_argv(stem, segs_path, audio_path, running_on_wsl):
    """Build the argv for diarization subprocess."""
    if running_on_wsl:
        win_audio = wsl_path_to_windows(audio_path)
        win_segs = wsl_path_to_windows(segs_path)
        win_token = HF_TOKEN_FILE
        wsl_python = "~/.openclaw/workspace/tools/whisperx-venv/Scripts/python.exe"
        win_script = str(WIN_ALIGN_WORKER)
        return [wsl_python, "-u", win_script, "--audio", win_audio, "--segments", win_segs, "--token", win_token]
    win_audio = str(audio_path)
    win_segs = str(segs_path)
    return [WIN_PYTHON, "-u", WIN_ALIGN_WORKER, "--audio", win_audio, "--segments", win_segs, "--token", HF_TOKEN_FILE]


def _run_diarize_one(stem, segs_path, audio_path, transcript_dir, running_on_wsl):
    """Run diarization for one file. Returns True on success."""
    argv = _build_diarize_argv(stem, segs_path, audio_path, running_on_wsl)
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace",
        )
        result_path = transcript_dir / (stem + ".segments_result.json")
        if result_path.exists():
            size = result_path.stat().st_size
            meta_ok = True
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                meta = payload.get("_meta", {})
                if not meta.get("align_ok", False) or not meta.get("diarize_ok", False):
                    meta_ok = False
            except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
                log.warning("could not inspect result metadata: %s", exc)
            if not meta_ok or result.returncode not in (0, 3221226505):
                log.warning("%s bytes, exit=%s, meta_ok=%s", size, result.returncode, meta_ok)
                return False
            log.info("  OK (%s bytes)", size)
            return True
        else:
            log.error("FAIL (exit=%s)", result.returncode)
            if result.stderr:
                log.info("  stderr: %s", result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        log.info("  TIMEOUT")
        return False
    except Exception as e:  # noqa: BLE001
        log.error("%s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch diarize missing segments")
    parser.add_argument("--transcript-dir", default=None,
                        help="Transcript directory (auto-detected if omitted)")
    parser.add_argument("--audio-dir", default=None,
                        help="Audio directory (auto-detected if omitted)")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0=all)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have _result.json")
    parser.add_argument("--days", type=int, default=0,
                        help="Only process files from last N days (0=all)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only retry files with failed diarization results")
    args = parser.parse_args()

    running_on_wsl = is_wsl()
    if running_on_wsl:
        transcript_dir = Path(args.transcript_dir) if args.transcript_dir else Path("data/전사본")
    else:
        transcript_dir = Path(args.transcript_dir or WIN_TRANSCRIPT_DIR)
    if not transcript_dir.exists():
        log.error("transcript dir not found: %s", transcript_dir)
        sys.exit(1)

    if args.retry_failed:
        missing = find_retry_failed(transcript_dir, args.limit)
    else:
        missing = find_missing(transcript_dir, args.limit)
    if args.days > 0:
        from datetime import datetime, timedelta, timezone
        KST = timezone(timedelta(hours=9))
        cutoff = datetime.now(KST) - timedelta(days=args.days)
        missing = [
            (stem, segs, audio) for stem, segs, audio in missing
            if datetime.fromtimestamp(segs.stat().st_mtime, tz=KST) >= cutoff
        ]
    if not missing:
        log.info("No missing diarization results found.")
        sys.exit(0)
    log.info("Found %d files needing diarization.", len(missing))

    ok = 0
    fail = 0
    for i, (stem, segs_path, audio_path) in enumerate(missing):
        log.info("\n[%d/%d] %s", i + 1, len(missing), stem)
        if _run_diarize_one(stem, segs_path, audio_path, transcript_dir, running_on_wsl):
            ok += 1
        else:
            fail += 1

    log.info("DONE %d/%d diarized, %d failed", ok, len(missing), fail)
    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()



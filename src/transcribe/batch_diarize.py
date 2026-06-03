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

from src.config import (
    HF_TOKEN_FILE,
)
from src.config import (
    KCT_ALIGN_WORKER as WIN_ALIGN_WORKER,
)
from src.config import (
    TRANSCRIPT_DIR as WIN_TRANSCRIPT_DIR,
)
from src.config import (
    WHISPERX_PYTHON as WIN_PYTHON,
)
from src.pipeline.paths import is_wsl
from src.pipeline.paths import wsl_to_win as wsl_path_to_windows

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
    args = parser.parse_args()

    # Detect OS and set paths
    running_on_wsl = is_wsl()

    if running_on_wsl:
        if args.transcript_dir:
            transcript_dir = Path(args.transcript_dir)
        else:
            transcript_dir = Path("data/전사본")
    else:
        transcript_dir = Path(args.transcript_dir or WIN_TRANSCRIPT_DIR)

    if not transcript_dir.exists():
        log.error(f"transcript dir not found: {transcript_dir}")
        sys.exit(1)

    # Find missing segments
    missing = find_missing(transcript_dir, args.limit)

    # Filter by days if requested
    if args.days > 0:
        from datetime import datetime, timedelta, timezone
        KST = timezone(timedelta(hours=9))
        cutoff = datetime.now(KST) - timedelta(days=args.days)
        filtered = []
        for stem, segs, audio in missing:
            try:
                mt = datetime.fromtimestamp(segs.stat().st_mtime, tz=KST)
                if mt >= cutoff:
                    filtered.append((stem, segs, audio))
            except Exception as exc:
                log.warning(f"failed to stat segments file: {exc}")
        missing = filtered

    if not missing:
        print("No missing diarization results found.")
        sys.exit(0)

    print(f"Found {len(missing)} files needing diarization.")

    ok = 0
    fail = 0
    for i, (stem, segs_path, audio_path) in enumerate(missing):
        print(f"\n[{i+1}/{len(missing)}] {stem}")

        if running_on_wsl:
            # Convert WSL paths to Windows paths for the worker
            win_audio = wsl_path_to_windows(audio_path)
            win_segs = wsl_path_to_windows(segs_path)
            win_token = HF_TOKEN_FILE
            python_exe = WIN_PYTHON
            script = WIN_ALIGN_WORKER
        else:
            win_audio = str(audio_path)
            win_segs = str(segs_path)
            win_token = HF_TOKEN_FILE
            python_exe = WIN_PYTHON
            script = WIN_ALIGN_WORKER

        if running_on_wsl:
            # Run Windows Python directly from WSL without cmd.exe.
            # subprocess(list) preserves spaces in data safely.
            wsl_python = "~/.openclaw/workspace/tools/whisperx-venv/Scripts/python.exe"
            win_script = str(WIN_ALIGN_WORKER)
            argv = [wsl_python, "-u", win_script, "--audio", win_audio, "--segments", win_segs, "--token", win_token]
        else:
            argv = [python_exe, "-u", script, "--audio", win_audio, "--segments", win_segs, "--token", win_token]

        try:
            result = subprocess.run(
                argv,
                capture_output=True, text=True, timeout=600,
                encoding="utf-8", errors="replace"
            )
            # Check output file existence (Windows subprocess exit codes are unreliable)
            result_path = transcript_dir / (stem + ".segments_result.json")
            if result_path.exists():
                size = result_path.stat().st_size
                meta_ok = True
                try:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    meta = payload.get("_meta", {})
                    if not meta.get("align_ok", False) or not meta.get("diarize_ok", False):
                        meta_ok = False
                except Exception as exc:
                    log.warning("could not inspect result metadata: %s", exc)
                if not meta_ok or result.returncode not in (0, 3221226505):
                    log.warning("%s bytes, exit=%s, meta_ok=%s", size, result.returncode, meta_ok)
                    fail += 1
                else:
                    print(f"  OK ({size} bytes)")
                    ok += 1
            else:
                log.error(f"FAIL (exit={result.returncode})")
                if result.stderr:
                    print(f"  stderr: {result.stderr[-500:]}")
                fail += 1
        except subprocess.TimeoutExpired:
            print("  TIMEOUT")
            fail += 1
        except Exception as e:
            log.error(f"{e}")
            fail += 1

    log.info(f"DONE {ok}/{len(missing)} diarized, {fail} failed")
    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()



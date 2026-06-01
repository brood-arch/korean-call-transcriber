#!/usr/bin/env python3
"""Batch diarize: find .segments.json files without _result.json and run align+diarize.

Designed to run from WSL, invoking whisperx-venv Python on Windows.
Uses file-based JSON input (avoids CLI escaping issues on WSL→Windows).

Usage:
    python batch_diarize.py [--transcript-dir DIR] [--limit N] [--skip-existing]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
WIN_PYTHON = r".\tools\whisperx-venv\Scripts\python.exe"
WIN_ALIGN_WORKER = os.environ.get("KCT_ALIGN_WORKER", r"src\transcribe\align_worker.py")
WIN_AUDIO_DIR = os.environ.get("KCT_AUDIO_DIR", "data/audio")
WIN_TRANSCRIPT_DIR = os.environ.get("KCT_TRANSCRIPT_DIR", "output/transcripts")
HF_TOKEN_FILE = r".\memory\api-keys\pyannote_hf_token.txt"
CMD_EXE = "/mnt/c/Windows/System32/cmd.exe"


def wsl_path_to_windows(path: Path) -> str:
    """Convert /mnt/<drive>/... paths to Windows drive paths."""
    value = str(path)
    if value.startswith("/mnt/") and len(value) > 7 and value[6] == "/":
        drive = value[5].upper()
        return f"{drive}:\\" + value[7:].replace("/", "\\")
    return value.replace("/", "\\")


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
    is_wsl = os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower()

    if is_wsl:
        if args.transcript_dir:
            transcript_dir = Path(args.transcript_dir)
        else:
            transcript_dir = Path("data/전사본")
        if args.audio_dir:
            audio_dir = Path(args.audio_dir)
        else:
            audio_dir = Path("data/통화녹음")
    else:
        transcript_dir = Path(args.transcript_dir or WIN_TRANSCRIPT_DIR)
        audio_dir = Path(args.audio_dir or WIN_AUDIO_DIR)

    if not transcript_dir.exists():
        print(f"ERROR: transcript dir not found: {transcript_dir}")
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
            except Exception:
                pass
        missing = filtered

    if not missing:
        print("No missing diarization results found.")
        sys.exit(0)

    print(f"Found {len(missing)} files needing diarization.")

    ok = 0
    fail = 0
    for i, (stem, segs_path, audio_path) in enumerate(missing):
        print(f"\n[{i+1}/{len(missing)}] {stem}")

        if is_wsl:
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

        if is_wsl:
            # Run Windows Python directly from WSL without cmd.exe.
            # subprocess(list) preserves spaces in data safely.
            wsl_python = "~/.openclaw/workspace/tools/whisperx-venv/Scripts/python.exe"
            win_script = os.environ.get("KCT_ALIGN_WORKER", r"src\transcribe\align_worker.py")
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
                except Exception:
                    pass
                if not meta_ok or result.returncode not in (0, 3221226505):
                    print(f"  WARN ({size} bytes, exit={result.returncode}, meta_ok={meta_ok})")
                    fail += 1
                else:
                    print(f"  OK ({size} bytes)")
                    ok += 1
            else:
                print(f"  FAIL (exit={result.returncode})")
                if result.stderr:
                    print(f"  stderr: {result.stderr[-500:]}")
                fail += 1
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
            fail += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            fail += 1

    print(f"\nDONE {ok}/{len(missing)} diarized, {fail} failed")
    sys.exit(1 if fail > 0 else 0)


if __name__ == "__main__":
    main()



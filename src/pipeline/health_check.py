#!/usr/bin/env python3
"""Pipeline health check: detect missed transcriptions and failed notifications."""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
log = logging.getLogger(__name__)

try:
    from src.pipeline.paths import AUDIO_DIR, STATE_DIR, TRANSCRIPT_DIR
except Exception as exc:
    log.debug("Falling back to local health-check paths: %s", exc)
    TRANSCRIPT_DIR = Path(os.environ.get("TRANSCRIPT_DIR", "./data/transcripts"))
    AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", "./data/audio"))
    STATE_DIR = Path(os.environ.get("KCT_STATE_DIR", "./state"))

STATE_FILE = STATE_DIR / "call_recordings_automation_state.json"
BLACKLIST_FILE = STATE_DIR / "transcribe_blacklist.json"
PERSISTENT_TODOS = STATE_DIR / "persistent_todos.json"
EXTRACT_PROCESSED = STATE_DIR / "integrated_extraction" / "processed_files.json"

def canonical_transcript_stem(stem: str) -> str:
    """Return the source audio stem for generated transcript variants.

    Normal transcripts are ``{audio_stem}.txt``. Manual/diagnostic diarization
    rechecks may create ``{audio_stem}_{HHMMSS}.txt``; those are derivative
    artifacts and should not be counted as new unprocessed call transcripts.
    """
    import re

    m = re.match(r"^(.+_\d{14})_\d{6}$", stem)
    return m.group(1) if m else stem


def main():
    issues = []
    now = datetime.now(KST)

    # Load state
    state = json.loads(STATE_FILE.read_text("utf-8")) if STATE_FILE.exists() else {}
    pt = state.get("processed_transcripts", {})
    # Also load extract_all's processed index — a transcript is considered
    # processed if it appears in *either* index.
    if EXTRACT_PROCESSED.exists():
        try:
            ex = json.loads(EXTRACT_PROCESSED.read_text("utf-8"))
            if isinstance(ex, dict):
                for k, v in ex.items():
                    pt.setdefault(k, v)
        except Exception as exc:
            log.debug("Failed to load extraction processed index %s: %s", EXTRACT_PROCESSED, exc)
    blacklist = set()
    if BLACKLIST_FILE.exists():
        bl = json.loads(BLACKLIST_FILE.read_text("utf-8"))
        blacklist = {k for k in bl if k != "_meta"}

    # Check 1: Recent audio files without transcripts (30min–3h window)
    # Files younger than 30min are considered "pending" (pipeline hasn't had
    # a chance to process them yet) and are excluded from health checks.
    cutoff_upper = now - timedelta(minutes=30)
    cutoff_lower = now - timedelta(hours=3)
    missing_transcripts = []
    pending_transcripts = []
    for f in sorted(AUDIO_DIR.glob("*.m4a")):
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=KST)
        if mtime < cutoff_lower:
            continue
        if f.stem in blacklist:
            continue
        txt = TRANSCRIPT_DIR / (f.stem + ".txt")
        if not txt.exists():
            if mtime >= cutoff_upper:
                pending_transcripts.append(f.name)
            else:
                missing_transcripts.append(f.name)

    if missing_transcripts:
        issues.append(f"MISSING_TRANSCRIPTS: {len(missing_transcripts)} audio files without transcript")
        for name in missing_transcripts:
            print(f"  🔴 {name}")
    if pending_transcripts:
        # Informational only — not a health issue
        for name in pending_transcripts:
            print(f"  ⏳ {name} (pending, <30min old)")

    # Check 2: Transcripts not processed for TODO extraction (last 3 hours)
    # Files marked as "skipped_fast_score" or "fallback" are considered processed.
    unprocessed = []
    processed_stems = set()
    # Load batch results to find skipped/fast_score files
    extract_dir = STATE_DIR / "integrated_extraction"
    if extract_dir.exists():
        for bf in extract_dir.glob("batch_*.json"):
            try:
                bd = json.loads(bf.read_text("utf-8"))
                for r in bd.get("results", []):
                    if isinstance(r, dict):
                        status = r.get("status", "")
                        if status in ("ok", "fallback", "skipped_fast_score"):
                            stem = r.get("file", "")
                            if stem:
                                processed_stems.add(stem)
            except Exception as exc:
                log.debug("Failed to inspect extraction batch %s: %s", bf, exc)
    for f in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=KST)
        if mtime < cutoff_lower:
            continue
        if len(f.read_text("utf-8").strip()) < 20:
            continue
        canonical_stem = canonical_transcript_stem(f.stem)
        if canonical_stem != f.stem and (AUDIO_DIR / f"{canonical_stem}.m4a").exists():
            continue
        if canonical_stem in pt:
            continue
        if canonical_stem in processed_stems:
            continue
        unprocessed.append(f.name)

    if unprocessed:
        issues.append(f"UNPROCESSED_TODOS: {len(unprocessed)} transcripts not processed for TODOs")
        for name in unprocessed:
            print(f"  ⚠️ {name}")

    # Check 3: Persistent todos with no status field (indicates interrupted processing)
    if PERSISTENT_TODOS.exists():
        pdata = json.loads(PERSISTENT_TODOS.read_text("utf-8"))
        todos = pdata.get("todos", {})
        no_status = [k for k, v in todos.items() if isinstance(v, dict) and "status" not in v]
        if no_status:
            issues.append(f"ORPHAN_TODOS: {len(no_status)} todos without status field")

    # Summary
    if issues:
        print(f"PIPELINE ISSUES ({len(issues)}):")
        for issue in issues:
            print(f"  🔴 {issue}")
        return 1
    else:
        print("PIPELINE HEALTHY: All recent files processed")
        return 0

if __name__ == "__main__":
    sys.exit(main())

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
    from kct.config import AUDIO_DIR, STATE_DIR, TRANSCRIPT_DIR
except ImportError as exc:  # noqa: BLE001
    log.debug("Falling back to local health-check paths: %s", exc)
    TRANSCRIPT_DIR = Path(os.environ.get("TRANSCRIPT_DIR", "./data/transcripts"))
    AUDIO_DIR = Path(os.environ.get("AUDIO_DIR", "./data/audio"))
    STATE_DIR = Path(os.environ.get("KCT_STATE_DIR", "./state"))

STATE_FILE = STATE_DIR / "call_recordings_automation_state.json"
BLACKLIST_FILE = STATE_DIR / "transcribe_blacklist.json"
PERSISTENT_TODOS = STATE_DIR / "persistent_todos.json"
EXTRACT_PROCESSED = STATE_DIR / "integrated_extraction" / "processed_files.json"


def canonical_transcript_stem(stem: str) -> str:
    """전사 파일명 stem에서 타임스탬프 접미사를 제거해 원본 오디오 stem을 반환."""
    import re
    m = re.match(r"^(.+_\d{14})_\d{6}$", stem)
    return m.group(1) if m else stem


def _load_and_merge_processed():
    """Load both state and extraction processed indexes."""
    state = json.loads(STATE_FILE.read_text("utf-8")) if STATE_FILE.exists() else {}
    pt = state.get("processed_transcripts", {})
    if EXTRACT_PROCESSED.exists():
        try:
            ex = json.loads(EXTRACT_PROCESSED.read_text("utf-8"))
            if isinstance(ex, dict):
                for k, v in ex.items():
                    pt.setdefault(k, v)
        except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
            log.debug(
                "Failed to load extraction processed index %s: %s",
                EXTRACT_PROCESSED, exc,
            )
    return pt


def _load_blacklist():
    """Load blacklist as a set of stems."""
    if not BLACKLIST_FILE.exists():
        return set()
    bl = json.loads(BLACKLIST_FILE.read_text("utf-8"))
    return {k for k in bl if k != "_meta"}


def _check_missing_transcripts(now, blacklist):
    """Check for recent audio without transcripts."""
    issues = []
    cutoff_upper = now - timedelta(minutes=30)
    cutoff_lower = now - timedelta(hours=3)
    missing = []
    pending = []
    for f in sorted(AUDIO_DIR.glob("*.m4a")):
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=KST)
        if mtime < cutoff_lower or f.stem in blacklist:
            continue
        txt = TRANSCRIPT_DIR / (f.stem + ".txt")
        if not txt.exists():
            if mtime >= cutoff_upper:
                pending.append(f.name)
            else:
                missing.append(f.name)
    if missing:
        issues.append(
            f"MISSING_TRANSCRIPTS: {len(missing)} audio files without transcript"
        )
        for name in missing:
            log.info("  🔴 %s", name)
    for name in pending:
        log.info("  ⏳ %s (pending, <30min old)", name)
    return issues


def _load_batch_processed_stems():
    """Load processed stems from batch results."""
    processed_stems = set()
    extract_dir = STATE_DIR / "integrated_extraction"
    if not extract_dir.exists():
        return processed_stems
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
        except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
            log.debug("Failed to inspect extraction batch %s: %s", bf, exc)
    return processed_stems


def _check_unprocessed_todos(now, pt, processed_stems):
    """Check for transcripts not processed for TODO extraction."""
    issues = []
    cutoff_lower = now - timedelta(hours=3)
    unprocessed = []
    for f in sorted(TRANSCRIPT_DIR.glob("*.txt")):
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=KST)
        if mtime < cutoff_lower:
            continue
        if len(f.read_text("utf-8").strip()) < 20:
            continue
        canonical_stem = canonical_transcript_stem(f.stem)
        if canonical_stem != f.stem and (AUDIO_DIR / f"{canonical_stem}.m4a").exists():
            continue
        if canonical_stem in pt or canonical_stem in processed_stems:
            continue
        unprocessed.append(f.name)
    if unprocessed:
        issues.append(
            f"UNPROCESSED_TODOS: {len(unprocessed)} transcripts not processed for TODOs"
        )
        for name in unprocessed:
            log.info("  ⚠️ %s", name)
    return issues


def main() -> int:
    """파이프라인 건강 상태를 점검한다.

    Return codes:
        0 — healthy or minor issues (logged but non-fatal)
        2 — critical issues that require immediate attention

    Issues are always included in the JSON result printed to stdout
    regardless of the exit code, so callers can inspect without
    relying on rc alone.
    """
    now = datetime.now(KST)
    pt = _load_and_merge_processed()
    blacklist = _load_blacklist()

    issues = []
    issues.extend(_check_missing_transcripts(now, blacklist))
    processed_stems = _load_batch_processed_stems()
    issues.extend(_check_unprocessed_todos(now, pt, processed_stems))

    if PERSISTENT_TODOS.exists():
        pdata = json.loads(PERSISTENT_TODOS.read_text("utf-8"))
        todos = pdata.get("todos", {})
        no_status = [
            k for k, v in todos.items()
            if isinstance(v, dict) and "status" not in v
        ]
        if no_status:
            issues.append(f"ORPHAN_TODOS: {len(no_status)} todos without status field")

    # Determine severity: critical issues get rc=2, everything else rc=0
    critical_keywords = ("MISSING_TRANSCRIPTS",)
    has_critical = any(any(iss.startswith(kw) for kw in critical_keywords) for iss in issues)

    result = {"issues": issues, "issue_count": len(issues), "critical": has_critical}
    print(json.dumps(result, ensure_ascii=False))

    if issues:
        log.info("PIPELINE ISSUES (%s):", len(issues))
        for issue in issues:
            log.info("  🔴 %s", issue)
    else:
        log.info("PIPELINE HEALTHY: All recent files processed")

    return 2 if has_critical else 0


if __name__ == "__main__":
    sys.exit(main())

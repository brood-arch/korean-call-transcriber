#!/usr/bin/env python3
"""
State Validator — Check state files for existence, staleness, and integrity.

Monitors expected state files for:
- Existence (file present on disk)
- Staleness (modified longer ago than threshold)
- Integrity (valid JSON, no null-byte corruption)

Environment variables:
    KCT_STATE_DIR — Base state directory (default: state)

Usage:
    python -m kct.pipeline.validate_state [--fix] [--json] [--quiet]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_STATE_DIR = Path(os.environ.get("KCT_STATE_DIR", "state"))

# Files to monitor (relative to state dir)
EXPECTED_FILES = [
    "pipeline_state.json",
    "persistent_todos.json",
    "events.jsonl",
    "correction_rules.json",
]

# Maximum age in hours before flagged as stale
STALE_THRESHOLDS = {
    "pipeline_state.json": 72,
    "persistent_todos.json": 48,
    "events.jsonl": 168,  # 1 week
    "correction_rules.json": 168,
}


def check_file(name: str, state_dir: Path | None = None, fix: bool = False) -> dict:
    """Check a single state file for existence, staleness, and integrity.

    Args:
        name: Filename relative to state directory.
        state_dir: Override state directory path.
        fix: If True, attempt auto-fix (currently only flags).

    Returns:
        Dict with file status details.
    """
    sdir = state_dir or _STATE_DIR
    fpath = sdir / name
    threshold = STALE_THRESHOLDS.get(name, 48)

    if not fpath.exists():
        return {"file": name, "exists": False, "stale": True, "age_hours": -1}

    try:
        mtime = datetime.fromtimestamp(fpath.stat().st_mtime, tz=timezone.utc)
    except OSError as e:
        return {"file": name, "exists": True, "stale": True, "error": str(e), "age_hours": -1}

    age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
    stale = age_hours > threshold

    # Optional: validate JSON files
    parse_ok = True
    if name.endswith(".json") and fpath.stat().st_size > 0:
        try:
            raw = fpath.read_bytes()
            if b"\x00" in raw:
                parse_ok = False
            else:
                json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            parse_ok = False

    return {
        "file": name,
        "exists": True,
        "stale": stale,
        "age_hours": round(age_hours, 1),
        "parse_ok": parse_ok,
        "threshold_hours": threshold,
    }


def check_all(state_dir: Path | None = None, fix: bool = False) -> dict:
    """Check all expected state files.

    Args:
        state_dir: Override state directory path.
        fix: Attempt auto-fix where possible.

    Returns:
        Validation report dict.
    """
    results = [check_file(name, state_dir, fix) for name in EXPECTED_FILES]
    all_ok = all(
        r["exists"] and not r["stale"] and r.get("parse_ok", True) for r in results
    )
    return {
        "ok": all_ok,
        "files": results,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    """CLI entry point for state validation."""
    parser = argparse.ArgumentParser(description="Validate workspace state files")
    parser.add_argument("--fix", action="store_true", help="Attempt auto-fix where possible")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print if issues found")
    args = parser.parse_args()

    report = check_all(fix=args.fix)

    if args.json:
        log.info(json.dumps(report, indent=2, ensure_ascii=False))
        sys.exit(0 if report["ok"] else 1)

    issues = []
    for f in report["files"]:
        if not f["exists"]:
            issues.append(f"  ❌ MISSING: {f['file']}")
        elif not f.get("parse_ok", True):
            issues.append(f"  ⚠️  CORRUPT: {f['file']} (parse error or null bytes)")
        elif f["stale"]:
            issues.append(
                f"  ⏰ STALE: {f['file']} ({f['age_hours']}h ago, "
                f"threshold {f.get('threshold_hours', 48)}h)"
            )

    if args.quiet and not issues:
        sys.exit(0)

    log.info(f"State validation — {'✅ OK' if report['ok'] else '❌ ISSUES FOUND'}")
    log.info(f"Checked at: {report['checked_at']}")
    if issues:
        for line in issues:
            log.info(line)
    else:
        log.info("  All files present, fresh, and parseable.")

    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()

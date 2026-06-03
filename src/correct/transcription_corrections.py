#!/usr/bin/env python3
"""Persistent correction layer for call transcription text.

Supports two correction types:
1. exact replacements: confident literal substitutions for recurring STT mistakes
2. aliases: normalize known company/person/product/model variants to one canonical term
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.pipeline.utils import safe_read_json, safe_write_json

log = logging.getLogger(__name__)

try:
    from src.pipeline.paths import STATE_DIR, WORKSPACE
except Exception as exc:
    log.debug("Falling back to local correction state paths: %s", exc)
    WORKSPACE = Path.cwd()
    STATE_DIR = Path("state")

RULES_PATH = STATE_DIR / "transcription_correction_rules.json"
LOG_PATH = STATE_DIR / "transcription_correction_events.jsonl"
KST = timezone(timedelta(hours=9))

DEFAULT_RULES: dict[str, Any] = {
    "exact_replacements": [],
    "aliases": [],
}


def _now_iso() -> str:
    return datetime.now(KST).isoformat()


def ensure_rules_file() -> dict[str, Any]:
    if not RULES_PATH.exists():
        safe_write_json(RULES_PATH, DEFAULT_RULES, origin='transcription_corrections')
    return load_rules()


# --- Hot-reload support: track rules file mtime ---
_rules_mtime: float = 0.0
_rules_cache: dict = {}


def load_rules() -> dict[str, Any]:
    """Load rules with hot-reload support (non-recursive)."""
    global _rules_mtime, _rules_cache

    # Fast path: cache is valid and file hasn't changed
    if _rules_cache and RULES_PATH.exists():
        try:
            current_mtime = RULES_PATH.stat().st_mtime
            if current_mtime <= _rules_mtime:
                return _rules_cache
        except Exception as exc:
            log.debug("Failed to stat correction rules %s: %s", RULES_PATH, exc)

    # Reload from disk
    if not RULES_PATH.exists():
        safe_write_json(RULES_PATH, DEFAULT_RULES, origin="transcription_corrections")

    try:
        data = safe_read_json(RULES_PATH, default=DEFAULT_RULES.copy())
    except Exception as exc:
        log.debug("Failed to load correction rules %s: %s", RULES_PATH, exc)
        data = DEFAULT_RULES.copy()

    if not isinstance(data, dict):
        data = DEFAULT_RULES.copy()
    data.setdefault("exact_replacements", [])
    data.setdefault("aliases", [])

    _rules_cache = data
    try:
        _rules_mtime = RULES_PATH.stat().st_mtime
    except Exception as exc:
        log.debug("Failed to stat correction rules after load %s: %s", RULES_PATH, exc)
        _rules_mtime = 0.0

    return data


def save_rules(data: dict[str, Any]) -> None:
    """Atomic write via safe_io."""
    safe_write_json(RULES_PATH, data, origin='transcription_corrections')


def _term_boundary_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    return re.compile(rf"(?<![0-9A-Za-z가-힣]){escaped}(?![0-9A-Za-z가-힣])")


def apply_corrections(text: str, source: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    rules = load_rules()
    updated = text
    changes: list[dict[str, Any]] = []

    for rule in rules.get("exact_replacements", []):
        if not rule.get("enabled", True):
            continue
        old = (rule.get("from") or "").strip()
        new = (rule.get("to") or "").strip()
        if not old or old == new:
            continue
        count = updated.count(old)
        if count <= 0:
            continue
        updated = updated.replace(old, new)
        changes.append({
            "type": "exact",
            "from": old,
            "to": new,
            "count": count,
            "note": rule.get("note"),
        })

    for rule in rules.get("aliases", []):
        if not rule.get("enabled", True):
            continue
        canonical = (rule.get("canonical") or "").strip()
        variants = [str(v).strip() for v in rule.get("variants", []) if str(v).strip()]
        if not canonical:
            continue
        unique_variants = []
        seen = set()
        for variant in variants:
            if variant == canonical or variant in seen:
                continue
            seen.add(variant)
            unique_variants.append(variant)
        for variant in sorted(unique_variants, key=len, reverse=True):
            pattern = _term_boundary_pattern(variant)
            updated, count = pattern.subn(canonical, updated)
            if count > 0:
                changes.append({
                    "type": "alias",
                    "from": variant,
                    "to": canonical,
                    "count": count,
                    "category": rule.get("category"),
                })

    if changes:
        log_event({
            "at": _now_iso(),
            "source": source,
            "change_count": len(changes),
            "changes": changes,
        })
    return updated, changes


def log_event(event: dict[str, Any]) -> None:
    """Append-only write with retry (P2-8: handles concurrent writes).
    P2-9: Auto-rotate log when it exceeds 5MB.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    # P2-9: Rotate if log file is too large (>5MB)
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > 5 * 1024 * 1024:
            lines = LOG_PATH.read_text(encoding="utf-8").splitlines()
            keep_lines = lines[-5000:]  # ~5000 events
            tmp = LOG_PATH.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(keep_lines) + "\n", encoding="utf-8")
            tmp.replace(LOG_PATH)
    except Exception:
        pass  # Rotation failure should not block logging

    line = json.dumps(event, ensure_ascii=False) + "\n"
    for attempt in range(3):
        try:
            with LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line)
            return
        except (OSError, IOError):
            if attempt < 2:
                time.sleep(0.1 * (attempt + 1))
            else:
                # Last resort: ignore — logging should not break the pipeline
                pass


def add_exact_replacement(old: str, new: str, note: str | None = None) -> dict[str, Any]:
    rules = ensure_rules_file()
    old = old.strip()
    new = new.strip()
    for rule in rules["exact_replacements"]:
        if (rule.get("from") or "").strip() == old:
            rule.update({"to": new, "note": note, "enabled": True, "updated_at": _now_iso()})
            save_rules(rules)
            return rule
    rule = {"from": old, "to": new, "note": note, "enabled": True, "updated_at": _now_iso()}
    rules["exact_replacements"].append(rule)
    save_rules(rules)
    return rule


def add_alias(canonical: str, variants: list[str], category: str | None = None) -> dict[str, Any]:
    rules = ensure_rules_file()
    canonical = canonical.strip()
    clean_variants = []
    seen = set()
    for variant in variants:
        v = str(variant).strip()
        if not v or v == canonical or v in seen:
            continue
        seen.add(v)
        clean_variants.append(v)

    for rule in rules["aliases"]:
        if (rule.get("canonical") or "").strip() == canonical:
            existing = [str(v).strip() for v in rule.get("variants", []) if str(v).strip()]
            merged = []
            seen2 = set()
            for item in existing + clean_variants:
                if item == canonical or item in seen2:
                    continue
                seen2.add(item)
                merged.append(item)
            rule.update({"variants": merged, "category": category or rule.get("category"), "enabled": True, "updated_at": _now_iso()})
            save_rules(rules)
            return rule

    rule = {
        "canonical": canonical,
        "variants": clean_variants,
        "category": category,
        "enabled": True,
        "updated_at": _now_iso(),
    }
    rules["aliases"].append(rule)
    save_rules(rules)
    return rule



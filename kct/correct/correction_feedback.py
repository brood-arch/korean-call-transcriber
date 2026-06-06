"""Correction feedback loop — accumulate LLM-discovered corrections.

When the unified extraction LLM finds transcription errors (the ``corrections``
field), this module:

1. Accumulates each correction into ``state/correction_suggestions.json``
2. Tracks how many times each (original, corrected) pair has been seen
3. When a pair reaches the configurable threshold (default 3), it is
   automatically promoted to ``correction_rules.json`` as a candidate
   exact-replacement rule with ``enabled: false`` (awaiting human review).

The module is intentionally dependency-free (stdlib only) and compatible
with the existing ``kct.correct.corrections`` rule file format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from kct.config import CORRECTIONS_RULES_PATH, STATE_DIR

log = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────
SUGGESTIONS_PATH = STATE_DIR / "correction_suggestions.json"
PROMOTION_THRESHOLD = 3  # sightings before auto-promotion


def _load_json(path: Path, default: Any) -> Any:
    """Load JSON file with fallback to *default*."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return default


def _atomic_write(path: Path, data: Any) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _make_key(original: str, corrected: str) -> str:
    """Create a dedup key from original + corrected pair."""
    return f"{original}→{corrected}"


def record_corrections(
    corrections: list[dict[str, Any]],
    source: str | None = None,
    *,
    suggestions_path: Path | None = None,
    rules_path: Path | None = None,
    threshold: int = PROMOTION_THRESHOLD,
) -> list[str]:
    """Record LLM-discovered corrections and auto-promote if recurring.

    Parameters
    ----------
    corrections
        List of correction dicts from the LLM extraction output.  Each item
        should have at least ``original`` and ``corrected`` keys.
    source
        Optional source identifier (e.g. transcript filename).
    suggestions_path
        Override path for the suggestions file (useful in tests).
    rules_path
        Override path for the correction rules file (useful in tests).
    threshold
        Number of sightings before auto-promotion.  Defaults to
        ``PROMOTION_THRESHOLD`` (3).

    Returns
    -------
    list[str]
        Keys of newly promoted corrections (empty if none).
    """
    if not corrections:
        return []

    s_path = suggestions_path or SUGGESTIONS_PATH
    r_path = rules_path or CORRECTIONS_RULES_PATH

    suggestions: dict[str, Any] = _load_json(s_path, {})
    promoted: list[str] = []

    for corr in corrections:
        original = (corr.get("original") or "").strip()
        corrected = (corr.get("corrected") or "").strip()
        if not original or not corrected or original == corrected:
            continue

        key = _make_key(original, corrected)
        entry = suggestions.get(key, {
            "original": original,
            "corrected": corrected,
            "count": 0,
            "first_seen": None,
            "last_seen": None,
            "sources": [],
        })

        entry["count"] = entry.get("count", 0) + 1
        entry["last_seen"] = _now_iso()
        if not entry.get("first_seen"):
            entry["first_seen"] = entry["last_seen"]
        if source:
            src_list = entry.get("sources", [])
            if source not in src_list:
                src_list.append(source)
                entry["sources"] = src_list[-20:]  # cap to last 20

        # Auto-promote when threshold is reached
        if entry["count"] >= threshold and not entry.get("promoted"):
            if _promote_to_rules(original, corrected, corr.get("reason"), r_path):
                entry["promoted"] = True
                entry["promoted_at"] = _now_iso()
                promoted.append(key)
                log.info(
                    "Correction auto-promoted after %d sightings: %s",
                    entry["count"],
                    key,
                )

        suggestions[key] = entry

    _atomic_write(s_path, suggestions)
    return promoted


def _promote_to_rules(
    original: str,
    corrected: str,
    reason: str | None,
    rules_path: Path,
) -> bool:
    """Add a candidate rule to correction_rules.json with enabled=false.

    Returns True on success, False on failure.
    """
    rules: dict[str, Any] = _load_json(
        rules_path,
        {"exact_replacements": [], "aliases": []},
    )
    rules.setdefault("exact_replacements", [])

    # Check if already exists (enabled or not)
    for rule in rules["exact_replacements"]:
        if (rule.get("from") or "").strip() == original:
            return False  # already present

    candidate = {
        "from": original,
        "to": corrected,
        "note": f"auto-promoted ({reason})" if reason else "auto-promoted",
        "enabled": False,  # requires human review
        "auto_promoted": True,
        "promoted_at": _now_iso(),
    }
    rules["exact_replacements"].append(candidate)

    try:
        _atomic_write(rules_path, rules)
        return True
    except OSError as exc:
        log.warning("Failed to promote correction to rules: %s", exc)
        return False


def get_suggestions(
    *,
    suggestions_path: Path | None = None,
) -> dict[str, Any]:
    """Return all accumulated correction suggestions."""
    s_path = suggestions_path or SUGGESTIONS_PATH
    return _load_json(s_path, {})


def get_pending_promotions(
    *,
    suggestions_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return suggestions that have reached threshold but are not yet promoted."""
    suggestions = get_suggestions(suggestions_path=suggestions_path)
    return [
        entry
        for entry in suggestions.values()
        if isinstance(entry, dict)
        and entry.get("count", 0) >= PROMOTION_THRESHOLD
        and not entry.get("promoted")
    ]


def _now_iso() -> str:
    """Return current ISO timestamp (KST)."""
    from datetime import timedelta, timezone

    KST = timezone(timedelta(hours=9))
    from datetime import datetime

    return datetime.now(KST).isoformat()

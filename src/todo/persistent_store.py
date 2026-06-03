#!/usr/bin/env python3
"""
Persistent TODO Store — TODOs survive across pipeline runs.

Only removed when user explicitly marks them as completed. Features:
- Same-source deduplication (LLM often extracts overlapping TODOs)
- Cross-source fuzzy dedup via bigram Jaccard similarity (threshold ≥ 0.55)
- Completed-TODO tracking with fuzzy match to prevent re-extraction
- Atomic file writes for crash safety

Environment variables:
    KCT_STATE_DIR — Base state directory (default: state)

Usage:
    from src.todo.persistent_store import load_store, merge_todos, save_store, get_active
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))

# State paths — configurable via env var
_STATE_DIR = Path(os.environ.get("KCT_STATE_DIR", "state"))
PERSISTENT_FILE = _STATE_DIR / "persistent_todos.json"
COMPLETED_FILE = _STATE_DIR / "completed_todos.json"

log = logging.getLogger("persistent_todo_store")


def _now() -> str:
    return datetime.now(KST).isoformat()


# ── Bigram Jaccard fuzzy matching (inline, no external dep) ─────────────

def _char_bigrams(s: str) -> set[str]:
    """Character bigram set, whitespace removed."""
    s = s.replace(" ", "").lower()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _jaccard_similarity(s1: str, s2: str) -> float:
    """Bigram Jaccard similarity (0.0–1.0)."""
    bg1 = _char_bigrams(s1)
    bg2 = _char_bigrams(s2)
    if not bg1 or not bg2:
        return 0.0
    return len(bg1 & bg2) / len(bg1 | bg2)


def _is_fuzzy_match(a: str, b: str, threshold: float = 0.55) -> bool:
    """Check if two strings are similar above threshold."""
    return _jaccard_similarity(a, b) >= threshold


# ── Title normalization ─────────────────────────────────────────────────

def _normalize(title) -> str:
    """Simple normalization for dedup. Handles both str and dict entries."""
    if isinstance(title, dict):
        title = title.get("title", "")
    if not isinstance(title, str):
        return ""
    return title.strip().lower().replace(" ", "").replace("_", "")[:100]


# ── Store I/O ───────────────────────────────────────────────────────────

def load_store() -> dict:
    """Load persistent TODO store from disk.

    Returns:
        Store dict with keys: version, todos (dict), last_updated.
    """
    if PERSISTENT_FILE.exists():
        try:
            return json.loads(PERSISTENT_FILE.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Failed to parse %s, starting fresh", PERSISTENT_FILE)
    return {"version": 1, "todos": {}, "last_updated": _now()}


def save_store(store: dict) -> None:
    """Save persistent TODO store atomically (write-to-temp then rename)."""
    store["last_updated"] = _now()
    PERSISTENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PERSISTENT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PERSISTENT_FILE)


# ── Completed set ────────────────────────────────────────────────────────

def _load_completed_titles() -> set:
    """Load completed TODO titles as normalized set."""
    if not COMPLETED_FILE.exists():
        return set()
    try:
        with open(COMPLETED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    if isinstance(data, list):
        return {
            _normalize(item.get("title", ""))
            for item in data
            if isinstance(item, dict) and item.get("title")
        }
    return {_normalize(t) for t in data.get("completed_titles", [])}


def _is_completed(title: str, completed_set: set) -> bool:
    return _normalize(title) in completed_set


def _is_completed_fuzzy(title: str, completed_set: set, threshold: float = 0.45) -> bool:
    """Check if title matches any completed TODO via fuzzy matching."""
    if _is_completed(title, completed_set):
        return True
    norm = _normalize(title)
    for ct in completed_set:
        if _is_fuzzy_match(norm, ct, threshold):
            return True
    return False


# ── Same-source dedup ───────────────────────────────────────────────────

def _same_source_overlap(t1: str, t2: str, threshold: float = 0.55) -> bool:
    """Check if two TODO titles from the same source overlap in meaning."""
    return _jaccard_similarity(t1, t2) >= threshold


def _dedup_same_source(todos_list: list) -> list:
    """Deduplicate TODOs from the same source that overlap in meaning.

    Merges shorter title into longer one (usually more descriptive).
    """
    if len(todos_list) <= 1:
        return todos_list

    from collections import defaultdict
    by_source = defaultdict(list)
    for t in todos_list:
        by_source[t.get("source", "")].append(t)

    result = []
    merged_count = 0
    for source, items in by_source.items():
        if len(items) <= 1:
            result.extend(items)
            continue
        kept = [items[0]]
        for item in items[1:]:
            title = item.get("title", "")
            merged = False
            for i, existing in enumerate(kept):
                etitle = existing.get("title", "")
                if _same_source_overlap(title, etitle):
                    if len(title) > len(etitle):
                        kept[i] = item
                    merged = True
                    merged_count += 1
                    log.info(
                        "Same-source merge: '%s' + '%s' → '%s'",
                        etitle, title, kept[i]["title"],
                    )
                    break
            if not merged:
                kept.append(item)
        result.extend(kept)

    if merged_count:
        log.info("Same-source dedup: %d merges", merged_count)
    return result


# ── Main merge ──────────────────────────────────────────────────────────

def merge_todos(store: dict, new_todos: list) -> list:
    """Merge newly extracted TODOs into persistent store.

    Pipeline:
        1. Same-source dedup (LLM often extracts overlapping TODOs)
        2. Remove completed TODOs from store (exact + fuzzy)
        3. Add new TODOs not already in store (exact + fuzzy check)
        4. Return list of actually-new TODOs (for notification)

    Args:
        store: The persistent store dict.
        new_todos: List of new TODO dicts (must have 'title' key).

    Returns:
        List of actually-new TODOs that were added.
    """
    # Phase 0: Same-source dedup before merging
    new_todos = _dedup_same_source(new_todos)

    completed_set = _load_completed_titles()
    todos = store.get("todos", {})
    actually_new = []

    # Remove completed TODOs first (exact + fuzzy)
    removed = []
    for key in list(todos.keys()):
        ttitle = todos[key].get("title", "")
        if _is_completed_fuzzy(ttitle, completed_set):
            removed.append(ttitle)
            del todos[key]
    if removed:
        log.info("Removed completed: %s", removed)

    # Add new TODOs
    existing_titles = [v.get("title", "") for v in todos.values()]
    existing_keys = set(todos.keys())

    for t in new_todos:
        title = t.get("title", "").strip()
        if not title:
            continue

        # Skip if completed (exact + fuzzy)
        if _is_completed_fuzzy(title, completed_set):
            log.info("Skipping re-extracted completed TODO: '%s'", title)
            continue

        norm = _normalize(title)
        source = t.get("source", "")
        # Normalize source: strip extensions (.m4a, .txt)
        for ext in (".m4a", ".txt"):
            if source.endswith(ext):
                source = source[: -len(ext)]
                break
        key = f"{norm}|{source}"

        # Skip if already in store
        if source and any(source + ext in existing_keys for ext in (".m4a", ".txt")):
            continue

        # Fuzzy dedup against existing
        if any(_is_fuzzy_match(title, existing) for existing in existing_titles):
            log.info("Fuzzy dedup: '%s' matches existing, skipping", title)
            continue

        entry = {
            "title": title,
            "source": source,
            "priority": t.get("priority", "medium"),
            "status": t.get("status", "active"),
            "details": t.get("details", ""),
            "counterparty": t.get("counterparty", ""),
            "phone": t.get("phone", ""),
            "called_at": t.get("called_at", ""),
            "added_at": _now(),
        }
        for field in (
            "owner", "due_date", "due_time", "created_at", "run_id",
            "source_name", "source_email", "email_subject",
        ):
            if field in t:
                entry[field] = t.get(field)
        todos[key] = entry
        existing_titles.append(title)
        existing_keys.add(key)
        actually_new.append(t)

    store["todos"] = todos
    log.info(
        "Store: %d active, %d new, %d removed",
        len(todos), len(actually_new), len(removed),
    )
    return actually_new


def get_active(store: dict) -> list:
    """Get all active TODOs from persistent store (excluding completed, exact + fuzzy)."""
    completed_set = _load_completed_titles()
    return [
        t for t in store.get("todos", {}).values()
        if not _is_completed_fuzzy(t.get("title", ""), completed_set)
    ]


def todo_key(todo: dict) -> str:
    """Generate a stable key for a TODO entry."""
    return f"{_normalize(todo.get('title', ''))}|{todo.get('source', '')}"


def sync_completed_to_file(
    store: dict,
    completed_path: str = "state/completed_todos.json",
) -> int:
    """Sync completed TODOs to completed_todos.json for dedup.

    Handles both legacy string format and dict format in completed_titles.
    Always writes back as list of dicts with {title, source, status}.

    Args:
        store: The persistent store dict.
        completed_path: Path to the completed TODOs file.

    Returns:
        Number of completed entries after sync.
    """
    import json as _json

    # Collect completed entries from persistent store
    completed: dict[tuple, dict] = {}
    for t in store.get("todos", {}).values():
        if isinstance(t, dict) and t.get("status") == "completed":
            title = t.get("title", "")
            source = t.get("source", "") or ""
            key = (title, source)
            if key not in completed:
                completed[key] = {
                    "title": title,
                    "source": source,
                    "status": "completed",
                    "completed_at": t.get("completed_at", ""),
                }

    # Load existing
    existing: dict[tuple, dict] = {}
    cpath = Path(completed_path)
    if cpath.exists():
        with open(cpath, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        titles_list = raw.get("completed_titles", []) if isinstance(raw, dict) else raw
        for entry in titles_list:
            if isinstance(entry, str) and entry.strip():
                key = (entry.strip(), "")
                if key not in existing:
                    existing[key] = {"title": entry.strip(), "source": "", "status": "completed"}
            elif isinstance(entry, dict):
                title = entry.get("title", "")
                source = entry.get("source", "") or ""
                key = (title, source)
                if key not in existing:
                    existing[key] = entry

    # Merge and write
    all_entries = {**existing, **completed}
    result = sorted(all_entries.values(), key=lambda x: x.get("title", ""))

    cpath.parent.mkdir(parents=True, exist_ok=True)
    with open(cpath, "w", encoding="utf-8") as f:
        _json.dump({"completed_titles": result}, f, ensure_ascii=False, indent=2)
    return len(result)


# ── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        print("Migration from legacy files not supported in open-source version.")
        print("Start with an empty store.")
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        store = load_store()
        active = get_active(store)
        print(f"Persistent TODOs: {len(active)} active / {len(store.get('todos', {}))} total")
        for t in active:
            p = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get("priority", ""), "")
            print(f"  {p} {t.get('title')}")
    elif len(sys.argv) > 1 and sys.argv[1] == "sync":
        store = load_store()
        completed_set = _load_completed_titles()
        before = len(store.get("todos", {}))
        for key in list(store.get("todos", {}).keys()):
            if _is_completed(store["todos"][key].get("title", ""), completed_set):
                del store["todos"][key]
        save_store(store)
        after = len(store.get("todos", {}))
        print(f"Synced: {before} -> {after} (removed {before - after} completed)")
    else:
        print("Usage: persistent_store.py [init|status|sync]")

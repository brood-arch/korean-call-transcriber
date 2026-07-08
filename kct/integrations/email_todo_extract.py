#!/usr/bin/env python3
"""
Email TODO Extraction — Extract action items from archived emails.

Runs after new emails are staged. Only processes NEW emails (not retroactive).
Filters out promotional/advertising emails, excluded senders, and auto-replies.

Uses an OpenAI-compatible LLM API for extraction (configured via env vars).

Environment variables:
    LLM_API_KEY      — API key for the LLM endpoint
    LLM_BASE_URL     — OpenAI-compatible API base URL
    LLM_MODEL        — Model name (default: glm-5-turbo)
    EMAIL_TODO_STATE — Path to extraction state JSON file
    EMAIL_TODO_EXCLUSIONS — Path to sender exclusion list JSON file
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from kct.config import STATE_DIR, get_llm_config
from kct.extract.client import call_llm_json
from kct.pipeline.prompt_security import wrap_untrusted_source
from kct.pipeline.utils import safe_load_json, safe_save_json

log = logging.getLogger("email_todo_extract")

# ── Config (all via env vars) ───────────────────────────────────────────
EXTRACT_STATE_PATH = Path(os.environ.get(
    "EMAIL_TODO_STATE",
    str(STATE_DIR / "email_todo_state.json"),
))
EXCLUSION_PATH = Path(os.environ.get(
    "EMAIL_TODO_EXCLUSIONS",
    str(STATE_DIR / "email_todo_exclusions.json"),
))

KST = timezone(timedelta(hours=9))


def _now() -> str:
    return datetime.now(KST).isoformat()


# ── Promotional email detection ─────────────────────────────────────────
PROMO_SUBJECT_PATTERNS = [
    r"\[광고\]", r"\[AD\]", r"\[EVENT\]", r"\[마케팅\]",
    r"\[프로모션\]", r"\[할인\]", r"\[SALE\]", r"\[ANNOUNCE\]",
    r"뉴스레터", r"newsletter", r"이벤트 안내", r"특가", r"한정.*할인",
    r"수신거부", r"unsubscribe", r"수신동의", r"opt.?out",
    r"쿠폰", r"coupon", r"혜택", r"event 안내",
    r"무료.*체험", r"free trial", r"체험판",
    r"업데이트 소식$", r"update$",
]

PROMO_BODY_KEYWORDS = [
    "수신거부", "unsubscribe", "수신을 원치 않으", "이 메일은 발신전용",
    "광고메일", "promotional email", "마케팅 메일",
]


# ── Exclusion list management ───────────────────────────────────────────
def load_exclusions() -> dict:
    """Load sender exclusion list. Format: {"senders": ["addr@example.com", ...]}"""
    return safe_load_json(EXCLUSION_PATH, default={"senders": []}) or {"senders": []}


def save_exclusions(excl: dict) -> None:
    """Save exclusion list to disk."""
    safe_save_json(EXCLUSION_PATH, excl, origin="email_todo_extract")


def add_exclusion(addr: str) -> dict:
    """Add a sender to the exclusion list."""
    excl = load_exclusions()
    addr = addr.lower().strip()
    if addr not in excl["senders"]:
        excl["senders"].append(addr)
        save_exclusions(excl)
    return excl


def remove_exclusion(addr: str) -> dict:
    """Remove a sender from the exclusion list."""
    excl = load_exclusions()
    addr = addr.lower().strip()
    excl["senders"] = [a for a in excl["senders"] if a != addr]
    save_exclusions(excl)
    return excl


# ── Promotional email filter ────────────────────────────────────────────
def is_promotional(subject: str, body_preview: str) -> bool:
    """Check if an email is promotional/advertising."""
    text = f"{subject} {body_preview[:500]}"
    for pat in PROMO_SUBJECT_PATTERNS:
        if re.search(pat, text, re.I):
            return True
    lower = body_preview[:500].lower()
    for kw in PROMO_BODY_KEYWORDS:
        if kw.lower() in lower:
            return True
    return False


# ── State management ────────────────────────────────────────────────────
def load_state() -> dict:
    """Load extraction state (tracks which UIDs have been processed)."""
    return safe_load_json(EXTRACT_STATE_PATH, default={"extracted_uids": {}, "last_extraction": None}) or {
        "extracted_uids": {},
        "last_extraction": None,
    }


def save_state(state: dict) -> None:
    """Save extraction state to disk."""
    safe_save_json(EXTRACT_STATE_PATH, state, origin="email_todo_extract")


# ── LLM extraction ─────────────────────────────────────────────────────
EXTRACTION_PROMPT = """Extract action items (TODOs) from the following email content.

Rules:
1. Only extract items that require action from the recipient (skip info-only, greetings)
2. Each TODO must be specific and actionable
3. Priority: high (urgent/deadline approaching), medium (normal), low (reference)
4. Include both requests from sender and commitments from recipient
5. If a date/deadline is mentioned, include it
6. Respond in JSON only (no other text)

Response format:
{
  "has_actionable_items": true/false,
  "todos": [
    {
      "title": "One-line TODO summary",
      "priority": "high/medium/low",
      "details": "Details if needed",
      "due_date": "YYYY-MM-DD or null",
      "requested_by": "Requester name"
    }
  ]
}

Email content:
{content}"""

MAX_CONTENT_CHARS = 8000
def call_llm_extract(content: str, api_key: str = "", base_url: str = "", model: str = "") -> dict | None:
    """Call OpenAI-compatible LLM to extract TODOs from email content.

    Args:
        content: Email text content.
        api_key: API key for authentication.
        base_url: OpenAI-compatible API base URL.
        model: Model name to use.

    Returns:
        Parsed JSON dict from LLM response, or None on failure.
    """
    untrusted_content = wrap_untrusted_source("email", content[:MAX_CONTENT_CHARS])
    prompt = EXTRACTION_PROMPT.replace("{content}", untrusted_content, 1)
    result, _usage = call_llm_json(
        prompt,
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=2048,
        timeout=60,
        response_format=False,
    )
    if result is None:
        log.warning("LLM email TODO extraction failed")
    return result


# ── Main extraction pipeline ───────────────────────────────────────────
def _parse_sender(meta: dict) -> tuple[str, str]:
    """Extract (from_addr, from_name) from email meta."""
    from_raw = meta.get("from", "")
    email_match = re.search(r"<([^>]+)>", from_raw)
    from_addr = email_match.group(1).lower().strip() if email_match else from_raw.lower().strip()
    name_match = re.match(r'["\']?(.+?)["\']?\s*<', from_raw)
    from_name = name_match.group(1).strip().strip('"').strip("'") if name_match else from_addr.split("@")[0]
    return from_addr, from_name


def _extract_todos_from_row(row, excluded_addrs, extracted_uids) -> list[dict[str, Any]] | None:
    """Process one email row. Returns list of TODO entries, or None to skip."""
    meta = row.get("meta", {})
    staged_path = row.get("staged", "")
    folder = row.get("folder", "INBOX")

    if folder == "Sent Messages":
        return None

    from_addr, from_name = _parse_sender(meta)

    if from_addr in excluded_addrs:
        log.debug("Skipping excluded sender: %s", from_addr)
        return None

    uid = meta.get("uid", "")
    uid_key = f"{folder}:{uid}"
    if uid_key in extracted_uids:
        return None

    staged = Path(staged_path) if staged_path else None
    if not staged or not staged.exists():
        return None
    content = staged.read_text(encoding="utf-8", errors="replace")

    subject = meta.get("subject", "(no subject)")

    body_preview = content[:500]
    if is_promotional(subject, body_preview):
        log.debug("Skipping promotional email: %s", subject[:50])
        extracted_uids[uid_key] = {"status": "promo_skipped", "at": _now()}
        return None

    result = call_llm_extract(content)
    if not result:
        extracted_uids[uid_key] = {"status": "llm_failed", "at": _now()}
        return None

    if not result.get("has_actionable_items"):
        extracted_uids[uid_key] = {"status": "no_actions", "at": _now()}
        return None

    todos = []
    for todo in result.get("todos", []):
        todo_entry = {
            "title": todo.get("title", "").strip(),
            "priority": todo.get("priority", "medium"),
            "details": todo.get("details", ""),
            "due_date": todo.get("due_date"),
            "requested_by": todo.get("requested_by", from_name),
            "source": "email",
            "source_email": from_addr,
            "source_name": from_name,
            "email_subject": subject,
            "email_uid": uid_key,
            "email_date": meta.get("date", ""),
            "added_at": _now(),
        }
        if todo_entry["title"]:
            todos.append(todo_entry)

    extracted_uids[uid_key] = {
        "status": "extracted",
        "todo_count": len(result.get("todos", [])),
        "at": _now(),
    }
    return todos


def _process_email_rows(rows, excluded_addrs, extracted_uids):
    """Process email rows and return extracted TODOs."""
    all_todos: list[dict[str, Any]] = []
    for row in rows:
        todos = _extract_todos_from_row(row, excluded_addrs, extracted_uids)
        if todos:
            all_todos.extend(todos)
    return all_todos


def extract_todos_from_emails(
    rows: list[dict[str, Any]],
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Extract TODOs from newly archived emails.

    Args:
        rows: List of email row dicts with 'meta' and 'staged' path keys.
        dry_run: If True, don't call LLM or save state.

    Returns:
        List of extracted TODO dicts (with source metadata added).
    """
    if dry_run or not rows:
        return []

    llm_config = get_llm_config()
    if not llm_config.api_key:
        log.warning("No LLM_API_KEY available, skipping TODO extraction")
        return []

    excl = load_exclusions()
    excluded_addrs = {a.lower() for a in excl.get("senders", [])}
    state = load_state()
    extracted_uids = state.setdefault("extracted_uids", {})

    all_todos = _process_email_rows(rows, excluded_addrs, extracted_uids)

    state["last_extraction"] = _now()
    save_state(state)

    if all_todos:
        log.info("Extracted %d TODOs from %d emails", len(all_todos), len(rows))

    return all_todos


# ── CLI interface ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Email TODO extraction")
    sub = parser.add_subparsers(dest="command")

    exc = sub.add_parser("exclude", help="Add sender to exclusion list")
    exc.add_argument("addr", help="Email address to exclude")

    inc = sub.add_parser("include", help="Remove sender from exclusion list")
    inc.add_argument("addr", help="Email address to include again")

    sub.add_parser("list-exclusions", help="List excluded senders")
    sub.add_parser("status", help="Show extraction state")

    args = parser.parse_args()

    if args.command == "exclude":
        result = add_exclusion(args.addr)
        print(f"Excluded: {args.addr}")
        print(f"Total exclusions: {len(result['senders'])}")
    elif args.command == "include":
        result = remove_exclusion(args.addr)
        print(f"Included: {args.addr}")
        print(f"Total exclusions: {len(result['senders'])}")
    elif args.command == "list-exclusions":
        excl = load_exclusions()
        if excl["senders"]:
            print(f"Excluded senders ({len(excl['senders'])}):")
            for a in excl["senders"]:
                print(f"  {a}")
        else:
            print("No excluded senders.")
    elif args.command == "status":
        state = load_state()
        total = len(state.get("extracted_uids", {}))
        statuses: dict[str, int] = {}
        for v in state.get("extracted_uids", {}).values():
            s = v.get("status", "unknown")
            statuses[s] = statuses.get(s, 0) + 1
        print(f"Extracted UIDs: {total}")
        for s, c in sorted(statuses.items()):
            print(f"  {s}: {c}")
        print(f"Last extraction: {state.get('last_extraction', 'never')}")
    else:
        parser.print_help()

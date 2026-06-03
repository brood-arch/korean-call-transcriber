#!/usr/bin/env python3
"""
Naver Mail IMAP Archiver — Archive Naver Mail messages into structured data.

Connects to Naver Mail via IMAP, fetches new messages, and archives them
as structured JSON for downstream TODO extraction, entity analysis, and
knowledge graph construction.

Prerequisites:
    - Enable IMAP in Naver Mail web settings (Settings → POP3/IMAP → IMAP)
    - Set an app-specific password if 2FA is enabled

Environment variables:
    NAVER_MAIL_ADDRESS   — Naver email address (e.g., user@naver.com)
    NAVER_MAIL_PASSWORD  — Naver mail password or app-specific password

Optional:
    NAVER_MAIL_HOST      — IMAP host (default: imap.naver.com)
    NAVER_MAIL_PORT      — IMAP port (default: 993)
    NAVER_MAIL_FOLDERS   — Comma-separated folder list (default: INBOX,"Sent Messages")
    NAVER_MAIL_LIMIT     — Max messages per folder per run (default: 100)
    NAVER_MAIL_STATE_DIR — State directory for tracking processed UIDs
"""
from __future__ import annotations

import email as email_lib
import email.header
import imaplib
import json
import logging
import re
from datetime import timedelta, timezone
from email.message import Message
from pathlib import Path
from typing import Optional

from src.config import (
    NAVER_MAIL_ADDRESS,
    NAVER_MAIL_FOLDERS,
    NAVER_MAIL_HOST,
    NAVER_MAIL_LIMIT,
    NAVER_MAIL_PASSWORD,
    NAVER_MAIL_PORT,
    NAVER_MAIL_STATE_DIR,
)
from src.pipeline.utils import safe_save_json

log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────

KST = timezone(timedelta(hours=9))

IMAP_HOST = NAVER_MAIL_HOST
IMAP_PORT = int(NAVER_MAIL_PORT)
DEFAULT_FOLDERS = [f.strip().strip('"') for f in NAVER_MAIL_FOLDERS.split(",")]
LIMIT = int(NAVER_MAIL_LIMIT)
STATE_DIR = NAVER_MAIL_STATE_DIR


def _get_credentials() -> tuple[str, str]:
    """Read Naver Mail credentials from environment variables."""
    if not NAVER_MAIL_ADDRESS or not NAVER_MAIL_PASSWORD:
        raise EnvironmentError(
            "Set NAVER_MAIL_ADDRESS and NAVER_MAIL_PASSWORD environment variables."
        )
    return NAVER_MAIL_ADDRESS, NAVER_MAIL_PASSWORD


# ── Helpers ────────────────────────────────────────────────────────────

def _safe_slug(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    return re.sub(r"[^\w.-]", "_", text.strip())[:80]


def _decode_header_value(value: Optional[str]) -> str:
    """Decode RFC 2047 encoded header value."""
    if not value:
        return ""
    parts = []
    for raw, enc in email.header.decode_header(value):
        if isinstance(raw, bytes):
            parts.append(raw.decode(enc or "utf-8", errors="replace"))
        else:
            parts.append(raw)
    return " ".join(parts).strip()


def _extract_body(msg: Message) -> str:
    """Extract plain text body from an email message, with HTML fallback."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace") if payload else ""
        # Fallback: HTML
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/html":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace") if payload else ""
        return ""
    else:
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace") if payload else ""


# ── Core Functions ─────────────────────────────────────────────────────

def load_state(state_dir: Path) -> dict:
    """Load processed UID state from JSON file."""
    state_file = state_dir / "processed_uids.json"
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.debug("Failed to load Naver mail state %s: %s", state_file, exc)
    return {}


def save_state(state_dir: Path, state: dict) -> None:
    """Save processed UID state to JSON file."""
    state_file = state_dir / "processed_uids.json"
    safe_save_json(state_file, state, origin="naver_mail")


def parse_message(raw: bytes) -> dict:
    """Parse a raw email into structured metadata dict."""
    msg = email_lib.message_from_bytes(raw)
    return {
        "subject": _decode_header_value(msg.get("Subject", "")),
        "from": _decode_header_value(msg.get("From", "")),
        "to": _decode_header_value(msg.get("To", "")),
        "date": msg.get("Date", ""),
        "message_id": msg.get("Message-ID", ""),
        "body": _extract_body(msg),
    }


def fetch_messages(
    account: str,
    password: str,
    folders: list[str] | None = None,
    limit: int = 100,
    state: dict | None = None,
) -> list[dict]:
    """Fetch new (unprocessed) messages from Naver Mail via IMAP.

    Args:
        account: Naver email address.
        password: Password or app-specific password.
        folders: IMAP folders to scan. Defaults to INBOX + Sent.
        limit: Max messages per folder.
        state: Dict of {folder: set_of_uids} tracking processed messages.

    Returns:
        List of dicts with keys: uid, folder, subject, from, to, date, body.
    """
    if folders is None:
        folders = DEFAULT_FOLDERS
    if state is None:
        state = {}

    results = []
    with imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT) as imap:
        imap.login(account, password)

        for folder in folders:
            # Naver IMAP requires quoted-string for folder names with spaces
            imap_folder = f'"{folder}"' if " " in folder else folder
            status, _ = imap.select(imap_folder)
            if status != "OK":
                log.warning(f"Skipping folder {folder}: select failed")
                continue

            # Fetch all UIDs
            status, data = imap.uid("SEARCH", None, "ALL")
            if status != "OK" or not data[0]:
                continue

            uids = data[0].split()
            processed = set(state.get(folder, []))
            new_uids = [u.decode() for u in uids if u.decode() not in processed]

            # Process newest first, up to limit
            for uid in reversed(new_uids[-limit:]):
                status, msg_data = imap.uid("FETCH", uid, "(RFC822)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue

                raw = msg_data[0][1]
                parsed = parse_message(raw)
                parsed["uid"] = uid
                parsed["folder"] = folder
                results.append(parsed)

                # Mark as processed
                processed.add(uid)

            state[folder] = list(processed)

    return results


# ── CLI Entry Point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Archive Naver Mail via IMAP")
    ap.add_argument("--limit", type=int, default=LIMIT, help="Max messages per folder")
    ap.add_argument("--folder", default=None, help="Single IMAP folder to scan")
    ap.add_argument("--dry-run", action="store_true", help="Don't write state or output files")
    ap.add_argument("--state-dir", default=str(STATE_DIR), help="State directory")
    args = ap.parse_args()

    account, password = _get_credentials()
    folders = [args.folder] if args.folder else DEFAULT_FOLDERS

    state = load_state(Path(args.state_dir))
    messages = fetch_messages(account, password, folders=folders, limit=args.limit, state=state)

    print(f"Fetched {len(messages)} new messages from {account}")

    for msg in messages:
        print(f"  [{msg['folder']}] {msg['subject'][:60]} — {msg['from']}")

    if not args.dry_run and messages:
        save_state(Path(args.state_dir), state)
        print(f"State saved. {sum(len(v) for v in state.values())} total UIDs tracked.")

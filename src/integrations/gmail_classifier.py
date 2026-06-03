#!/usr/bin/env python3
"""
Gmail Auto Classifier — Classify and triage Gmail inbox via IMAP.

- Promotional/advertising emails are automatically moved to trash.
- Important business emails are identified and summarized.
- Classification uses keyword scoring on subject, sender, and body text.

Environment variables:
    GMAIL_ADDRESS      — Gmail email address for IMAP login
    GMAIL_APP_PASSWORD — Gmail app-specific password (16-char, no spaces)

All credentials are read from environment variables; no hardcoded secrets.
"""
from __future__ import annotations

import email as email_lib
import imaplib
import logging
import os
from datetime import datetime
from email.header import decode_header
from typing import Optional

from src.config import GMAIL_ADDRESS, GMAIL_APP_PASSWORD

log = logging.getLogger(__name__)

# Gmail IMAP settings
IMAP_SERVER = "imap.gmail.com"

# Keywords for classification
AD_KEYWORDS = [
    "광고", "홍보", "프로모션", "세일", "할인", "특가",
    "advertising", "promotion", "sale", "discount",
    "뉴스레터", "newsletter", "구독", "subscribe",
    "이벤트", "event", "참여", "응모", "당첨",
    "무료", "free", "체험", "trial",
]

IMPORTANT_KEYWORDS = [
    "긴급", "urgent", "중요", "important",
    "계약", "contract", "견적", "quotation",
    "주문", "order", "발송", "shipping",
    "송장", "invoice", "결제", "payment",
]


def _get_credentials() -> tuple[str, str]:
    """Read Gmail credentials from environment variables.

    Returns:
        Tuple of (email_address, app_password).

    Raises:
        EnvironmentError: If either variable is unset.
    """
    addr = os.environ.get("GMAIL_ADDRESS", GMAIL_ADDRESS)
    pw = os.environ.get("GMAIL_APP_PASSWORD", GMAIL_APP_PASSWORD)
    if not addr or not pw:
        raise EnvironmentError(
            "GMAIL_ADDRESS and GMAIL_APP_PASSWORD environment variables must be set"
        )
    return addr, pw


def connect_gmail() -> imaplib.IMAP4_SSL:
    """Connect to Gmail IMAP using environment-supplied credentials."""
    addr, pw = _get_credentials()
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(addr, pw)
    return mail


def decode_mime_words(s: Optional[str]) -> str:
    """Decode MIME encoded words in email headers."""
    if s is None:
        return ""
    decoded_list = decode_header(s)
    result = []
    for text, encoding in decoded_list:
        if isinstance(text, bytes):
            try:
                text = text.decode(encoding or "utf-8", errors="ignore")
            except (LookupError, UnicodeDecodeError):
                text = text.decode("utf-8", errors="ignore")
        result.append(str(text))
    return "".join(result)


def get_email_subject(msg) -> str:
    """Get decoded email subject."""
    return decode_mime_words(msg.get("Subject", ""))


def get_email_sender(msg) -> str:
    """Get decoded email sender (From header)."""
    return decode_mime_words(msg.get("From", ""))


def classify_email(subject: str, sender: str, body: str) -> str:
    """Classify email as 'ad', 'important', or 'normal'.

    Uses keyword scoring: ads need >=2 ad keywords, important needs >=1
    important keyword, otherwise classified as normal.
    """
    text = f"{subject} {sender} {body}".lower()

    ad_score = sum(1 for kw in AD_KEYWORDS if kw.lower() in text)
    important_score = sum(1 for kw in IMPORTANT_KEYWORDS if kw.lower() in text)

    if ad_score >= 2:
        return "ad"
    elif important_score >= 1:
        return "important"
    else:
        return "normal"


def get_email_body(msg) -> str:
    """Extract plain-text email body."""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    body = payload.decode("utf-8", errors="ignore")
                except Exception as exc:
                    log.debug("Failed to decode multipart text/plain body: %s", exc)
    else:
        try:
            payload = msg.get_payload(decode=True)
            body = payload.decode("utf-8", errors="ignore")
        except Exception as exc:
            log.debug("Failed to decode singlepart email body: %s", exc)
    return body


def scan_inbox(max_emails: int = 50) -> dict:
    """Scan inbox and classify unread emails.

    Returns:
        Dict with keys 'ads', 'important', 'normal', each a list of
        email info dicts with 'id', 'subject', 'sender', 'classification'.
    """
    mail = connect_gmail()
    mail.select("inbox")

    status, messages = mail.search(None, "UNSEEN")
    email_ids = messages[0].split()[:max_emails]

    ads, important, normal = [], [], []

    for email_id in email_ids:
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email_lib.message_from_bytes(response_part[1])
                subject = get_email_subject(msg)
                sender = get_email_sender(msg)
                body = get_email_body(msg)

                classification = classify_email(subject, sender, body)
                email_info = {
                    "id": email_id,
                    "subject": subject,
                    "sender": sender,
                    "classification": classification,
                }

                if classification == "ad":
                    ads.append(email_info)
                elif classification == "important":
                    important.append(email_info)
                else:
                    normal.append(email_info)

    mail.logout()
    return {"ads": ads, "important": important, "normal": normal}


def move_to_trash(mail: imaplib.IMAP4_SSL, email_ids: list) -> None:
    """Move emails to Gmail trash (Bin)."""
    mail.select("inbox")
    for email_id in email_ids:
        try:
            mail.copy(email_id, "[Gmail]/Bin")
            mail.store(email_id, "+FLAGS", "\\Deleted")
        except Exception as e:
            log.error(f"Error moving email {email_id}: {e}")
    mail.expunge()


def run_auto_classifier() -> dict:
    """Main entry point: classify and move ads to trash.

    Returns:
        Classification results dict from scan_inbox().
    """
    print("=" * 80)
    print(f"Gmail Auto Classifier - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = scan_inbox(max_emails=50)

    print("\n[Results]")
    print(f"  - Ads: {len(results['ads'])}")
    print(f"  - Important: {len(results['important'])}")
    print(f"  - Normal: {len(results['normal'])}")

    if results["important"]:
        print("\n[Important Emails]:")
        for i, em in enumerate(results["important"], 1):
            print(f"  {i}. {em['subject']}")
            print(f"     From: {em['sender']}")

    if results["ads"]:
        print("\n[Ads to Move to Trash]:")
        for i, em in enumerate(results["ads"], 1):
            print(f"  {i}. {em['subject']}")

        mail = connect_gmail()
        ad_ids = [e["id"] for e in results["ads"]]
        move_to_trash(mail, ad_ids)
        mail.logout()
        print(f"\n[OK] Moved {len(ad_ids)} ad emails to trash")

    return results


if __name__ == "__main__":
    run_auto_classifier()

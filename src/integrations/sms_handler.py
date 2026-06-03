#!/usr/bin/env python3
"""
SMS Handler — SMS-to-transcription pipeline integration pattern.

This is a placeholder module documenting the integration pattern for
connecting SMS messages to the call transcription pipeline. It defines
the expected interfaces and data flow for SMS-based TODO extraction.

Architecture:
    SMS Gateway → webhook/poll → normalize → extract TODOs → persistent store

Integration patterns supported:
    1. **Webhook receiver**: SMS gateway posts incoming messages to an HTTP endpoint.
       The handler normalizes the payload and passes it to the extraction pipeline.

    2. **Polling**: Periodically fetch messages from an SMS API (e.g., Twilio,
       local Android gateway) and feed new messages into the pipeline.

    3. **File-based**: Monitor a directory for SMS export files and process
       them as they appear.

Expected SMS payload fields:
    - sender: Phone number or contact name
    - message: SMS body text
    - timestamp: ISO 8601 datetime string
    - thread_id: (optional) Conversation thread identifier

Environment variables:
    SMS_GATEWAY_URL  — SMS gateway API endpoint (for polling mode)
    SMS_API_KEY      — API key for SMS gateway authentication
    SMS_WEBHOOK_PORT — Port for webhook receiver (default: 8080)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


KST = timezone(timedelta(hours=9))


@dataclass
class SMSMessage:
    """Normalized SMS message structure.

    All SMS handlers should convert gateway-specific formats into this
    common structure before passing to the extraction pipeline.
    """
    sender: str
    message: str
    timestamp: str  # ISO 8601
    thread_id: str = ""
    raw_data: dict = field(default_factory=dict)


def normalize_sms_payload(payload: dict) -> SMSMessage:
    """Normalize a raw SMS gateway payload into SMSMessage.

    Override this function for your specific SMS gateway format.

    Args:
        payload: Raw payload dict from the SMS gateway.

    Returns:
        Normalized SMSMessage instance.
    """
    return SMSMessage(
        sender=payload.get("from", payload.get("sender", "")),
        message=payload.get("body", payload.get("text", payload.get("message", ""))),
        timestamp=payload.get("timestamp", datetime.now(KST).isoformat()),
        thread_id=payload.get("thread_id", ""),
        raw_data=payload,
    )


def extract_sms_todos(sms: SMSMessage) -> list[dict]:
    """Extract action items from an SMS message.

    This is a placeholder for SMS-specific TODO extraction. In production,
    this would call the LLM extraction pipeline (same as email_todo_extract)
    or use rule-based extraction for structured SMS formats.

    Args:
        sms: Normalized SMS message.

    Returns:
        List of TODO dicts. Empty list by default.
    """
    # Placeholder: in production, pass sms.message through the LLM extraction
    # pipeline or use rule-based patterns for common SMS formats.
    return []


def sms_to_todo_entry(sms: SMSMessage, todos: list[dict]) -> list[dict]:
    """Convert extracted TODOs into persistent store entries with SMS metadata.

    Args:
        sms: Original SMS message.
        todos: Extracted TODO dicts.

    Returns:
        List of TODO entries enriched with SMS source metadata.
    """
    entries = []
    for todo in todos:
        entry = {
            **todo,
            "source": "sms",
            "source_phone": sms.sender,
            "sms_thread": sms.thread_id,
            "sms_timestamp": sms.timestamp,
        }
        entries.append(entry)
    return entries


# ── Integration notes ───────────────────────────────────────────────────
# To integrate with your SMS gateway:
#
# 1. Implement a webhook handler (Flask/FastAPI) that receives POST requests
#    from your SMS gateway and calls normalize_sms_payload().
#
# 2. Pass the normalized SMSMessage to extract_sms_todos() for TODO extraction.
#
# 3. Feed the resulting entries into the persistent TODO store via:
#        from src.todo.persistent_store import load_store, merge_todos, save_store
#
# 4. Optionally, forward the SMS content through the signal detector for
#    entity/idea extraction:
#        from src.knowledge.signal_detector import detect_signals


if __name__ == "__main__":
    # Demo: normalize a sample payload
    sample = {
        "from": "+82-10-XXXX-XXXX",
        "body": "Meeting confirmed for 3pm tomorrow",
        "timestamp": "2026-06-03T10:00:00+09:00",
    }
    sms = normalize_sms_payload(sample)
    print(f"Normalized SMS: {sms}")

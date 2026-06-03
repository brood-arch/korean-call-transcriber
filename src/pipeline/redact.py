"""Sensitive text redaction helpers for logs and diagnostics."""

from __future__ import annotations

import re

# Korean phone numbers: 010-XXXX-XXXX, 010XXXXXXXX, 010 XXXX XXXX,
# landline: 02-XXX-XXXX, area codes 031-064, toll-free 1588-XXXX / 16XX-XXXX
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    # International +82 prefix (no leading 0)
    r"(?:\+82[- ]?)1[016-9][- ]?\d{3,4}[- ]?\d{4}"
    r"|"
    # Domestic 01x mobile
    r"01[016-9][- ]?\d{3,4}[- ]?\d{4}"
    r"|"
    # Landline area codes
    r"0[2-6][1-4]?[- ]?\d{3,4}[- ]?\d{4}"
    r"|"
    # VoIP / toll
    r"0(?:70|80)[- ]?\d{3,4}[- ]?\d{4}"
    r"|"
    # Toll-free 1588 etc
    r"15\d{2}[- ]?\d{4}"
    r"|"
    # Toll-free 1644 etc
    r"16\d{2}[- ]?\d{4}"
    r")"
    r"(?!\d)"
)

# Tokens/keys with common prefixes
_TOKEN_RE = re.compile(r"\b(?:sk|key|tok)-[A-Za-z0-9._-]{8,}\b")

# Key=value pairs where the value looks like a secret
_LONG_KEY_PARAM_RE = re.compile(
    r"(?i)(?:^|[\s,])(api[_-]?key|token|secret|password|bearer|authorization)\s*[:=\s]\s*"
    r"([A-Za-z0-9._-]{16,})"
)

# Bearer token in Authorization header
_BEARER_RE = re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s]+")

# Email (middle part masked)
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])([A-Za-z0-9._%+-]*)(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

# Windows user paths: C:\Users\username\...
_WIN_USER_PATH_RE = re.compile(r"[A-Za-z]:\\Users\\[^\s\"']+")


def redact_sensitive_text(text: str) -> str:
    """Redact credentials and personal identifiers from text."""
    value = str(text or "")
    value = _BEARER_RE.sub(r"\1[REDACTED_BEARER]", value)
    value = _PHONE_RE.sub("[REDACTED_PHONE]", value)
    value = _EMAIL_RE.sub(lambda m: f"{m.group(1)}***{m.group(3)}", value)
    value = _TOKEN_RE.sub("[REDACTED_TOKEN]", value)
    value = _LONG_KEY_PARAM_RE.sub(r"\1=[REDACTED]", value)
    value = _WIN_USER_PATH_RE.sub("[REDACTED_PATH]", value)
    return value

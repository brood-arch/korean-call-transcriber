"""Sensitive text redaction helpers for logs and diagnostics."""

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])([A-Za-z0-9._%+-]*)(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_PHONE_RE = re.compile(r"\b010-\d{4}-\d{4}\b")
_TOKEN_RE = re.compile(r"\b(?:sk|key)-[A-Za-z0-9._-]{8,}\b")
_LONG_KEY_PARAM_RE = re.compile(r"(?i)\b(key|token|secret|password)=([A-Fa-f0-9]{12,}|[A-Za-z0-9._-]{12,})")
_WIN_USER_PATH_RE = re.compile(r"[A-Za-z]:\\Users\\[^\s\"']+")


def redact_sensitive_text(text: str) -> str:
    """Redact credentials and personal identifiers from text."""
    value = str(text or "")
    value = _PHONE_RE.sub("010-****-****", value)
    value = _EMAIL_RE.sub(lambda m: f"{m.group(1)}***{m.group(3)}", value)
    value = _TOKEN_RE.sub("[REDACTED_TOKEN]", value)
    value = _LONG_KEY_PARAM_RE.sub(lambda m: f"{m.group(1)}=[REDACTED_TOKEN]", value)
    value = _WIN_USER_PATH_RE.sub("[REDACTED_PATH]", value)
    return value

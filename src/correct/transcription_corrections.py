"""Transcription corrections — re-exported from corrections module.

This module is retained for backward compatibility.
Use src.correct.corrections directly.
"""
from src.correct.corrections import (
    add_alias,
    add_exact_replacement,
    apply_corrections,
    ensure_rules_file,
    load_rules,
    log_event,
    save_rules,
)

__all__ = [
    "add_alias",
    "add_exact_replacement",
    "apply_corrections",
    "ensure_rules_file",
    "load_rules",
    "log_event",
    "save_rules",
]

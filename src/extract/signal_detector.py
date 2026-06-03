"""Compatibility shim for signal detection.

The canonical implementation lives in :mod:`src.knowledge.signal_detector`.
This module remains so older imports from ``src.extract.signal_detector`` keep
working while avoiding duplicate implementations.
"""

from __future__ import annotations

from src.knowledge.signal_detector import detect_signals, fast_score_transcript, main

__all__ = ["detect_signals", "fast_score_transcript", "main"]

if __name__ == "__main__":
    raise SystemExit(main())

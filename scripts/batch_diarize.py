#!/usr/bin/env python3
"""batch_diarize.py — kct 패키지 래퍼. 롤백: _archive_kct/batch_diarize.py.bak.*"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kct_bridge  # noqa: F401

from kct.transcribe.batch_diarize import main  # noqa: E402


def _inject_default_args() -> None:
    """kct batch_diarize가 --transcript-dir을 받지 않으면
    data/전사본(기본값)을 쓰므로, kct_bridge가 세팅한 환경변수에서
    경로를 주입한다."""
    if "--transcript-dir" not in sys.argv:
        td = os.environ.get("KCT_TRANSCRIPT_DIR")
        if td:
            sys.argv.insert(1, "--transcript-dir")
            sys.argv.insert(2, td)


if __name__ == "__main__":
    _inject_default_args()
    raise SystemExit(main())

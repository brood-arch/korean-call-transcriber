#!/usr/bin/env python3
"""kct_bridge.py — Workspace 환경변수를 kct config에 매핑.

kct 패키지(src/config.py)는 환경변수로 모든 설정을 읽음.
이 브릿지는 workspace의 paths.json + shared_api_keys 값을
kct가 이해하는 환경변수로 노출한다.

Usage:
    # 스크립트 최상단에서 import
    import kct_bridge  # noqa: F401 — sets os.environ for kct

    # 그 후 kct 모듈 import
    from kct.config import TRANSCRIPT_DIR, STATE_DIR
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent


def _load_paths() -> dict:
    p = _WORKSPACE / "paths.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _load_api_keys() -> None:
    """shared_api_keys에서 API 값을 읽어 환경변수로 세팅."""
    sys.path.insert(0, str(_SCRIPTS))
    try:
        from shared_api_keys import (  # noqa: I001
            get_zai_api_key,
            get_zai_base_url,
            get_telegram_config,
            get_langfuse_config,
        )
    except ImportError:
        return

    if not os.environ.get("LLM_API_KEY"):
        os.environ["LLM_API_KEY"] = get_zai_api_key() or ""
    if not os.environ.get("LLM_BASE_URL"):
        os.environ["LLM_BASE_URL"] = get_zai_base_url() or ""

    tg_token, tg_chat = get_telegram_config()
    if tg_token and not os.environ.get("TELEGRAM_BOT_TOKEN"):
        os.environ["TELEGRAM_BOT_TOKEN"] = tg_token
    if tg_chat and not os.environ.get("TELEGRAM_CHAT_ID"):
        os.environ["TELEGRAM_CHAT_ID"] = str(tg_chat)

    try:
        lf_s, lf_p, lf_b = get_langfuse_config()
        if lf_s and not os.environ.get("LANGFUSE_SECRET_KEY"):
            os.environ["LANGFUSE_SECRET_KEY"] = lf_s
        if lf_p and not os.environ.get("LANGFUSE_PUBLIC_KEY"):
            os.environ["LANGFUSE_PUBLIC_KEY"] = lf_p
        if lf_b and not os.environ.get("LANGFUSE_HOST"):
            os.environ["LANGFUSE_HOST"] = lf_b
    except Exception:
        pass


def _is_windows() -> bool:
    return sys.platform == "win32"


def setup() -> None:
    """workspace paths.json → kct 환경변수 매핑."""
    paths = _load_paths()
    # Windows에서는 windows 섹션, WSL에서는 wsl 섹션 사용
    section = "windows" if _is_windows() else "wsl"
    plat = paths.get(section, {})

    env_map = {
        "KCT_WORKSPACE": str(_WORKSPACE),
        "KCT_TRANSCRIPT_DIR": plat.get("transcript_dir", ""),
        "KCT_AUDIO_DIR": plat.get("audio_dir", ""),
        "KCT_STATE_DIR": plat.get("state_dir", ""),
        "KCT_LOG_DIR": plat.get("log_dir", ""),
        "KCT_OUTPUT_DIR": str(_WORKSPACE / "output"),
        "MY_NAME": os.environ.get("MY_NAME", ""),
    }

    for key, val in env_map.items():
        if val and not os.environ.get(key):
            os.environ[key] = val

    _load_api_keys()


# 모듈 import 시 자동 실행
setup()

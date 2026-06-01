#!/usr/bin/env python3
"""
pipeline_utils.py - 공통 파이프라인 유틸리티.

여러 파이프라인 스크립트에 중복되어 있던 TODO 키 정규화, 통화 파일명 파싱,
안전한 JSON I/O 로직을 한 곳에 모은 모듈입니다.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))


def normalize_title(title: str) -> str:
    """TODO 제목을 dedup 키 용도로 정규화한다.

    원본 출처: persistent_todo_store.py:_normalize,
    extract_all.py:_normalize_title, todo_auto_expire.py:normalize
    """
    if isinstance(title, dict):
        title = title.get("title", "")
    if not isinstance(title, str):
        return ""
    return title.strip().lower().replace(" ", "").replace("_", "")[:100]


def normalize_source(source: str) -> str:
    """source에서 알려진 오디오/텍스트 확장자(.m4a/.txt)를 제거한다.

    원본 출처: extract_all.py:_normalize_source,
    persistent_todo_store.py:merge_todos, todo_report.py:format_call_context
    """
    source = str(source or "")
    for ext in (".m4a", ".txt"):
        if source.endswith(ext):
            return source[: -len(ext)]
    return source


def parse_call_context(filename: str) -> dict:
    """통화 녹음 파일명에서 caller, phone, called_at, suffix를 추출한다.

    파일명 형식: '이름_전화번호_YYYYMMDDHHMMSS' 또는
    '이름_전화번호_YYYYMMDDHHMMSS_dddddd'. 이름에 '_'가 포함되어도
    전화번호/타임스탬프 기준으로 파싱한다.

    원본 출처: call_recordings_automation.py:parse_source_name,
    extract_all.py:_parse_call_context, todo_report.py:format_call_context
    """
    base = Path(str(filename or "")).stem
    match = re.match(r"^(.*?)_(\d+)_(\d{14})(?:_(\d{6}))?$", base)
    if not match:
        return {"caller": base, "phone": "", "called_at": "", "suffix": ""}

    caller, phone, stamp, suffix = match.groups()
    called_at = ""
    try:
        called_at = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=KST).isoformat()
    except ValueError:
        called_at = ""

    return {
        "caller": caller,
        "phone": phone,
        "called_at": called_at,
        "suffix": suffix or "",
    }


def todo_key(title: str, source: str) -> str:
    """정규화된 TODO 제목과 정규화된 source로 stable dedup 키를 만든다.

    원본 출처: persistent_todo_store.py:todo_key/_normalize/merge_todos,
    extract_all.py:_normalize_title/_normalize_source
    """
    return f"{normalize_title(title)}|{normalize_source(source)}"


def safe_load_json(path, default=None):
    """JSON 파일을 안전하게 읽고, 파일 없음/JSON 파싱 오류 시 기본값을 반환한다.

    safe IO helper가 있으면 safe_read_json을 참조한다.

    원본 출처: call_recordings_automation.py:safe_io,
    extract_all.py 직접 구현, persistent_todo_store.py:load_store,
    todo_auto_expire.py:safe_load_json
    """
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def safe_save_json(path, data, origin: str | None = None):
    """JSON 파일을 .tmp 파일 경유로 atomic write 한다.

    safe IO helper가 있으면 safe_write_json을 참조한다.

    원본 출처: call_recordings_automation.py:safe_io,
    extract_all.py 직접 구현, persistent_todo_store.py:save_store,
    todo_auto_expire.py:safe_save_json
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def safe_read_json(path, default=None):
    return safe_load_json(path, default=default)


def safe_write_json(path, data, origin: str | None = None):
    return safe_save_json(path, data, origin=origin)


def safe_write_text(path, text: str, origin: str | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)



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

    scripts/safe_io.py가 있으면 safe_read_json을 참조한다.

    원본 출처: call_recordings_automation.py:safe_io,
    extract_all.py 직접 구현, persistent_todo_store.py:load_store,
    todo_auto_expire.py:safe_load_json
    """
    path = Path(path)
    try:
        from safe_io import safe_read_json

        return safe_read_json(path, default=default)
    except ImportError:
        pass

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def safe_save_json(path, data, origin: str | None = None):
    """JSON 파일을 .tmp 파일 경유로 atomic write 한다.

    scripts/safe_io.py가 있으면 safe_write_json을 참조한다.

    원본 출처: call_recordings_automation.py:safe_io,
    extract_all.py 직접 구현, persistent_todo_store.py:save_store,
    todo_auto_expire.py:safe_save_json
    """
    path = Path(path)
    try:
        from safe_io import safe_write_json

        return safe_write_json(path, data, origin=origin or "pipeline_utils")
    except ImportError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


# ── Token compression (inspired by OpenHuman TokenJuice concept) ──────────
# Adapted for Korean call transcripts: reduce token count before LLM API calls
# to cut ZAI API costs without losing signal quality.

# Patterns that add no signal value but consume tokens
_NOISE_PATTERNS: list[tuple[str, str | None]] = [
    # Timestamps in brackets: [00:01:23] → remove
    (r"\[\d{1,2}:\d{2}(?::\d{2})?\]", None),
    # Speaker labels like "화자1:", "Speaker 2:" → remove (keep what follows)
    (r"(?:화자|Speaker|speaker)\s*\d+\s*:\s*", None),
    # Repeated whitespace/newlines
    (r"\n{3,}", "\n\n"),
    (r" {2,}", " "),
    # URLs (rare in transcripts but eat tokens)
    (r"https?://\S+", ""),
    # Consecutive punctuation "..." or "ㅡㅡㅡ"
    (r"[.。]{4,}", "..."),
    (r"[ㅡ\-]{3,}", "—"),
]

# Maximum character budget for transcript before compression
_DEFAULT_BUDGET_CHARS = 12000


def compress_transcript(text: str, budget: int = _DEFAULT_BUDGET_CHARS) -> str:
    """Compress transcript text by removing noise and truncating to budget.

    Unlike OpenHuman's TokenJuice (which targets CLI output), this focuses on
    Korean call transcripts: remove timestamps, speaker labels, and excessive
    whitespace while preserving all substantive content.

    Args:
        text: Raw transcript text.
        budget: Maximum character budget (default 12000 ≈ ~3k tokens).

    Returns:
        Compressed text within budget.
    """
    if not text:
        return text

    # If already under budget, just do light cleanup
    result = text
    for pattern, replacement in _NOISE_PATTERNS:
        result = re.sub(pattern, replacement or "", result)

    # Strip leading/trailing whitespace per line
    result = "\n".join(line.strip() for line in result.split("\n"))
    result = result.strip()

    # Truncate to budget if needed (prefer keeping beginning — usually the
    # most signal-rich part of a call)
    if len(result) > budget:
        result = result[:budget]

    return result


def fallback_summary(content: str, source: str = "") -> dict:
    """Deterministic fallback when LLM extraction fails.

    Inspired by OpenHuman's summarise.rs fallback_summary: if the LLM call
    fails, the pipeline should never stop — produce a best-effort result
    using heuristics instead.

    Returns a minimal extraction dict compatible with extract_all.py's
    unified extraction schema.

    Args:
        content: The transcript text that failed LLM extraction.
        source: Source filename for traceability.

    Returns:
        dict with keys: summary, todos, appointments, entities, products,
        money, risks, corrections — all populated heuristically.
    """
    content = content or ""

    # Heuristic one-line summary: first non-trivial sentence
    first_sentence = ""
    for line in content.split("\n"):
        line = line.strip()
        if len(line) > 10:
            first_sentence = line[:100]
            break

    # Count business keywords for a rough importance indicator
    business_kw = [
        "주문", "견적", "발주", "납기", "결제", "입금", "출고", "배송",
        "수령", "방문", "연락", "확인", "처리", "준비", "요청",
    ]
    kw_hits = sum(1 for kw in business_kw if kw in content)

    return {
        "summary": {
            "one_line": first_sentence or "(LLM 추출 실패 - 수동 확인 필요)",
            "details": [f"[자동 폴백] 키워드 {kw_hits}개 감지됨"],
            "call_type": "unknown",
            "overall_confidence": 0.1,
            "fallback": True,
        },
        "todos": [],
        "appointments": [],
        "entities": [],
        "products": [],
        "money": [],
        "risks": [],
        "corrections": [],
        "source": source,
        "fallback_reason": "llm_extraction_failed",
    }

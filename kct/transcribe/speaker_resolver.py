#!/usr/bin/env python3
"""전화번호 기반 화자 자동 매핑 모듈.

전사 파일명에서 전화번호를 추출하여 연락처 DB와 매칭한 뒤,
기존 ``map_speakers()`` 결과("회사"/"신혁" 등)를 실제 이름으로 교체한다.

파일명 패턴:
  - ``YYYYMMDD_HHMM_거래처명_010XXXXXXXX.m4a``
  - ``YYYYMMDD_HHMM_010XXXXXXXX.m4a``

연락처 DB는 JSON 파일(``state/contacts.json``)에서 로드하며,
파일이 없으면 빈 딕셔너리를 사용한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_contacts(path: Path | str | None = None) -> dict[str, str]:
    """연락처 JSON 파일을 로드한다.

    Args:
        path: 연락처 JSON 파일 경로.
              ``None``이면 기본 경로(``state/contacts.json``)를 사용한다.

    Returns:
        전화번호(하이픈 제거) → 이름 매핑 딕셔너리.
        파일이 없거나 파싱에 실패하면 빈 딕셔너리를 반환한다.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text("utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    # 키에서 하이픈/공백 제거
    normalized: dict[str, str] = {}
    for key, value in data.items():
        clean_key = re.sub(r"[\-\s]", "", str(key))
        if clean_key and isinstance(value, str) and value:
            normalized[clean_key] = value
    return normalized


def extract_phone_from_filename(filename: str) -> str | None:
    """파일명에서 전화번호를 추출한다.

    지원 패턴:
      - ``YYYYMMDD_HHMM_거래처명_010XXXXXXXX.ext``
      - ``YYYYMMDD_HHMM_010XXXXXXXX.ext``
      - ``거래처명_010XXXXXXXX.ext``

    Args:
        filename: 파일명(경로 포함 가능).

    Returns:
        추출된 전화번호(하이픈 제거) 또는 ``None``.
    """
    stem = Path(filename).stem
    # 마지막에 위치한 한국 휴대전화번호 패턴 (010, 011, 016~019)
    m = re.search(r"(0(?:1[0-6789]|10)\d{7,8})", stem)
    if m:
        return m.group(1)
    return None


def extract_counterparty_from_filename(filename: str) -> str | None:
    """파일명에서 거래처명을 추출한다.

    지원 패턴:
      - ``YYYYMMDD_HHMM_거래처명_010XXXXXXXX.ext`` (4+2+거래처+번호)
      - ``거래처명_010XXXXXXXX.ext`` (거래처+번호)

    Args:
        filename: 파일명(경로 포함 가능).

    Returns:
        거래처명 또는 ``None``.
    """
    stem = Path(filename).stem
    # 패턴 1: YYYYMMDD_HHMM_거래처명_010XXXXXXXX
    m = re.match(r"^\d{8}_\d{4}_(.+?)_(0(?:1[0-6789]|10)\d{7,8})$", stem)
    if m:
        return m.group(1)
    # 패턴 2: 거래처명_010XXXXXXXX (날짜 접두 없음, 전화번호 앞이 순수 날짜+시간이면 제외)
    m = re.match(r"^(.+?)_(0(?:1[0-6789]|10)\d{7,8})$", stem)
    if m:
        candidate = m.group(1)
        # 날짜_시간 패턴(예: 20260506_1430)은 거래처명이 아님
        if not re.match(r"^\d{8}_\d{4}$", candidate):
            return candidate
    return None


def resolve_speaker_names(
    segments: list[dict[str, Any]],
    filename: str,
    contacts: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """전화번호 기반으로 화자 레이블을 실제 이름으로 교체한다.

    ``map_speakers()``가 이미 "회사"/"신혁" 등으로 레이블링한 세그먼트에서
    "회사"에 해당하는 화자를 연락처 DB의 실제 이름으로 대체한다.

    대체 규칙:
      1. 파일명에서 전화번호 추출 → contacts에서 이름 조회
      2. 파일명에서 거래처명 추출 (fallback)
      3. 둘 다 없으면 세그먼트를 변경하지 않음

    Args:
        segments: 화자 레이블이 포함된 전사 세그먼트 리스트.
                  각 dict는 최소한 ``"speaker"`` 키를 가져야 한다.
        filename: 원본 오디오 파일명.
        contacts: 전화번호 → 이름 매핑. ``None``이면 이름 대체를 시도하지 않는다.

    Returns:
        화자 레이블이 교체된 세그먼트 리스트 (새 리스트, 원본 수정 안 함).
    """
    if contacts is None:
        contacts = {}

    # 파일명에서 전화번호 및 거래처명 추출
    phone = extract_phone_from_filename(filename)
    counterparty = extract_counterparty_from_filename(filename)

    # 연락처에서 실제 이름 조회
    resolved_name: str | None = None
    if phone and phone in contacts:
        resolved_name = contacts[phone]

    # fallback: 거래처명이 있고 연락처 매칭이 안 됐으면 거래처명 사용은 하지 않음
    # (거래처명은 이미 map_speakers에서 caller_name으로 들어갔을 가능성이 높음)
    if not resolved_name:
        return [dict(seg) for seg in segments]

    # 새 리스트 생성 (원본 수정 방지)
    new_segments: list[dict[str, Any]] = []
    for seg in segments:
        new_seg = dict(seg)
        speaker = new_seg.get("speaker", "")
        # "회사" 또는 파일명의 거래처명에 해당하는 화자를 실제 이름으로 교체
        # map_speakers()는 caller_name을 직접 사용하므로,
        # 전화번호로 조회된 이름이 기존 speaker와 다르면 교체
        if speaker and phone:
            # 전화번호로 식별된 상대방의 기존 레이블을 찾아 교체
            # map_speakers()는 caller_name을 그대로 사용하므로
            # 거래처명 기반 레이블이나 "회사" 등을 대체
            if speaker == counterparty or speaker == "회사":
                new_seg["speaker"] = resolved_name
        new_segments.append(new_seg)

    return new_segments

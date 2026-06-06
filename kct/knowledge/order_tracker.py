#!/usr/bin/env python3
"""주문 추적 체인 모듈.

같은 상대방과의 통화를 타임라인으로 엮어서
주문→견적→발주→납기→결제 흐름을 추적한다.

입력: 추출 결과 JSON (summary.call_type, todos, entities, money 등)
출력: ``state/order_tracking.json``에 저장되는 추적 상태

상태 머신:
  ``order`` → ``quote`` → ``po`` → ``delivery`` → ``payment`` → ``complete``
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 상태 정의 (순서대로 진행)
STATES = ("order", "quote", "po", "delivery", "payment", "complete")

# 통화 유형 → 상태 매핑 힌트
CALL_TYPE_HINTS: dict[str, str] = {
    "주문": "order",
    "견적": "quote",
    "견적요청": "quote",
    "발주": "po",
    "납기": "delivery",
    "납품": "delivery",
    "결제": "payment",
    "입금": "payment",
    "수금": "payment",
}

# 금액 관련 키워드
MONEY_KEYWORDS = ("원", "만원", "억", "금액", "단가", "총액", "가격")

# 기본 상태 저장 경로
DEFAULT_STATE_PATH = Path("state") / "order_tracking.json"


def _extract_counterparty(result: dict[str, Any]) -> str | None:
    """추출 결과에서 상대방 식별자를 추출한다.

    우선순위:
      1. ``result["counterparty"]``
      2. ``result["metadata"]["counterparty"]``
      3. ``result["summary"]["counterparty"]``
      4. 첫 번째 entity 중 인물/회사 type
    """
    if result.get("counterparty"):
        return str(result["counterparty"])

    meta = result.get("metadata", {})
    if isinstance(meta, dict) and meta.get("counterparty"):
        return str(meta["counterparty"])

    summary = result.get("summary", {})
    if isinstance(summary, dict) and summary.get("counterparty"):
        return str(summary["counterparty"])

    entities = result.get("entities", [])
    if isinstance(entities, list):
        for ent in entities:
            if isinstance(ent, dict):
                etype = ent.get("type", "").lower()
                if etype in ("person", "company", "organization", "거래처", "회사"):
                    name = ent.get("name") or ent.get("value")
                    if name:
                        return str(name)

    return None


def _extract_phone(result: dict[str, Any]) -> str | None:
    """추출 결과에서 전화번호를 추출한다."""
    meta = result.get("metadata", {})
    if isinstance(meta, dict) and meta.get("phone"):
        return re.sub(r"[\-\s]", "", str(meta["phone"]))

    entities = result.get("entities", [])
    if isinstance(entities, list):
        for ent in entities:
            if isinstance(ent, dict):
                etype = ent.get("type", "").lower()
                if etype in ("phone", "전화번호"):
                    val = ent.get("value") or ent.get("name")
                    if val:
                        return re.sub(r"[\-\s]", "", str(val))

    return None


def _infer_stage(result: dict[str, Any]) -> str:
    """추출 결과에서 주문 단계를 추론한다.

    call_type, todos, money 정보를 종합하여 판단한다.
    """
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    call_type = summary.get("call_type", "")
    if isinstance(call_type, str):
        for keyword, stage in CALL_TYPE_HINTS.items():
            if keyword in call_type:
                return stage

    # todos에서 단계 힌트 추출
    todos = result.get("todos", [])
    if isinstance(todos, list):
        all_text = " ".join(
            t.get("text", "") if isinstance(t, dict) else str(t)
            for t in todos
        )
        for keyword, stage in CALL_TYPE_HINTS.items():
            if keyword in all_text:
                return stage

    # 금액 정보가 있고 call_type에 단서가 없으면 payment 단계 가정
    money = result.get("money", [])
    if isinstance(money, list) and len(money) > 0:
        return "payment"

    return "order"  # 기본값


def _extract_amounts(result: dict[str, Any]) -> list[dict[str, Any]]:
    """추출 결과에서 금액 정보를 추출한다."""
    amounts: list[dict[str, Any]] = []
    money = result.get("money", [])
    if isinstance(money, list):
        for item in money:
            if isinstance(item, dict):
                amounts.append(item)
    return amounts


def _state_index(stage: str) -> int:
    """상태의 진행 순서 인덱스를 반환한다."""
    if stage in STATES:
        return STATES.index(stage)
    return 0


def _counterparty_key(counterparty: str | None, phone: str | None) -> str:
    """상대방 식별 키를 생성한다."""
    if phone:
        return f"phone:{phone}"
    if counterparty:
        return f"name:{counterparty}"
    return ""


class OrderTracker:
    """주문 추적 체인 관리자.

    같은 상대방(전화번호/거래처명)과의 통화를 타임라인으로 엮어서
    주문→견적→발주→납기→결제 흐름을 추적한다.

    Args:
        state_path: 추적 상태를 저장할 JSON 파일 경로.
                    ``None``이면 ``state/order_tracking.json``을 사용한다.

    Attributes:
        state_path: 상태 파일 경로.
        orders: 거래처별 주문 추적 데이터.
    """

    def __init__(self, state_path: Path | str | None = None) -> None:
        if state_path is None:
            self.state_path = DEFAULT_STATE_PATH
        else:
            self.state_path = Path(state_path)
        self.orders: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """상태 파일에서 기존 추적 데이터를 로드한다."""
        if not self.state_path.exists():
            self.orders = {}
            return
        try:
            data = json.loads(self.state_path.read_text("utf-8-sig"))
            if isinstance(data, dict) and "orders" in data:
                self.orders = data["orders"]
            elif isinstance(data, dict):
                self.orders = data
        except (json.JSONDecodeError, OSError):
            self.orders = {}

    def save(self) -> None:
        """현재 추적 상태를 파일에 저장한다."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "schema_version": 1,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "orders": self.orders,
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.state_path)

    def ingest_extraction(self, result: dict[str, Any]) -> str | None:
        """추출 결과를 수용하여 주문 타임라인을 업데이트한다.

        같은 상대방의 기존 타임라인에 새 이벤트를 추가하고,
        상태 머신의 진행 상태를 갱신한다.

        Args:
            result: 추출 결과 dict.
                    ``summary``, ``todos``, ``entities``, ``money`` 등의 키를 가질 수 있다.

        Returns:
            업데이트된 상대방 식별 키. 매칭 실패 시 ``None``.
        """
        counterparty = _extract_counterparty(result)
        phone = _extract_phone(result)
        key = _counterparty_key(counterparty, phone)

        if not key:
            return None

        # 추출 결과에서 메타데이터 구성
        now = datetime.now(timezone.utc).isoformat()
        source = result.get("source_file") or result.get("metadata", {}).get("source_file", "")
        stage = _infer_stage(result)
        amounts = _extract_amounts(result)
        summary = result.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        event: dict[str, Any] = {
            "timestamp": now,
            "source_file": source,
            "stage": stage,
            "call_type": summary.get("call_type", ""),
            "todos": result.get("todos", []),
            "amounts": amounts,
        }

        if key not in self.orders:
            # 새 거래처: 타임라인 생성
            self.orders[key] = {
                "counterparty": counterparty,
                "phone": phone,
                "current_stage": stage,
                "created_at": now,
                "updated_at": now,
                "timeline": [event],
            }
        else:
            # 기존 거래처: 타임라인에 이벤트 추가
            order = self.orders[key]
            order["timeline"].append(event)
            order["updated_at"] = now

            # counterparty/phone 보완 (이전에 누락된 경우)
            if not order.get("counterparty") and counterparty:
                order["counterparty"] = counterparty
            if not order.get("phone") and phone:
                order["phone"] = phone

            # 상태 머신: 뒤로 가지 않음 (더 진행된 단계로만 갱신)
            new_idx = _state_index(stage)
            current_idx = _state_index(order.get("current_stage", "order"))
            if new_idx > current_idx:
                order["current_stage"] = stage

        return key

    def get_timeline(self, counterparty: str) -> list[dict[str, Any]]:
        """특정 상대방과의 통화 타임라인을 반환한다.

        상대방 식별은 이름 또는 전화번호로 시도한다.

        Args:
            counterparty: 거래처명 또는 전화번호.

        Returns:
            타임라인 이벤트 리스트. 매칭 실패 시 빈 리스트.
        """
        # 전화번호로 직접 조회
        phone_clean = re.sub(r"[\-\s]", "", counterparty)
        phone_key = f"phone:{phone_clean}"
        if phone_key in self.orders:
            return list(self.orders[phone_key].get("timeline", []))

        # 이름으로 조회
        name_key = f"name:{counterparty}"
        if name_key in self.orders:
            return list(self.orders[name_key].get("timeline", []))

        # 부분 매칭 (counterparty 필드에 포함되어 있는지)
        for order_data in self.orders.values():
            cp = order_data.get("counterparty", "")
            ph = order_data.get("phone", "")
            if counterparty in cp or counterparty in ph:
                return list(order_data.get("timeline", []))

        return []

    def get_active_orders(self) -> list[dict[str, Any]]:
        """진행 중인(완료되지 않은) 주문 목록을 반환한다.

        Returns:
            ``current_stage``가 ``"complete"``가 아닌 주문 리스트.
        """
        active: list[dict[str, Any]] = []
        for key, order_data in self.orders.items():
            if order_data.get("current_stage") != "complete":
                active.append({"key": key, **order_data})
        return active

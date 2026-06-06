"""LLM prompt construction for unified extraction.

Provides the hardcoded unified-extraction prompt and optional
Langfuse-based prompt loading.
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# --- Few-shot examples for manufacturing B2B calls ---
_FEWSHOT_EXAMPLES = r"""

--- FEW-SHOT EXAMPLES (참고용, 동일 형식으로 출력할 것) ---

예시 1: 견적 요청 통화
전사:
A: 네, 삼성엔지니어링 김대리입니다.
B: 아 네, 신형송풍기 오종서 과장입니다. 저번에 문의하신 환풍기 케이스 견적 나왔나요?
A: 아 네, 케이스 50대 견적입니다. 단가는 12만 원이고요, 납기는 2주 정도 걸릴 것 같습니다.
B: 네, 그러면 견적서 팩스로 좀 보내주세요.

출력 (JSON):
{
  "summary": {
    "one_line": "삼성엔지니어링 김대리에게 환풍기 케이스 50대 견적 확인 및 견적서 발송 요청",
    "details": ["단가 12만 원/대", "납기 2주", "견적서 팩스 발송 요청"],
    "call_type": "quote",
    "overall_confidence": 0.9
  },
  "todos": [
    {
      "id_hint": "quote_fax_send",
      "title": "환풍기 케이스 견적서 팩스 발송",
      "owner": "me",
      "priority": "medium",
      "urgency": "this_week",
      "status": "new",
      "due_date": null,
      "due_time": null,
      "context": "삼성엔지니어링 김대리에게 케이스 50대 견적서 팩스 요청",
      "source_quote": "견적서 팩스로 좀 보내주세요",
      "confidence": 0.95
    }
  ],
  "appointments": [],
  "entities": [
    {"name": "김대리", "type": "Person", "canonical_name": null, "role": "customer", "attributes": {"company": "삼성엔지니어링"}, "source_quote": "삼성엔지니어링 김대리입니다", "confidence": 0.95},
    {"name": "삼성엔지니어링", "type": "Organization", "canonical_name": null, "role": "customer", "attributes": {}, "source_quote": "삼성엔지니어링 김대리입니다", "confidence": 0.95},
    {"name": "오종서 과장", "type": "Person", "canonical_name": null, "role": "employee", "attributes": {"company": "신형송풍기"}, "source_quote": "신형송풍기 오종서 과장입니다", "confidence": 0.95}
  ],
  "products": [
    {"name": "환풍기 케이스", "canonical_name": null, "category": "케이스", "spec": null, "quantity": {"value": 50, "unit": "대"}, "action": "quote", "source_quote": "케이스 50대 견적", "confidence": 0.95}
  ],
  "money": [
    {"amount": 120000, "currency": "KRW", "kind": "price", "related_to": "환풍기 케이스 50대", "payment_status": "unknown", "due_date": null, "source_quote": "단가는 12만 원이고요", "confidence": 0.9}
  ],
  "risks": [],
  "corrections": []
}

예시 2: 납기 확인 통화
전사:
A: 여보세요, 한라산업 박과장입니다. 저번 주에 주문한 송풍기 날개 200장 언제쯤 받을 수 있나요?
B: 아 한라산업 박과장님, 날개 200장은 이번 주 금요일까지는 출고 가능합니다.
A: 금요일이요? 그러면 월요일에는 도착하겠네요?
B: 네, 월요일 오전에 도착할 것입니다.
A: 좋습니다. 그리고 채반 대자 10개도 같이 보내주세요.

출력 (JSON):
{
  "summary": {
    "one_line": "한라산업 박과장 날개 200장 납기 확인 - 금요일 출고, 월요일 도착 예정",
    "details": ["송풍기 날개 200장 금요일 출고", "월요일 오전 도착 예정", "채반 대자 10개 추가 발송 요청"],
    "call_type": "delivery",
    "overall_confidence": 0.9
  },
  "todos": [
    {
      "id_hint": "blade_200_ship",
      "title": "송풍기 날개 200장 출고 준비 (금요일)",
      "owner": "me",
      "priority": "high",
      "urgency": "this_week",
      "status": "new",
      "due_date": null,
      "due_time": null,
      "context": "한라산업 날개 200장 금요일 출고 약속",
      "source_quote": "이번 주 금요일까지는 출고 가능합니다",
      "confidence": 0.9
    },
    {
      "id_hint": "basket_10_add",
      "title": "채반 대자 10개 동봉 발송",
      "owner": "me",
      "priority": "medium",
      "urgency": "this_week",
      "status": "new",
      "due_date": null,
      "due_time": null,
      "context": "박과장 추가 요청, 날개와 함께 발송",
      "source_quote": "채반 대자 10개도 같이 보내주세요",
      "confidence": 0.9
    }
  ],
  "appointments": [],
  "entities": [
    {"name": "박과장", "type": "Person", "canonical_name": null, "role": "customer", "attributes": {"company": "한라산업"}, "source_quote": "한라산업 박과장입니다", "confidence": 0.95},
    {"name": "한라산업", "type": "Organization", "canonical_name": null, "role": "customer", "attributes": {}, "source_quote": "한라산업 박과장입니다", "confidence": 0.95}
  ],
  "products": [
    {"name": "송풍기 날개", "canonical_name": null, "category": "날개", "spec": null, "quantity": {"value": 200, "unit": "장"}, "action": "deliver", "source_quote": "날개 200장", "confidence": 0.95},
    {"name": "채반 대자", "canonical_name": null, "category": "채반", "spec": null, "quantity": {"value": 10, "unit": "개"}, "action": "deliver", "source_quote": "채반 대자 10개도 같이 보내주세요", "confidence": 0.95}
  ],
  "money": [],
  "risks": [],
  "corrections": []
}

예시 3: 대금 결제 통화
전사:
A: 네, 대풍전기 이사님, 저번 달 거래명세서 잔금 언제 입금 가능하신가요?
B: 아, 죄송합니다. 이번 주 안으로 입금하겠습니다. 총 450만 원이죠?
A: 네 맞습니다. 450만 원이고, 오늘까지 입금해주시면 감사하겠습니다.
B: 네, 알겠습니다. 오늘 오후에 입금하겠습니다.

출력 (JSON):
{
  "summary": {
    "one_line": "대풍전기 잔금 450만 원 입금 독촉 - 오늘 오후 입금 약속",
    "details": ["지난달 거래명세서 잔금 450만 원", "오늘 오후 입금 약속"],
    "call_type": "payment",
    "overall_confidence": 0.9
  },
  "todos": [
    {
      "id_hint": "payment_confirm",
      "title": "대풍전기 잔금 450만 원 입금 확인",
      "owner": "me",
      "priority": "high",
      "urgency": "immediate",
      "status": "new",
      "due_date": null,
      "due_time": null,
      "context": "오늘 오후 입금 약속, 입금 확인 필요",
      "source_quote": "오늘 오후에 입금하겠습니다",
      "confidence": 0.95
    }
  ],
  "appointments": [],
  "entities": [
    {"name": "이사님", "type": "Person", "canonical_name": null, "role": "customer", "attributes": {"company": "대풍전기"}, "source_quote": "대풍전기 이사님", "confidence": 0.85},
    {"name": "대풍전기", "type": "Organization", "canonical_name": null, "role": "customer", "attributes": {}, "source_quote": "대풍전기 이사님", "confidence": 0.95}
  ],
  "products": [],
  "money": [
    {"amount": 4500000, "currency": "KRW", "kind": "balance", "related_to": "지난달 거래명세서", "payment_status": "unpaid", "due_date": null, "source_quote": "총 450만 원이죠", "confidence": 0.95}
  ],
  "risks": [
    {
      "severity": "medium",
      "type": "payment_delay",
      "description": "대풍전기 잔금 450만 원 입금 지연, 오늘 약속",
      "recommended_action": "오늘 오후 입금 확인, 미입금 시 내일 재독촉",
      "source_quote": "이번 주 안으로 입금하겠습니다",
      "confidence": 0.85
    }
  ],
  "corrections": []
}

--- END EXAMPLES ---
"""

# --- Unified extraction prompt ---
UNIFIED_EXTRACT_PROMPT = """다음 통화 전사본에서 8가지를 한꺼번에 추출해줘:
1. 요약 (Summary) - 통화 핵심 1문장 + 상세 bullet
2. 할 일 (TODO) - tasks that need to be handled by the user
3. 일정/약속 (Appointment) - 날짜·시간이 명확한 약속
4. 엔티티 (Entity) - 사람, 조직, 장소, 전화번호, 제품명 등
5. 제품 (Product) - 제품/부품/규격/수량
6. 금액 (Money) - 가격, 입금, 잔금, 배송비 등
7. 위험 신호 (Risk) - 고객 불만, 납기 지연, 미수금 등
8. 교정 (Correction) - 전사 오류 교정 사항

출력 형식 (반드시 JSON, 다른 텍스트 금지):
{{
  "summary": {{
    "one_line": "통화 핵심 1문장 요약",
    "details": ["상세 내용 bullet 1", "상세 내용 bullet 2"],
    "call_type": "order|delivery|as|quote|payment|schedule|internal|personal|unknown",
    "overall_confidence": 0.0~1.0
  }},
  "todos": [
    {{
      "id_hint": "todo_핵심어",
      "title": "할 일 제목 (간결하게)",
      "owner": "me|partner|unknown",
      "priority": "high|medium|low",
      "urgency": "immediate|today|this_week|low",
      "status": "new|in_progress|waiting|done|cancelled",
      "due_date": "YYYY-MM-DD 또는 null",
      "due_time": "HH:MM 또는 null",
      "context": "짧은 근거/문맥 (1문장)",
      "source_quote": "전사문 원문 인용",
      "confidence": 0.0~1.0
    }}
  ],
  "appointments": [
    {{
      "title": "일정 제목",
      "date": "YYYY-MM-DD",
      "time": "HH:MM 또는 null",
      "timezone": "Asia/Seoul",
      "location": "장소 또는 null",
      "participants": ["참석자1", "참석자2"],
      "description": "짧은 설명",
      "source_quote": "전사문 원문 인용",
      "confidence": 0.0~1.0
    }}
  ],
  "entities": [
    {{
      "name": "엔티티 이름",
      "type": "Person|Organization|Location|PhoneNumber|Product|Project|Event|Contract|Other",
      "canonical_name": "정규화된 이름 또는 null",
      "role": "customer|supplier|employee|carrier|unknown|null",
      "attributes": {{}},
      "source_quote": "전사문 원문 인용",
      "confidence": 0.0~1.0
    }}
  ],
  "products": [
    {{
      "name": "제품명",
      "canonical_name": "정규화된 제품명 또는 null",
      "category": "환풍기|송풍기|케이스|날개|채반|부품|기타|unknown",
      "spec": "규격 또는 null",
      "quantity": {{"value": 0, "unit": "개|대|박스|세트|장|unknown|null"}},
      "action": "quote|order|deliver|repair|check_stock|manufacture|unknown",
      "source_quote": "전사문 원문 인용",
      "confidence": 0.0~1.0
    }}
  ],
  "money": [
    {{
      "amount": 0,
      "currency": "KRW",
      "kind": "price|deposit|balance|shipping|discount|tax|unknown",
      "related_to": "관련 제품/계약 또는 null",
      "payment_status": "paid|unpaid|partial|unknown",
      "due_date": "YYYY-MM-DD 또는 null",
      "source_quote": "전사문 원문 인용",
      "confidence": 0.0~1.0
    }}
  ],
  "risks": [
    {{
      "severity": "high|medium|low",
      "type": (
          "missed_deadline|payment_delay|customer_complaint|"
          "stock_shortage|quality_issue|privacy|"
          "ambiguous_request|other"
      ),
      "description": "위험 설명",
      "recommended_action": "권장 조치 또는 null",
      "source_quote": "전사문 원문 인용",
      "confidence": 0.0~1.0
    }}
  ],
  "corrections": [
    {{
      "original": "원문",
      "corrected": "교정문",
      "rule_id": "규칙 ID 또는 null",
      "reason": "exact_rule|alias|contextual|spacing|number_normalization|other",
      "source_quote": "전사문 원문 인용",
      "confidence": 0.0~1.0
    }}
  ]
}}

규칙:
- **실행 주체 판단 (가장 중요):**
  - 상대방이 직접 수행하는 일(상대방이 발송, 상대방이 제조, 상대방이 배송 등)은
    TODO에서 제외하거나 owner="상대방"으로 표시
  - 단, 상대방이 우리에게 무언가를 보낸다면 그건 우리 입장에선
    "수령 대기"/"입고 확인"이지 "발송"/"배송 준비"가 아님
  - 항상 화자(사용자 vs 상대방)를 기준으로 누가 행동의 주체인지 파악할 것
  - 예시: 상대방이 "내일 제품 3개 보내겠다" → title: "상대방 제품 3개 수령 대기", owner: "me"
    (우리가 받아야 하므로)
  - 예시: 사용자가 "내일 보내드리겠습니다" → title: "제품 발송", owner: "me"
  - 예시: 상대방이 "내일 방문하겠다" → TODO에서 제외 (상대방이 하는 일)
- 날짜/시간이 불명확한 일정은 date/time을 null로
- 같은 할 일을 여러 버전으로 나누지 말 것 (의미가 같으면 하나로)
  예: "수요일에 케이스 조립"과 "빈 케이스 20개 준비"가 같은 작업이면 "수요일 케이스 20개 조립" 하나로 합칠 것
  예: "가격 확인"과 "가격 문자 전송"은 서로 다른 행동이므로 분리 OK
- 단순 정보 공유, 인사말, 이미 끝난 행동은 제외
- 엔티티는 중복 이름이면 한 번만 (같은 사람/회사가 여러 번 나와도 1개)
- 관계는 명확하게 언급된 경우만
- 추출할 게 없으면 해당 필드를 빈 배열로
- 금액은 정수 KRW로 정규화 (예: "120만 원" → 1200000)
{corrections_block}
- 개인정보(주민번호/카드번호)는 마스킹하고 risks에 표시
- **오직 JSON만 출력. 마크다운, 설명, 주석 모두 금지**

urgency 분류 기준:
- immediate: 납기가 오늘이거나 이미 지남, 고객 불만/클레임, 미수금/결제 문제, 긴급 AS
- today: 오늘 중 처리 필요하다고 명시된 일, 상대방이 오늘 기다리는 것
- this_week: 이번 주 내 처리, "이번 주에", "빠른 시일 내" 등
- low: 명확한 기한 없음, "나중에", "시간 될 때" 등
{_fewshot_examples}

전사본:
{content}"""


def _load_corrections_block() -> str:
    """Load domain corrections from JSON and format as prompt bullet list.

    Returns the formatted corrections block string, or empty string if
    the JSON file is missing or unreadable.
    """
    corrections_path = Path(__file__).resolve().parent / "domain_corrections.json"
    if not corrections_path.is_file():
        return ""
    try:
        with open(corrections_path, encoding="utf-8") as f:
            rules = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to load domain_corrections.json: %s", exc)
        return ""
    lines = []
    for rule in rules:
        note = f" ({rule['note']})" if rule.get("note") else ""
        lines.append(f"  - \"{rule['from']}\" → \"{rule['to']}\"{note}")
    if not lines:
        return ""
    return "- 전사 교정 규칙 (반드시 적용, corrections에 기록):\n" + "\n".join(lines)


_CORRECTIONS_BLOCK = _load_corrections_block()


def get_prompt() -> str:
    """Langfuse에서 프롬프트를 로드하거나 하드코딩된 기본 프롬프트를 반환한다."""
    try:
        _lf_s = os.environ.get("LANGFUSE_SECRET_KEY", "")
        _lf_p = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        if _lf_s and _lf_p:
            from langfuse import get_client
            lf = get_client()
            try:
                lf_prompt = lf.get_prompt("unified-extraction", label="latest")
                compiled = lf_prompt.compile()
                if compiled and len(compiled) > 100:
                    return compiled
            except (RuntimeError, ValueError) as exc:
                log.debug("Langfuse prompt fetch failed: %s", exc)
        return (
            UNIFIED_EXTRACT_PROMPT
            .replace("{corrections_block}", _CORRECTIONS_BLOCK)
            .replace("{_fewshot_examples}", _FEWSHOT_EXAMPLES)
        )
    except (ImportError, RuntimeError) as exc:
        log.debug("Prompt setup failed: %s", exc)
        return (
            UNIFIED_EXTRACT_PROMPT
            .replace("{corrections_block}", _CORRECTIONS_BLOCK)
            .replace("{_fewshot_examples}", _FEWSHOT_EXAMPLES)
        )


def setup_langfuse() -> bool:
    """환경변수 기반으로 Langfuse 사용 가능 여부를 확인한다."""
    _lf_s = os.environ.get("LANGFUSE_SECRET_KEY", "")
    _lf_p = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    return bool(_lf_s and _lf_p)

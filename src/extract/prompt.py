"""LLM prompt construction for unified extraction.

Provides the hardcoded unified-extraction prompt and optional
Langfuse-based prompt loading.
"""

import logging
import os

log = logging.getLogger(__name__)

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
- 전사 교정 규칙 (반드시 적용, corrections에 기록):
  - "재번 택장" → "채반 대자"
  - "점표" → "전표" (회계 용어)
  - "올프라자" → "오일프라자" (거래처명)
  - "조리배" → "조립" (조립/재발송 문맥에서)
  - "채반 택장" → "채반 대자"
- 개인정보(주민번호/카드번호)는 마스킹하고 risks에 표시
- **오직 JSON만 출력. 마크다운, 설명, 주석 모두 금지**

전사본:
{content}"""


def get_prompt():
    """Load prompt from Langfuse if available, else use hardcoded."""
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
            except Exception as exc:
                log.debug("Langfuse prompt fetch failed: %s", exc)
        return UNIFIED_EXTRACT_PROMPT
    except Exception as exc:
        log.debug("Prompt setup failed: %s", exc)
        return UNIFIED_EXTRACT_PROMPT


def setup_langfuse():
    """Check if Langfuse is available based on environment variables."""
    _lf_s = os.environ.get("LANGFUSE_SECRET_KEY", "")
    _lf_p = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    return bool(_lf_s and _lf_p)

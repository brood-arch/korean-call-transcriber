#!/usr/bin/env python3
"""Integrated LLM extraction from call transcription text.

Replaces the 3 separate extraction scripts (TODO, entity, schedule)
with a single unified prompt that extracts everything in one API call.

Usage:
    python extract_all.py                    # Full incremental run
    python extract_all.py --start-batch 10  # Resume from batch 10
    python extract_all.py --dry-run          # Validate without API calls
    

Config:
    Uses ZAI_API_KEY environment variable for API authentication.
    State: memory/state/integrated_extraction/
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add workspace root to path
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

try:
    from scripts.pipeline_utils import normalize_title, normalize_source, parse_call_context, safe_load_json, safe_save_json, compress_transcript, fallback_summary
    from scripts.persistent_todo_store import merge_todos, todo_key
except ImportError:
    from pipeline_utils import normalize_title, normalize_source, parse_call_context, safe_load_json, safe_save_json, compress_transcript, fallback_summary
    from persistent_todo_store import merge_todos, todo_key

# Lazy import for fast_score_transcript (avoids circular / heavy init at module load)
_fast_score_fn = None
def _get_fast_score():
    global _fast_score_fn
    if _fast_score_fn is None:
        try:
            from scripts.signal_detector import fast_score_transcript
            _fast_score_fn = fast_score_transcript
        except ImportError:
            try:
                from signal_detector import fast_score_transcript
                _fast_score_fn = fast_score_transcript
            except ImportError:
                _fast_score_fn = lambda text: {"score": 1.0, "band": "definite_keep", "should_process": True, "signals": {}, "drop_reason": None}
    return _fast_score_fn

KST = timezone(timedelta(hours=9))

# Pipeline config
try:
    from pipeline_paths import TRANSCRIPT_DIR
    DEFAULT_BASE_DIR = str(TRANSCRIPT_DIR)
except Exception:
    DEFAULT_BASE_DIR = os.environ.get("TRANSCRIPT_DIR", "./data/transcripts")
DEFAULT_STATE_DIR = WORKSPACE / "memory" / "state" / "integrated_extraction"
DEFAULT_BATCH_SIZE = 5            # files per API run (reduced for ZAI rate limit stability)
MAX_CONTENT_CHARS = 12000        # P1-4: GLM ctx 기준 여유 있음
DEFAULT_API_DELAY = 5.0          # seconds between API calls (ZAI rate limit safe)
MAX_RETRIES = 4                  # increased from 3 for better resilience
RETRY_BACKOFF = [5, 15, 45, 90]  # seconds — exponential for 429

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
      "type": "missed_deadline|payment_delay|customer_complaint|stock_shortage|quality_issue|privacy|ambiguous_request|other",
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
  - 상대방이 직접 수행하는 일(상대방이 발송, 상대방이 제조, 상대방이 배송 등)은 TODO에서 제외하거나 owner="상대방"으로 표시
  - 단, 상대방이 우리에게 무언가를 보낸다면 그건 우리 입장에선 "수령 대기"/"입고 확인"이지 "발송"/"배송 준비"가 아님
  - 항상 화자(사용자 vs 상대방)를 기준으로 누가 행동의 주체인지 파악할 것
  - 예시: 상대방이 "내일 제품 3개 보내겠다" → title: "상대방 제품 3개 수령 대기", owner: "me" (우리가 받아야 하므로)
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
        _lf_b = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
        if _lf_s and _lf_p:
            from langfuse import get_client
            lf = get_client()
            try:
                lf_prompt = lf.get_prompt("unified-extraction", label="latest")
                compiled = lf_prompt.compile()
                if compiled and len(compiled) > 100:
                    return compiled
            except Exception:
                pass
        return UNIFIED_EXTRACT_PROMPT
    except Exception:
        return UNIFIED_EXTRACT_PROMPT


# --- Langfuse setup ---
def _setup_langfuse():
    _lf_s = os.environ.get("LANGFUSE_SECRET_KEY", "")
    _lf_p = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    _lf_b = os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    if _lf_s and _lf_p:
        return True
    return False


def call_zai_extract(api_key: str, content: str, run_id: str = "") -> dict | None:
    """Call GLM via ZAI (OpenAI-compatible) with unified extraction prompt.
    
    Returns dict with keys: summary, todos, appointments, entities, products, money, risks, corrections
    Returns None on failure.
    """
    import urllib.request
    import urllib.error
    
    prompt_template = get_prompt()
    prompt = prompt_template.replace("{content}", content[:MAX_CONTENT_CHARS], 1)
    
    payload = json.dumps({
        "model": "glm-5.1",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        # GLM-5.1 defaults to long reasoning_content on the coding endpoint.
        # For deterministic JSON extraction we need fast final content, not thinking traces.
        "thinking": {"type": "disabled"},
    }).encode("utf-8")
    
    # IMPORTANT: use ZAI coding endpoint. The non-coding /api/paas endpoint returns
    # 429 "Insufficient balance or no resource package" for this account, causing
    # TODO extraction to fail silently after retries.
    api_url = os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4").rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    
    # Langfuse span
    _gen = None
    if _lf_available:
        try:
            from langfuse import get_client
            _tracer = get_client()
            _gen = _tracer.start_as_current_observation(
                as_type="generation",
                name="unified-extraction-llm",
                model="zai/glm-5.1",
                input=prompt[:300],
                metadata={"run_id": run_id or ""},
            )
        except Exception:
            pass
    
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(api_url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = parse_unified_response(text)
            parsed["raw_usage"] = result.get("usage", {})
            
            # Langfuse: record result
            if _gen:
                try:
                    _gen.update(
                        output={
                            "summary": bool(parsed.get("summary", {}).get("one_line")),
                            "todos": len(parsed.get("todos", [])),
                            "appointments": len(parsed.get("appointments", [])),
                            "entities": len(parsed.get("entities", [])),
                            "products": len(parsed.get("products", [])),
                            "money": len(parsed.get("money", [])),
                            "risks": len(parsed.get("risks", [])),
                            "corrections": len(parsed.get("corrections", [])),
                        },
                        metadata={
                            "prompt_tokens": parsed.get("raw_usage", {}).get("prompt_tokens", 0),
                            "completion_tokens": parsed.get("raw_usage", {}).get("completion_tokens", 0),
                        }
                    )
                    _gen.end()
                except Exception:
                    pass
            
            return parsed
            
        except urllib.error.HTTPError as e:
            status = e.code
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            if status == 429:
                # Exponential backoff: 30s, 60s, 120s
                wait = 30 * (2 ** attempt)
                print(f"    429 rate limit (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
            elif status >= 500:
                wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
                print(f"    Server error {status}: {body[:100]}, retrying in {wait}s...")
                time.sleep(wait)
            elif status == 401:
                print(f"    Auth error (key invalid?): {body[:100]}")
                break  # No point retrying with bad key
            else:
                wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
                print(f"    HTTP error {status}: {body[:100]}, retrying in {wait}s...")
                time.sleep(wait)
        except urllib.error.URLError as e:
            wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
            print(f"    Network error: {e}, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"    Unexpected error: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
            # Don't break — try remaining attempts

    # Langfuse cleanup on failure
    if _gen:
        try:
            _gen.end()
        except Exception:
            pass
    
    return None


def parse_unified_response(text: str) -> dict:
    """Parse unified JSON response, handling markdown code blocks."""
    cleaned = text.strip()
    
    # Strip markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    
    try:
        data = json.loads(cleaned)
        return {
            "summary": _validate_summary(data.get("summary", {})),
            "todos": _validate_todos(data.get("todos", [])),
            "appointments": _validate_appointments(data.get("appointments", [])),
            "entities": _validate_entities(data.get("entities", [])),
            "products": _validate_products(data.get("products", [])),
            "money": _validate_money(data.get("money", [])),
            "risks": _validate_risks(data.get("risks", [])),
            "corrections": _validate_corrections(data.get("corrections", [])),
            "parse_error": False,
        }
    except json.JSONDecodeError:
        # Try to extract JSON from text
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end])
                return {
                    "summary": _validate_summary(data.get("summary", {})),
                    "todos": _validate_todos(data.get("todos", [])),
                    "appointments": _validate_appointments(data.get("appointments", [])),
                    "entities": _validate_entities(data.get("entities", [])),
                    "products": _validate_products(data.get("products", [])),
                    "money": _validate_money(data.get("money", [])),
                    "risks": _validate_risks(data.get("risks", [])),
                    "corrections": _validate_corrections(data.get("corrections", [])),
                    "parse_error": False,
                }
            except json.JSONDecodeError:
                pass
        return {
            "summary": {}, "todos": [], "appointments": [], "entities": [],
            "products": [], "money": [], "risks": [], "corrections": [],
            "parse_error": True, "raw": cleaned[:500]
        }


def _validate_summary(item: dict) -> dict:
    if not isinstance(item, dict):
        return {}
    return {
        "one_line": str(item.get("one_line", ""))[:300],
        "details": [str(d)[:200] for d in item.get("details", []) if d][:5],
        "call_type": str(item.get("call_type", "unknown")),
        "overall_confidence": float(item.get("overall_confidence", 0.0)),
    }


def _validate_todos(items: list) -> list:
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("title"):
            result.append({
                "id_hint": str(item.get("id_hint", ""))[:50],
                "title": str(item["title"]).strip(),
                "owner": item.get("owner", "unknown") if item.get("owner") in ("me", "partner", "unknown") else "unknown",
                "priority": item.get("priority", "medium") if item.get("priority") in ("high", "medium", "low") else "medium",
                "status": item.get("status", "new") if item.get("status") in ("new", "in_progress", "waiting", "done", "cancelled") else "new",
                "due_date": str(item.get("due_date")) if item.get("due_date") else None,
                "due_time": str(item.get("due_time")) if item.get("due_time") else None,
                "context": str(item.get("context", ""))[:200],
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_appointments(items: list) -> list:
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("title"):
            result.append({
                "title": str(item["title"]).strip(),
                "date": str(item.get("date")) if item.get("date") else None,
                "time": str(item.get("time")) if item.get("time") else None,
                "timezone": str(item.get("timezone", "Asia/Seoul")),
                "location": str(item.get("location")) if item.get("location") else None,
                "participants": [str(p) for p in item.get("participants", []) if p][:10],
                "description": str(item.get("description", ""))[:200],
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_entities(items: list) -> list:
    result = []
    valid_types = {"Person", "Organization", "Location", "PhoneNumber", "Product", "Project", "Event", "Contract", "Other"}
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            etype = item.get("type", "")
            if etype not in valid_types:
                etype = "Other"
            result.append({
                "name": str(item["name"]).strip(),
                "type": etype,
                "canonical_name": str(item.get("canonical_name")) if item.get("canonical_name") else None,
                "role": item.get("role", "unknown") if item.get("role") in ("customer", "supplier", "employee", "carrier", "unknown") else "unknown",
                "attributes": item.get("attributes", {}) if isinstance(item.get("attributes"), dict) else {},
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_products(items: list) -> list:
    result = []
    valid_cats = {"환풍기", "송풍기", "케이스", "날개", "채반", "부품", "기타", "unknown"}
    valid_actions = {"quote", "order", "deliver", "repair", "check_stock", "manufacture", "unknown"}
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            qty = item.get("quantity", {})
            if not isinstance(qty, dict):
                qty = {}
            result.append({
                "name": str(item["name"]).strip(),
                "canonical_name": str(item.get("canonical_name")) if item.get("canonical_name") else None,
                "category": item.get("category", "unknown") if item.get("category") in valid_cats else "unknown",
                "spec": str(item.get("spec")) if item.get("spec") else None,
                "quantity": {
                    "value": int(qty.get("value", 0)),
                    "unit": str(qty.get("unit", "unknown")),
                },
                "action": item.get("action", "unknown") if item.get("action") in valid_actions else "unknown",
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_money(items: list) -> list:
    result = []
    valid_kinds = {"price", "deposit", "balance", "shipping", "discount", "tax", "unknown"}
    valid_statuses = {"paid", "unpaid", "partial", "unknown"}
    for item in items:
        if isinstance(item, dict):
            try:
                amount = int(item.get("amount", 0))
            except (ValueError, TypeError):
                amount = 0
            result.append({
                "amount": amount,
                "currency": str(item.get("currency", "KRW")),
                "kind": item.get("kind", "unknown") if item.get("kind") in valid_kinds else "unknown",
                "related_to": str(item.get("related_to")) if item.get("related_to") else None,
                "payment_status": item.get("payment_status", "unknown") if item.get("payment_status") in valid_statuses else "unknown",
                "due_date": str(item.get("due_date")) if item.get("due_date") else None,
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_risks(items: list) -> list:
    result = []
    valid_severities = {"high", "medium", "low"}
    valid_types = {"missed_deadline", "payment_delay", "customer_complaint", "stock_shortage", "quality_issue", "privacy", "ambiguous_request", "other"}
    for item in items:
        if isinstance(item, dict) and item.get("description"):
            result.append({
                "severity": item.get("severity", "medium") if item.get("severity") in valid_severities else "medium",
                "type": item.get("type", "other") if item.get("type") in valid_types else "other",
                "description": str(item["description"])[:300],
                "recommended_action": str(item.get("recommended_action")) if item.get("recommended_action") else None,
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_corrections(items: list) -> list:
    result = []
    valid_reasons = {"exact_rule", "alias", "contextual", "spacing", "number_normalization", "other"}
    for item in items:
        if isinstance(item, dict) and item.get("original") and item.get("corrected"):
            result.append({
                "original": str(item["original"])[:200],
                "corrected": str(item["corrected"])[:200],
                "rule_id": str(item.get("rule_id")) if item.get("rule_id") else None,
                "reason": item.get("reason", "other") if item.get("reason") in valid_reasons else "other",
                "source_quote": str(item.get("source_quote", ""))[:300],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _compute_file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# --- Pipeline orchestration ---

class IntegratedPipeline:
    """Unified extraction pipeline: TODO + Entity + Schedule in one LLM call."""
    
    def __init__(self, args):
        self.base_dir = Path(args.base_dir)
        self.state_dir = Path(args.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = args.batch_size
        self.api_delay = args.api_delay
        self.today_only = getattr(args, 'today', False)
        self.start_batch_override = getattr(args, 'start_batch', 0)  # P2-C8
        self.run_id = f"run_{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        # P0-2: --today 모드와 전체 모드 체크포인트 분리
        if self.today_only:
            self.checkpoint_file = self.state_dir / "checkpoint_today.json"
        else:
            self.checkpoint_file = self.state_dir / "checkpoint.json"
        self.processed_index_file = self.state_dir / "processed_files.json"
        self.stats = {"summary": 0, "todos": 0, "appointments": 0, "entities": 0, "products": 0, "money": 0, "risks": 0, "corrections": 0, "batches_done": 0, "errors": 0}
        self.notification_state_file = self.state_dir / "notification_state.json"
        self._last_new_todos = []
        self._last_notifications = []
        self._telegram_notified_new_todos = False
        
    def load_processed_index(self) -> dict:
        """Load {filename: file_hash} of already processed files."""
        if self.processed_index_file.exists():
            try:
                return json.loads(self.processed_index_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def save_processed_index(self, index: dict):
        """Save processed files index."""
        try:
            from safe_io import safe_write_json
            safe_write_json(self.processed_index_file, index, origin="integrated_pipeline")
        except ImportError:
            tmp = self.processed_index_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.processed_index_file)

    # --- persistent_todos.json path ---
    @property
    def persistent_todo_file(self) -> Path:
        return WORKSPACE / "memory" / "state" / "persistent_todos.json"

    def _load_persistent_todos(self) -> dict:
        """Load persistent_todos.json, creating if missing."""
        return safe_load_json(self.persistent_todo_file, default={"todos": {}}) or {"todos": {}}

    def _save_persistent_todos(self, data: dict):
        safe_save_json(self.persistent_todo_file, data)

    def sync_todos_to_persistent(self, batch_ok_results: list[dict]) -> int:
        """Sync new TODOs from extraction results to persistent_todos.json.
        
        Only syncs owner="me" tasks.
        Returns count of newly added TODOs.
        """
        MY_OWNERS = {"me"}
        self._last_new_todos = []
        if not batch_ok_results:
            return 0
        
        store = self._load_persistent_todos()
        before = json.dumps(store, ensure_ascii=False, sort_keys=True, default=str)
        candidates = []
        
        for result in batch_ok_results:
            stem = result.get("file", "")
            stem_base = normalize_source(stem)
            ctx = parse_call_context(stem_base)
            for todo in result.get("todos", []):
                owner = todo.get("owner", "me")
                if owner not in MY_OWNERS:
                    continue
                title = todo.get("title", "").strip()
                if not title:
                    continue
                todo_entry = {
                    "title": title,
                    "owner": owner,
                    "source": stem_base,
                    "counterparty": ctx["caller"],
                    "phone": ctx["phone"],
                    "called_at": ctx["called_at"],
                    "priority": todo.get("priority", "medium"),
                    "status": "active",
                    "details": todo.get("context", ""),
                    "due_date": todo.get("due_date") or None,
                    "created_at": datetime.now(KST).isoformat(),
                    "run_id": self.run_id,
                }
                candidates.append(todo_entry)
        
        new_todos = merge_todos(store, candidates)
        after = json.dumps(store, ensure_ascii=False, sort_keys=True, default=str)
        if after != before:
            self._save_persistent_todos(store)
        
        self._last_new_todos = new_todos
        return len(new_todos)

    @staticmethod
    def _appointment_key(appt: dict) -> str:
        source = normalize_source(appt.get("source", ""))
        return f"{appt.get('title','').strip()}|{appt.get('date')}|{appt.get('time')}|{source}"

    def _load_notification_state(self) -> dict:
        """Load extraction notification state and ensure notification buckets exist."""
        state = {}
        if self.notification_state_file.exists():
            try:
                state = json.loads(self.notification_state_file.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        state.setdefault("notified_todos", {})
        state.setdefault("notified_appointments", {})
        state.setdefault("calendar_drafts", {})
        state.setdefault("last_summary", {})
        return state

    def _save_notification_state(self, state: dict):
        self.notification_state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.notification_state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.notification_state_file)

    def _send_telegram(self, text: str):
        """Send a Telegram message using environment variables + curl subprocess."""
        import subprocess
        try:
            token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
            if not token:
                raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

            url = f"https://api.telegram.org/bot{token}/sendMessage"
            return subprocess.run(
                [
                    "curl", "-sS", "-X", "POST", url,
                    "-d", f"chat_id={chat_id}",
                    "--data-urlencode", f"text={text}",
                    "-d", "disable_web_page_preview=true",
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
            )
        except Exception as e:
            class _Result:
                returncode = 1
                stdout = ""
                stderr = str(e)
            return _Result()

    def _notify_new_items(self, new_todos: list, new_appointments: list | None = None, active_todos: list | None = None) -> list:
        """Notify about new TODOs/appointments; include active TODO backlog with new TODO alerts."""
        def _fmt_phone(p):
            p = str(p or "")
            if not p or len(p) < 10:
                return p
            return f"{p[:3]}-{p[3:7]}-{p[7:]}"

        def _fmt_date(d):
            if not d:
                return ""
            try:
                dt = datetime.fromisoformat(str(d))
                return f"{dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d}"
            except Exception:
                return str(d)[:16]

        new_appointments = new_appointments or []
        active_todos = active_todos or []
        notes = []
        msg_lines = []

        if new_todos:
            msg_lines.append("🆕 새 TODO")
            for t in new_todos[:8]:
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get("priority"), "")
                title = t.get("title", "")
                cp = t.get("counterparty") or ""
                phone = t.get("phone") or ""
                called = t.get("called_at") or ""
                src_parts = []
                if cp and cp != "알 수 없음":
                    src_parts.append(f"{cp} ({_fmt_phone(phone)})" if phone else cp)
                if called:
                    src_parts.append(_fmt_date(called))
                src = " / ".join(src_parts)
                msg_lines.append(f"  {priority_icon} {title}")
                if src:
                    msg_lines.append(f"    • 통화: {src}")

        if new_todos and active_todos:
            new_titles = {normalize_title(t.get("title", "")) for t in new_todos}
            existing = [t for t in active_todos if normalize_title(t.get("title", "")) not in new_titles]
            if existing:
                msg_lines.append("")
                msg_lines.append(f"📋 미완료 TODO ({len(existing)}건)")
                for t in existing[:10]:
                    priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get("priority"), "")
                    title = t.get("title", "")
                    cp = t.get("counterparty") or ""
                    phone = t.get("phone") or ""
                    called = t.get("called_at") or ""
                    src_parts = []
                    if cp and cp != "알 수 없음":
                        src_parts.append(f"{cp} ({_fmt_phone(phone)})" if phone else cp)
                    if called:
                        src_parts.append(_fmt_date(called))
                    src = " / ".join(src_parts)
                    msg_lines.append(f"  {priority_icon} {title}")
                    if src:
                        msg_lines.append(f"    • 통화: {src}")
                if len(existing) > 10:
                    msg_lines.append(f"  ... 외 {len(existing) - 10}건")

        if new_appointments:
            msg_lines.append("")
            msg_lines.append("📅 새 스케줄")
            for a in new_appointments[:5]:
                date_str = a.get("date") or "미정"
                time_str = a.get("time") or ""
                cp = a.get("counterparty") or ""
                src = f" / {cp}" if cp and cp != "알 수 없음" else ""
                msg_lines.append(f"  - {a.get('title')} / {date_str} {time_str}{src}")

        if not msg_lines:
            return []

        result = self._send_telegram("\n".join(msg_lines))
        ok = result.returncode == 0
        notes.append({"kind": "new_items", "ok": ok})
        if ok and new_todos:
            self._telegram_notified_new_todos = True
        if not ok:
            print(f"    WARN: telegram notification failed: {getattr(result, 'stderr', '') or getattr(result, 'stdout', '')}")
        return notes

    def _track_notified(self, new_todos: list, new_appointments: list, notifications: list | None = None):
        """Persist notified_todos/notified_appointments in shared state."""
        try:
            state = self._load_notification_state()
            now_iso = datetime.now(KST).isoformat()

            for a in new_appointments or []:
                source = normalize_source(a.get("source", ""))
                appt = {**a, "source": source}
                state.setdefault("notified_appointments", {})[self._appointment_key(appt)] = {
                    "title": appt.get("title"),
                    "date": appt.get("date"),
                    "updated_at": now_iso,
                }

            for t in new_todos or []:
                source = normalize_source(t.get("source", ""))
                todo = {**t, "source": source}
                state.setdefault("notified_todos", {})[todo_key(todo)] = {
                    "title": todo.get("title"),
                    "source": source,
                    "updated_at": now_iso,
                }

            state["last_run"] = now_iso
            state["last_summary"] = {
                **state.get("last_summary", {}),
                "new_todos": len(new_todos or []),
                "new_appointments": len(new_appointments or []),
                "notifications": notifications or [],
                "source": "extract_all",
                "run_id": self.run_id,
            }
            self._save_notification_state(state)
        except Exception as e:
            print(f"    WARN: notification state tracking failed: {e}")

    def _collect_new_appointments(self, batch_results: list[dict], state: dict) -> list:
        prev_notified = set(state.get("notified_appointments", {}).keys())
        new_appointments = []
        for result in batch_results:
            if result.get("status") != "ok":
                continue
            stem = normalize_source(result.get("file", ""))
            ctx = parse_call_context(stem)
            for appt in result.get("appointments", []):
                appt_entry = {**appt, "source": stem, "counterparty": ctx.get("caller", ""), "phone": ctx.get("phone", ""), "called_at": ctx.get("called_at", "")}
                if self._appointment_key(appt_entry) not in prev_notified:
                    new_appointments.append(appt_entry)
        return new_appointments

    def print_todo_alert(self):
        """Print full active TODO report for immediate notification."""
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, str(WORKSPACE / "scripts" / "todo_report.py"), "--status", "active"],
                capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE),
            )
            if result.stdout:
                print(f"\n{'='*50}")
                print("🚨 신규 TODO 발생 — 전체 활성 할 일:")
                print(f"{'='*50}")
                print(result.stdout)
        except Exception as e:
            print(f"    WARN: todo alert failed: {e}")

    def get_transcription_files(self) -> list:
        """Get files to process. If --today, only today's files. Skips already processed."""
        all_files = sorted(self.base_dir.glob("*.txt"))
        all_files = [f for f in all_files if f.stat().st_size > 50]

        # Filter by today if requested
        if self.today_only:
            today_str = datetime.now(KST).strftime("%Y%m%d")
            all_files = [f for f in all_files if today_str in f.stem]
            print(f"--today mode: filtering for date {today_str}")

        # Skip already processed files (same hash = unchanged)
        processed = self.load_processed_index()
        new_files = []
        for f in all_files:
            fhash = _compute_file_hash(f)
            if processed.get(f.stem) != fhash:
                new_files.append(f)

        skipped = len(all_files) - len(new_files)
        if skipped > 0:
            print(f"Skipped {skipped} already-processed (unchanged) files")

        return new_files
    
    def load_checkpoint(self) -> int:
        if self.checkpoint_file.exists():
            try:
                cp = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
                # --today 모드에서는 날짜가 바뀌면 체크포인트 초기화
                if self.today_only:
                    today_str = datetime.now(KST).strftime("%Y%m%d")
                    cp_date = cp.get("last_updated", "")[:10].replace("-", "")
                    if cp_date != today_str:
                        print(f"Checkpoint stale (from {cp.get('last_updated', '')}), resetting for today")
                        return 0
                return cp.get("last_completed_batch", -1) + 1
            except Exception:
                pass
        return 0
    
    def save_checkpoint(self, batch_idx: int, total: int, stats: dict):
        data = {
            "last_completed_batch": batch_idx,
            "total_batches": total,
            "last_updated": datetime.now(KST).isoformat(),
            "run_id": self.run_id,
            "stats": stats,
        }
        try:
            from safe_io import safe_write_json
            safe_write_json(self.checkpoint_file, data, origin="integrated_pipeline")
        except ImportError:
            tmp = self.checkpoint_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.checkpoint_file)
    
    def save_batch_result(self, batch_idx: int, batch_start: int, batch_files: list, results: list, errors: list, status: str):
        output = {
            "batch_index": batch_idx,
            "run_id": self.run_id,
            "timestamp": datetime.now(KST).isoformat(),
            "files": [f.stem for f in batch_files],
            "results": results,
            "errors": errors,
            "status": status,
        }
        batch_file = self.state_dir / f"batch_{batch_idx:04d}.json"
        try:
            from safe_io import safe_write_json
            safe_write_json(batch_file, output, origin="integrated_pipeline")
        except ImportError:
            batch_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    
    def run(self):
        api_key = os.environ.get("ZAI_API_KEY", "")
        if not api_key:
            print("ERROR: ZAI_API_KEY environment variable not set.")
            sys.exit(1)
        
        files = self.get_transcription_files()
        total_files = len(files)
        print(f"Found {total_files} transcription files")
        if total_files == 0:
            print("No files. Exiting.")
            return
        
        total_batches = (total_files + self.batch_size - 1) // self.batch_size
        # --today 모드에서는 파일 목록이 매번 동적으로 변하므로
        # 체크포인트의 batch 인덱스를 신뢰할 수 없음 → 항상 0부터 시작
        # (이미 처리된 파일은 get_transcription_files()에서 해시로 걸러짐)
        if self.today_only:
            start_batch = self.start_batch_override
            print(f"--today mode: always starting from batch {start_batch} (checkpoint ignored for batch index)")
        else:
            # P2-C8: --start-batch 인자가 있으면 체크포인트보다 우선
            start_batch = max(self.load_checkpoint(), self.start_batch_override)
        print(f"Total batches: {total_batches}, starting from: {start_batch}")
        print(f"Run ID: {self.run_id}")
        
        try:
            for batch_idx in range(start_batch, total_batches):
                batch_start_idx = batch_idx * self.batch_size
                batch_end_idx = min(batch_start_idx + self.batch_size, total_files)
                batch_files = files[batch_start_idx:batch_end_idx]
                
                # Skip already completed batches (NOT in --today mode: daily files change daily)
                if not self.today_only:
                    batch_file = self.state_dir / f"batch_{batch_idx:04d}.json"
                    if batch_file.exists():
                        try:
                            existing = json.loads(batch_file.read_text(encoding="utf-8"))
                            if existing.get("status") == "done":
                                print(f"  [{batch_idx:04d}] SKIP (already done)")
                                continue
                        except Exception:
                            pass
                
                print(f"  [{batch_idx:04d}/{total_batches-1}] Processing {len(batch_files)} files...")
                
                batch_results = []
                batch_errors = []
                batch_ok_stems = []   # stems successfully processed in this batch
                
                # Load processed index for updating
                processed = self.load_processed_index()

                for file_path in batch_files:
                    stem = file_path.stem
                    try:
                        content = file_path.read_text(encoding="utf-8").strip()
                        if not content:
                            batch_errors.append({"file": stem, "error": "empty"})
                            self.stats["errors"] += 1
                            continue
                        
                        # ── Phase 1: fast_score pre-filter (OpenHuman 3-band) ──
                        fast_score = _get_fast_score()(content)
                        if not fast_score.get("should_process", True):
                            self.stats["fast_score_dropped"] = self.stats.get("fast_score_dropped", 0) + 1
                            # Still mark as processed so we don't re-evaluate it
                            fhash = _compute_file_hash(file_path)
                            processed[stem] = fhash
                            batch_ok_stems.append(stem)
                            batch_results.append({
                                "file": stem, "source": str(file_path),
                                "file_hash": fhash, "status": "skipped_fast_score",
                                "fast_score": fast_score,
                            })
                            continue
                        
                        # ── Phase 2: compress (OpenHuman TokenJuice adapted) ──
                        content = compress_transcript(content, budget=MAX_CONTENT_CHARS)
                        
                        result = call_zai_extract(api_key, content, run_id=self.run_id)
                        
                        # ── Phase 3: fallback if LLM failed (OpenHuman fallback_summary) ──
                        if not result or result.get("parse_error"):
                            fallback = fallback_summary(content, source=stem)
                            fallback["fast_score"] = fast_score
                            fhash = _compute_file_hash(file_path)
                            file_result = {
                                "file": stem,
                                "source": str(file_path),
                                "file_hash": fhash,
                                "status": "fallback",
                                **fallback,
                            }
                            processed[stem] = fhash
                            batch_ok_stems.append(stem)
                            self.stats["fallbacks"] = self.stats.get("fallbacks", 0) + 1
                            batch_results.append(file_result)
                            continue
                        
                        if not result.get("parse_error"):
                            fhash = _compute_file_hash(file_path)
                            file_result = {
                                "file": stem,
                                "source": str(file_path),
                                "file_hash": fhash,
                                "status": "ok",
                                **result,
                            }
                            processed[stem] = fhash  # Mark as processed
                            batch_ok_stems.append(stem)
                            if result.get("summary", {}).get("one_line"):
                                self.stats["summary"] += 1
                            self.stats["todos"] += len(result.get("todos", []))
                            self.stats["appointments"] += len(result.get("appointments", []))
                            self.stats["entities"] += len(result.get("entities", []))
                            self.stats["products"] += len(result.get("products", []))
                            self.stats["money"] += len(result.get("money", []))
                            self.stats["risks"] += len(result.get("risks", []))
                            self.stats["corrections"] += len(result.get("corrections", []))
                        else:
                            file_result = {"file": stem, "status": "failed", "error": "api_or_parse_failed"}
                            batch_errors.append({"file": stem, "error": "api_or_parse_failed"})
                            self.stats["errors"] += 1
                        
                        batch_results.append(file_result)
                        
                    except Exception as e:
                        print(f"    ERROR {stem}: {e}")
                        batch_errors.append({"file": stem, "error": str(e)})
                        self.stats["errors"] += 1
                    
                    time.sleep(self.api_delay)
                
                # P0-1: 항상 성공한 파일의 processed_index 저장
                self.save_processed_index(processed)

                # TODO sync: 신규 TODO를 persistent_todos.json에 반영
                new_todo_count = self.sync_todos_to_persistent(batch_results)
                self.stats["new_todos"] = self.stats.get("new_todos", 0) + new_todo_count

                # Telegram notification + shared notified_* state tracking.
                # Do this immediately after the persistent TODO sync so alerts reflect
                # both newly added TODOs and the current active backlog.
                notification_state = self._load_notification_state()
                new_todos = list(self._last_new_todos)
                persistent_data = self._load_persistent_todos()
                active_todos = [
                    t for t in persistent_data.get("todos", {}).values()
                    if isinstance(t, dict) and t.get("status", "active") == "active"
                ]
                new_appointments = self._collect_new_appointments(batch_results, notification_state)
                notifications = self._notify_new_items(
                    new_todos,
                    new_appointments=new_appointments,
                    active_todos=active_todos,
                )
                if notifications or new_todos or new_appointments:
                    self._last_notifications.extend(notifications)
                    self._track_notified(new_todos, new_appointments, notifications)

                # Save batch result. Failed batches stay partial so a later run retries them.
                if batch_errors:
                    self.save_batch_result(batch_idx, batch_start_idx, batch_files, batch_results, batch_errors, "partial")
                else:
                    self.save_batch_result(batch_idx, batch_start_idx, batch_files, batch_results, batch_errors, "done")
                    self.stats["batches_done"] += 1
                    self.save_checkpoint(batch_idx, total_batches, self.stats)
                
                print(f"    Done: {self.stats['todos']} todos, {self.stats['entities']} entities, {self.stats['products']} products, {self.stats['money']} money, {self.stats['risks']} risks, {self.stats['corrections']} corrections"
                      f"| {len(batch_errors)} errors")
                
                if batch_idx < total_batches - 1:
                    time.sleep(self.api_delay)
        
        except KeyboardInterrupt:
            print(f"\nInterrupted at batch {batch_idx}. Run ID: {self.run_id}")
            print(f"Resume with: python extract_all.py --start-batch {batch_idx}")
            self.save_checkpoint(batch_idx, total_batches, self.stats)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"INTEGRATED EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Run ID:       {self.run_id}")
        print(f"Batches:      {self.stats['batches_done']}/{total_batches}")
        print(f"Summaries:    {self.stats['summary']}")
        print(f"TODOs:        {self.stats['todos']}")
        print(f"Appointments: {self.stats['appointments']}")
        print(f"Entities:     {self.stats['entities']}")
        print(f"Products:     {self.stats['products']}")
        print(f"Money:        {self.stats['money']}")
        print(f"Risks:        {self.stats['risks']}")
        print(f"Corrections:  {self.stats['corrections']}")
        print(f"Errors:       {self.stats['errors']}")
        print(f"Dropped:      {self.stats.get('fast_score_dropped', 0)} (fast_score pre-filter)")
        print(f"Fallbacks:    {self.stats.get('fallbacks', 0)} (LLM failed, heuristic)")
        print(f"New TODOs:    {self.stats.get('new_todos', 0)} (synced to persistent_todos.json)")
        if self._last_notifications:
            print(f"Notifications:{len(self._last_notifications)} attempted")
        print(f"~60-70% fewer LLM calls vs. separate scripts")
        print(f"{'='*60}")
        
        # TODO alert fallback: print full active report only if Telegram did not
        # successfully deliver a new-TODO notification.
        if self.stats.get("new_todos", 0) > 0 and not self._telegram_notified_new_todos:
            self.print_todo_alert()

        sys.exit(1 if self.stats["errors"] > 0 else 0)


def dry_run(args):
    """Validate pipeline setup."""
    base_dir = Path(args.base_dir)
    state_dir = Path(args.state_dir)
    
    print("=" * 60)
    print("DRY RUN — Integrated Extraction Pipeline")
    print("=" * 60)
    
    if not base_dir.exists():
        print(f"FAIL: Base directory not found: {base_dir}")
        return False
    print(f"OK: Base directory: {base_dir}")
    
    files = sorted(base_dir.glob("*.txt"))
    files = [f for f in files if f.stat().st_size > 50]
    print(f"OK: Found {len(files)} transcription files")
    
    total_batches = (len(files) + args.batch_size - 1) // args.batch_size
    print(f"OK: {total_batches} batches (size={args.batch_size})")
    
    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"OK: State directory: {state_dir}")
    
    # Check API key
    try:
        key = os.environ.get("ZAI_API_KEY", "")
        print(f"{'OK' if key else 'FAIL'}: ZAI_API_KEY {'set' if key else 'NOT set'}")
    except Exception as e:
        print(f"WARN: Could not check API key: {e}")
    
    completed = len(list(state_dir.glob("batch_*.json")))
    print(f"INFO: {completed} batches already completed")
    
    if files:
        sample = files[0]
        content = sample.read_text(encoding="utf-8")
        print(f"\nSample: {sample.name} ({len(content)} chars)")
        print(f"  Preview: {content[:80].strip()}...")
    
    print(f"\nDry run PASSED.")
    print(f"Command: python extract_all.py")
    return True


# --- Langfuse global ---
_lf_available = False


def main():
    global _lf_available
    
    parser = argparse.ArgumentParser(description="Integrated LLM Extraction Pipeline (TODO + Entity + Schedule)")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--api-delay", type=float, default=DEFAULT_API_DELAY)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--today", action="store_true", help="Only process today's files")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    _lf_available = _setup_langfuse()
    
    if args.dry_run:
        dry_run(args)
    else:
        pipeline = IntegratedPipeline(args)
        pipeline.run()


if __name__ == "__main__":
    main()

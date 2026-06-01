#!/usr/bin/env python3
"""
signal_detector.py — GBrain signal-detector 패턴의 Python 구현

모든 인바운드 메시지에서:
1. Original thinking (아이디어, 관찰, 가설, 프레임워크)
2. Entity mentions (사람, 회사, 장소, 제품)

를 자동 추출하여 events.jsonl에 기록 + 엔티티 강화 트리거.

Usage:
    from signal_detector import detect_signals
    result = detect_signals(text, source="user_message", timestamp="2026-05-20T17:00:00+09:00")

Fast score (pre-filter before LLM):
    from signal_detector import fast_score_transcript
    score = fast_score_transcript("주문 500개 확인해주세요")

CLI:
    python signal_detector.py --score "text to score"
"""

import json
import re
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 경로 설정
WORKSPACE = Path(__file__).resolve().parent.parent
EVENTS_PATH = WORKSPACE / "memory" / "shared_state" / "events.jsonl"
ENTITIES_DIR = Path(os.environ.get("ENTITIES_DIR", str(WORKSPACE / "data" / "entities")))

KST = timezone(timedelta(hours=9))

# ── 3-band fast_score constants (from OpenHuman score/signals) ──────────────

SIGNAL_WEIGHTS = {
    "token_count": 1.0,       # Text length (longer = more substance)
    "unique_words": 1.0,      # Vocabulary diversity
    "entity_density": 1.5,    # Named entities per token (phones, amounts, products)
    "keyword_signal": 2.0,    # Business keywords (주문, 견적, 납기, 결제, etc.)
    "interaction": 3.0,       # Strongest signal - direct engagement markers
}

DEFINITE_DROP = 0.15   # ≤ 0.15 → skip entirely (greetings, simple confirmations)
DROP_THRESHOLD = 0.30  # Final admission gate (reserved for future use)
DEFINITE_KEEP = 0.85   # ≥ 0.85 → definitely process (orders, quotes, payments)

# Business keywords (from OpenHuman analysis of Korean call transcripts)
BUSINESS_KEYWORDS = [
    "주문", "견적", "발주", "납기", "결제", "입금", "출고", "배송",
    "수령", "방문", "미팅", "약속", "연락", "확인", "처리", "준비",
    "발송", "회신", "요청", "교정",
]

# Product names for entity density
PRODUCT_NAMES = [
    "송풍기", "환풍기", "채반", "블로어", "모터", "케이스", "날개",
]

# Interaction / engagement markers
INTERACTION_PATTERNS = [
    # Specific questions
    r"언제", r"몇시", r"어디", r"어떻게", r"몇 개", r"얼마",
    # Imperative verbs (해줘/부탁드려 계열)
    r"해줘", r"부탁드려", r"보내줘", r"확인해", r"알려줘", r"전달해",
    # Explicit commitments
    r"하겠습니다", r"보내드리겠", r"처리하겠", r"진행하겠",
]

# Amount pattern: Korean-style amounts (500만원, 55,000원, 1000원 etc.)
AMOUNT_PATTERN = r"\d+[,\d]*\s*(?:만\s*)?원"

# Phone pattern: 010-XXXX-XXXX, 011-XXX-XXXX etc.
PHONE_PATTERN = r"01[016789]-\d{3,4}-\d{4}"


def _load_entity_index():
    """기존 거래처/엔티티 인덱스 로드"""
    known = set()
    if ENTITIES_DIR.exists():
        for f in ENTITIES_DIR.iterdir():
            if f.suffix == ".md":
                known.add(f.stem.lower())
    return known


def _append_event(event: dict):
    """events.jsonl에 이벤트 추가"""
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _detect_ideas(text: str) -> list[dict]:
    """사용자의 원본 사고(아이디어/관찰/가설) 감지"""
    ideas = []

    # 아이디어 패턴 감지
    idea_patterns = [
        r"(?:(?:아이디어|생각|아이디어|가설|프레임워크|아이디어인데)[은는이가]\s*:?\s*)(.+?)(?:\.|$)",
        r"(?:새로운\s+(?:접근|방법|아이디어|생각))\s*[은는]?\s*(.+?)(?:\.|$)",
        r"(?:이렇게\s+하면|이렇게\s+되면|만약\s+.+?면)\s+(.+?)(?:\.|$)",
    ]

    for pattern in idea_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            idea_text = m.group(1).strip()
            if len(idea_text) > 5:
                ideas.append({
                    "type": "idea",
                    "content": idea_text,
                    "original_phrase": m.group(0).strip(),
                    "phase": "original_thinking"
                })

    return ideas


def _detect_entities(text: str, known_entities: set) -> list[dict]:
    """텍스트에서 엔티티(사람, 회사, 장소) 감지"""
    entities = []

    # 고정 인물 명단 (오탐지 방지: 성씨만으로 매칭 금지)
    known_persons_env = os.environ.get("KNOWN_PERSONS", "")
    known_persons = [p.strip() for p in known_persons_env.split(",") if p.strip()]

    # 회사/조직 패턴 (고정 리스트 + 일반 패턴)
    org_exact_env = os.environ.get("KNOWN_ORGS", "")
    org_exact = [o.strip() for o in org_exact_env.split(",") if o.strip()]
    org_pattern = r"(?:[A-Z][a-zA-Z]+(?:코리아|케이스|산업|기업)?|(?:주식회사|㈜)\s*[\w]+)"

    # 제품 패턴
    product_pattern = r"(?:환풍기|송풍기|케이스|날개|채반|블로어|팬|모터)"

    # 1) 고정 인물
    for person in known_persons:
        if person in text:
            entities.append({
                "type": "entity",
                "entity_type": "person",
                "name": person,
                "known": person.lower().replace(" ", "") in known_entities,
                "action": "update"
            })

    # 2) 고정 회사명
    for org in org_exact:
        if org in text:
            entities.append({
                "type": "entity",
                "entity_type": "company",
                "name": org,
                "known": org.lower().replace(" ", "") in known_entities,
                "action": "enrich" if org.lower().replace(" ", "") not in known_entities else "update"
            })

    # 3) 패턴 기반 (조직, 제품만)
    for pattern, etype in [
        (org_pattern, "company"),
        (product_pattern, "product"),
    ]:
        for m in re.finditer(pattern, text):
            name = m.group(0).strip()
            if len(name) >= 2:
                is_known = name.lower().replace(" ", "") in known_entities
                entities.append({
                    "type": "entity",
                    "entity_type": etype,
                    "name": name,
                    "known": is_known,
                    "action": "enrich" if not is_known else "update"
                })

    # 중복 제거
    seen = set()
    unique = []
    for e in entities:
        key = (e["entity_type"], e["name"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


def fast_score_transcript(text: str) -> dict:
    """
    3-band fast scoring for call transcripts.
    A cheap, deterministic pre-filter that runs BEFORE any LLM API call.
    Returns a score dict with band classification and per-signal breakdown.

    Bands:
        definite_keep  (score >= 0.85) → definitely process (orders, quotes, payments)
        borderline     (0.15 < score < 0.85) → admit for further processing
        definite_drop  (score <= 0.15) → skip entirely (greetings, simple confirmations)

    Args:
        text: The transcript text to score.

    Returns:
        dict with keys: score, band, should_process, signals, drop_reason
    """
    tokens = text.strip().split()
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))

    # ── Signal 1: token_count ──────────────────────────────────────────
    # Normalize to ~200 tokens max
    token_count_signal = min(total_tokens / 200, 1.0)

    # ── Signal 2: unique_words ─────────────────────────────────────────
    # Vocabulary diversity as a fraction, clamped to [0, 1]
    unique_words_signal = unique_tokens / max(total_tokens, 1)

    # ── Signal 3: entity_density ───────────────────────────────────────
    # Count phone numbers, amounts, and product names
    phone_matches = len(re.findall(PHONE_PATTERN, text))
    amount_matches = len(re.findall(AMOUNT_PATTERN, text))
    product_matches = 0
    for prod in PRODUCT_NAMES:
        product_matches += len(re.findall(re.escape(prod), text))

    entity_count = phone_matches + amount_matches + product_matches
    # Normalize: entities per token, target ~1 entity per 100 tokens = 0.01
    entity_density_signal = min(entity_count / max(total_tokens, 1) / 0.01, 1.0)
    if total_tokens == 0:
        entity_density_signal = 0.0

    # ── Signal 4: keyword_signal ───────────────────────────────────────
    keyword_matches = 0
    for kw in BUSINESS_KEYWORDS:
        if kw in text:
            keyword_matches += 1
    # Score = min(matches / 5, 1.0) — 5+ distinct keywords = max signal
    keyword_signal = min(keyword_matches / 5, 1.0)

    # ── Signal 5: interaction ──────────────────────────────────────────
    interaction_matches = 0
    for pat in INTERACTION_PATTERNS:
        if re.search(pat, text):
            interaction_matches += 1
    # Score = min(matches / 3, 1.0) — 3+ distinct markers = max signal
    interaction_signal = min(interaction_matches / 3, 1.0)

    # ── Signal dict ────────────────────────────────────────────────────
    signals = {
        "token_count": round(token_count_signal, 4),
        "unique_words": round(unique_words_signal, 4),
        "entity_density": round(entity_density_signal, 4),
        "keyword_signal": round(keyword_signal, 4),
        "interaction": round(interaction_signal, 4),
    }

    # ── Weighted combine ───────────────────────────────────────────────
    total_weight = sum(SIGNAL_WEIGHTS.values())
    score = sum(
        signals[sig] * SIGNAL_WEIGHTS[sig]
        for sig in SIGNAL_WEIGHTS
    ) / total_weight
    score = round(score, 4)

    # ── Band classification ────────────────────────────────────────────
    if score >= DEFINITE_KEEP:
        band = "definite_keep"
    elif score <= DEFINITE_DROP:
        band = "definite_drop"
    else:
        band = "borderline"

    should_process = band != "definite_drop"

    # ── Drop reason ────────────────────────────────────────────────────
    drop_reason = None
    if band == "definite_drop":
        reasons = []
        if token_count_signal <= DEFINITE_DROP:
            reasons.append(f"token_count={token_count_signal}")
        if unique_words_signal <= DEFINITE_DROP:
            reasons.append(f"unique_words={unique_words_signal}")
        if entity_density_signal == 0.0:
            reasons.append("no_entities")
        if keyword_signal == 0.0:
            reasons.append("no_keywords")
        if interaction_signal == 0.0:
            reasons.append("no_interaction")
        drop_reason = "; ".join(reasons) if reasons else f"score={score}"

    return {
        "score": score,
        "band": band,
        "should_process": should_process,
        "signals": signals,
        "drop_reason": drop_reason,
    }


def detect_signals(text: str, source: str = "user_message",
                   timestamp: str = None) -> dict:
    """
    텍스트에서 시그널(아이디어 + 엔티티) 감지

    Uses fast_score_transcript as a pre-filter: if the 3-band score
    classifies the text as "definite_drop", it's marked as operational
    and skipped (no event recording, no LLM processing needed).

    Args:
        text: 분석할 텍스트
        source: 메시지 출처
        timestamp: ISO 타임스탬프

    Returns:
        dict: {ideas: [...], entities: [...], summary: str, fast_score: {...}}
    """
    if timestamp is None:
        timestamp = datetime.now(KST).isoformat()

    # ── 3-band fast pre-filter ────────────────────────────────────────
    fast_score = fast_score_transcript(text)

    known_entities = _load_entity_index()

    ideas = _detect_ideas(text)
    entities = _detect_entities(text, known_entities)

    # 짧은 운영 메시지는 스킵 (legacy check, kept for backward compat)
    is_operational = (
        len(text.strip()) < 5
        or text.strip().lower() in (
            "ok", "okay", "네", "응", "알겠어", "고마워", "감사", "ㅇㅇ", "ㄱㄱ"
        )
        # New: definite_drop from fast_score also marks as operational
        or not fast_score["should_process"]
    )

    result = {
        "timestamp": timestamp,
        "source": source,
        "ideas": ideas,
        "entities": entities,
        "is_operational": is_operational,
        "fast_score": fast_score,
        "summary": (
            f"Signals: {len(ideas)} ideas, {len(entities)} entities"
            f" | fast_score={fast_score['score']:.3f} [{fast_score['band']}]"
            + (" (skipped: operational)" if is_operational else "")
        ),
    }

    # 비운영 메시지만 이벤트 기록
    if not is_operational and (ideas or entities):
        _append_event({
            "event_type": "signal_detected",
            "timestamp": timestamp,
            "source": source,
            "ideas": ideas,
            "entities": entities,
            "fast_score": fast_score,
        })

    return result


def run(events: int = 100, dry_run: bool = False,
        verbose: bool = False, emit: bool = True) -> dict:
    """
    Dream Cycle 호환 래퍼
    events.jsonl에서 최근 이벤트를 읽어 시그널 감지 + 백링크 생성

    Args:
        events: 처리할 최대 이벤트 수
        dry_run: 실제 쓰기 안 함
        verbose: 상세 로그
        emit: emit_event 호출 여부

    Returns:
        dict: {processed: int, signals_found: int, backlinks_created: int}
    """
    if not EVENTS_PATH.exists():
        return {"processed": 0, "signals_found": 0, "backlinks_created": 0,
                "error": "events.jsonl not found"}

    processed = 0
    signals_found = 0
    backlinks_created = 0

    # 최근 이벤트 읽기
    all_events = []
    with open(EVENTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    recent = all_events[-events:]

    for event in recent:
        processed += 1
        text = json.dumps(event, ensure_ascii=False)
        result = detect_signals(text, source="dream_cycle")
        if result["ideas"] or result["entities"]:
            signals_found += 1
            if not dry_run:
                pass  # detect_signals가 이미 _append_event 호출

        if verbose:
            print(result["summary"])

    return {
        "processed": processed,
        "signals_found": signals_found,
        "backlinks_created": backlinks_created,
    }


def _cli_score(args: list[str]) -> int:
    """Handle --score CLI mode"""
    text = " ".join(args)
    result = fast_score_transcript(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cli_detect(args: list[str]) -> int:
    """Handle text detection (default)"""
    if args:
        text = " ".join(args)
    else:
        text = sys.stdin.read()

    result = detect_signals(text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main():
    """CLI: stdin 또는 인자로 텍스트 받아서 시그널 감지

    Modes:
        --score "text"     → run fast_score_transcript only
        (default) "text"   → run detect_signals
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--score":
        args = sys.argv[2:]
        return _cli_score(args)
    else:
        return _cli_detect(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
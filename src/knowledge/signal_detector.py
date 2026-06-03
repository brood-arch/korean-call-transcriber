#!/usr/bin/env python3
"""
Signal Detector — Automatic idea and entity extraction from text.

Inspired by the GBrain signal-detector pattern. Extracts from any text:
1. Original thinking (ideas, observations, hypotheses, frameworks)
2. Entity mentions (people, companies, places, products)

Features 3-band fast scoring as a pre-filter before any LLM API call:
    - definite_keep  (score >= 0.85) → definitely process
    - borderline     (0.15 < score < 0.85) → admit for further processing
    - definite_drop  (score <= 0.15) → skip entirely (greetings, confirmations)

Signals used in fast_score:
    - token_count:    Text length (longer = more substance)
    - unique_words:   Vocabulary diversity
    - entity_density: Named entities per token (phones, amounts, products)
    - keyword_signal: Business keyword presence (주문, 견적, 납기, etc.)
    - interaction:    Direct engagement markers (questions, imperatives, commitments)

Environment variables:
    KCT_STATE_DIR — Base state directory (default: state)

Usage:
    from src.knowledge.signal_detector import detect_signals, fast_score_transcript

    # Fast scoring (no LLM needed)
    result = fast_score_transcript("주문 500개 확인해주세요")
    print(result['band'], result['score'])

    # Full signal detection
    result = detect_signals("Meeting with Acme Corp about 500 units order")
    print(result['ideas'], result['entities'])
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

KST = timezone(timedelta(hours=9))

_STATE_DIR = Path(os.environ.get("KCT_STATE_DIR", "state"))
EVENTS_PATH = _STATE_DIR / "events.jsonl"

# ── 3-band fast_score constants ──────────────────────────────────────────

SIGNAL_WEIGHTS = {
    "token_count": 1.0,
    "unique_words": 1.0,
    "entity_density": 1.5,
    "keyword_signal": 2.0,
    "interaction": 3.0,
}

DEFINITE_DROP = 0.15
DEFINITE_KEEP = 0.85

# Business keywords (Korean call transcript patterns)
BUSINESS_KEYWORDS = [
    "주문", "견적", "발주", "납기", "결제", "입금", "출고", "배송",
    "수령", "방문", "미팅", "약속", "연락", "확인", "처리", "준비",
    "발송", "회신", "요청", "교정",
    "order", "quote", "delivery", "payment", "invoice", "shipping",
]

# Product names for entity density
PRODUCT_NAMES = [
    "송풍기", "환풍기", "채반", "블로어", "모터", "케이스", "날개",
    "blower", "fan", "motor", "blade",
]

# Interaction / engagement markers
INTERACTION_PATTERNS = [
    # Specific questions
    r"언제", r"몇시", r"어디", r"어떻게", r"몇 개", r"얼마",
    r"when", r"where", r"how", r"how many", r"how much",
    # Imperative verbs (Korean)
    r"해줘", r"부탁드려", r"보내줘", r"확인해", r"알려줘", r"전달해",
    # Explicit commitments
    r"하겠습니다", r"보내드리겠", r"처리하겠", r"진행하겠",
]

# Amount pattern: Korean-style amounts
AMOUNT_PATTERN = r"\d+[,\d]*\s*(?:만\s*)?원"

# Phone pattern
PHONE_PATTERN = r"01[016789]-\d{3,4}-\d{4}"


# ── Event logging ────────────────────────────────────────────────────────

def _append_event(event: dict) -> None:
    """Append an event to the events JSONL file."""
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# ── Fast scoring ─────────────────────────────────────────────────────────

def fast_score_transcript(text: str) -> dict:
    """3-band fast scoring for call transcripts.

    A cheap, deterministic pre-filter that runs BEFORE any LLM API call.
    Returns a score dict with band classification and per-signal breakdown.

    Bands:
        definite_keep  (score >= 0.85) → definitely process
        borderline     (0.15 < score < 0.85) → admit for further processing
        definite_drop  (score <= 0.15) → skip entirely

    Args:
        text: The transcript/text to score.

    Returns:
        dict with keys: score, band, should_process, signals, drop_reason
    """
    tokens = text.strip().split()
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))

    # Signal 1: token_count
    token_count_signal = min(total_tokens / 200, 1.0)

    # Signal 2: unique_words
    unique_words_signal = unique_tokens / max(total_tokens, 1)

    # Signal 3: entity_density
    phone_matches = len(re.findall(PHONE_PATTERN, text))
    amount_matches = len(re.findall(AMOUNT_PATTERN, text))
    product_matches = sum(
        len(re.findall(re.escape(prod), text)) for prod in PRODUCT_NAMES
    )
    entity_count = phone_matches + amount_matches + product_matches
    entity_density_signal = (
        min(entity_count / max(total_tokens, 1) / 0.01, 1.0)
        if total_tokens > 0
        else 0.0
    )

    # Signal 4: keyword_signal
    keyword_matches = sum(1 for kw in BUSINESS_KEYWORDS if kw in text)
    keyword_signal = min(keyword_matches / 5, 1.0)

    # Signal 5: interaction
    interaction_matches = sum(
        1 for pat in INTERACTION_PATTERNS if re.search(pat, text)
    )
    interaction_signal = min(interaction_matches / 3, 1.0)

    signals = {
        "token_count": round(token_count_signal, 4),
        "unique_words": round(unique_words_signal, 4),
        "entity_density": round(entity_density_signal, 4),
        "keyword_signal": round(keyword_signal, 4),
        "interaction": round(interaction_signal, 4),
    }

    # Weighted combine
    total_weight = sum(SIGNAL_WEIGHTS.values())
    score = sum(signals[sig] * SIGNAL_WEIGHTS[sig] for sig in SIGNAL_WEIGHTS) / total_weight
    score = round(score, 4)

    # Band classification
    if score >= DEFINITE_KEEP:
        band = "definite_keep"
    elif score <= DEFINITE_DROP:
        band = "definite_drop"
    else:
        band = "borderline"

    should_process = band != "definite_drop"

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


# ── Idea detection ───────────────────────────────────────────────────────

def _detect_ideas(text: str) -> list[dict]:
    """Detect original thinking (ideas/observations/hypotheses) in text."""
    ideas = []

    idea_patterns = [
        r"(?:(?:아이디어|생각|가설|프레임워크|아이디어인데)[은는이가]\s*:?\s*)(.+?)(?:\.|$)",
        r"(?:새로운\s+(?:접근|방법|아이디어|생각))\s*[은는]?\s*(.+?)(?:\.|$)",
        r"(?:이렇게\s+하면|이렇게\s+되면|만약\s+.+?면)\s+(.+?)(?:\.|$)",
        r"(?:idea|thought|hypothesis|framework)\s*:?\s*(.+?)(?:\.|$)",
    ]

    for pattern in idea_patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            idea_text = m.group(1).strip()
            if len(idea_text) > 5:
                ideas.append({
                    "type": "idea",
                    "content": idea_text,
                    "original_phrase": m.group(0).strip(),
                    "phase": "original_thinking",
                })

    return ideas


# ── Entity detection ─────────────────────────────────────────────────────

def _detect_entities(text: str) -> list[dict]:
    """Detect entities (people, companies, products) in text."""
    entities = []

    # General patterns (no hardcoded personal names in open-source version)
    org_pattern = r"(?:[A-Z][a-zA-Z]+(?:코리아|케이스|산업|기업|Corp|Inc|Ltd)?|(?:주식회사|㈜)\s*[\w]+)"
    product_pattern = r"(?:환풍기|송풍기|케이스|날개|채반|블로어|팬|모터|blower|fan|motor)"

    for pattern, etype in [
        (org_pattern, "company"),
        (product_pattern, "product"),
    ]:
        for m in re.finditer(pattern, text):
            name = m.group(0).strip()
            if len(name) >= 2:
                entities.append({
                    "type": "entity",
                    "entity_type": etype,
                    "name": name,
                    "action": "enrich",
                })

    # Deduplicate
    seen: set[tuple[str, str]] = set()
    unique = []
    for e in entities:
        key = (e["entity_type"], e["name"])
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


# ── Main detection ───────────────────────────────────────────────────────

def detect_signals(
    text: str,
    source: str = "user_message",
    timestamp: str | None = None,
) -> dict:
    """Detect signals (ideas + entities) from text.

    Uses fast_score_transcript as a pre-filter: if the 3-band score
    classifies the text as 'definite_drop', it's marked as operational
    and skipped (no event recording, no LLM processing needed).

    Args:
        text: Text to analyze.
        source: Message origin label.
        timestamp: ISO timestamp (defaults to now in KST).

    Returns:
        dict: {ideas, entities, summary, fast_score, is_operational}
    """
    if timestamp is None:
        timestamp = datetime.now(KST).isoformat()

    fast_score = fast_score_transcript(text)
    ideas = _detect_ideas(text)
    entities = _detect_entities(text)

    # Short operational messages are skipped
    is_operational = (
        len(text.strip()) < 5
        or text.strip().lower() in (
            "ok", "okay", "네", "응", "알겠어", "고마워", "감사", "ㅇㅇ", "ㄱㄱ",
            "yes", "no", "thanks", "got it", "sure",
        )
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

    # Record non-operational signals
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


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> int:
    """CLI: stdin or args → signal detection.

    Modes:
        --score "text"   → run fast_score_transcript only
        (default) "text" → run detect_signals
    """
    if len(sys.argv) > 1 and sys.argv[1] == "--score":
        text = " ".join(sys.argv[2:])
        result = fast_score_transcript(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    else:
        text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
        result = detect_signals(text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0


if __name__ == "__main__":
    sys.exit(main())

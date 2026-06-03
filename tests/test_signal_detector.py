"""Tests for src.knowledge.signal_detector — 3-band fast scoring, signal extraction."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Redirect state dir so event writes go to tmp_path."""
    monkeypatch.setenv("KCT_STATE_DIR", str(tmp_path))
    import src.knowledge.signal_detector as sd
    monkeypatch.setattr(sd, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(sd, "EVENTS_PATH", tmp_path / "events.jsonl")


# ── 3-band fast scoring ──────────────────────────────────────────────────

def test_fast_score_definite_keep():
    """Long text with many business keywords + interaction markers + entities → definite_keep."""
    from src.knowledge.signal_detector import fast_score_transcript
    text = (
        "주문 500개 견적 확인 부탁드려요. "
        "납기 언제인가요? 결제는 입금 완료했습니다. "
        "발송해주세요. 환풍기 송풍기 채반 모터 블로어 "
        "케이스 날개 모두 포함해서 발주합니다. "
        "배송 수령 방문 미팅 약속 연락 확인 처리 준비 "
        "발송 회신 요청 교정 모두 진행해주세요. "
        "하겠습니다 보내드리겠 처리하겠 진행하겠. "
        "언제 몇시 어디 어떻게 몇 개 얼마인지 알려주세요. "
        "해줘 부탁드려 보내줘 확인해 알려줘 전달해 주세요."
    )
    result = fast_score_transcript(text)
    assert result["band"] == "definite_keep", f"Got {result['band']} with score {result['score']}"
    assert result["score"] >= 0.85
    assert result["should_process"] is True
    assert result["drop_reason"] is None


def test_fast_score_definite_drop():
    """Very short, no keywords, no entities → definite_drop."""
    from src.knowledge.signal_detector import fast_score_transcript
    result = fast_score_transcript("ok")
    assert result["band"] == "definite_drop"
    assert result["score"] <= 0.15
    assert result["should_process"] is False
    assert result["drop_reason"] is not None


def test_fast_score_borderline():
    """Medium-length text with some substance → borderline."""
    from src.knowledge.signal_detector import fast_score_transcript
    text = "오늘 회의에서 논의한 내용 정리"
    result = fast_score_transcript(text)
    assert result["band"] in ("borderline", "definite_keep", "definite_drop")
    assert "score" in result
    assert "signals" in result


def test_fast_score_signals_structure():
    """All 5 signals present in output."""
    from src.knowledge.signal_detector import fast_score_transcript
    result = fast_score_transcript("테스트")
    signals = result["signals"]
    assert set(signals.keys()) == {
        "token_count", "unique_words", "entity_density",
        "keyword_signal", "interaction",
    }


def test_fast_score_empty_text():
    from src.knowledge.signal_detector import fast_score_transcript
    result = fast_score_transcript("")
    assert result["band"] == "definite_drop"
    assert result["should_process"] is False


# ── Signal extraction ────────────────────────────────────────────────────

def test_detect_signals_with_entities():
    """Text with product names should detect entities."""
    from src.knowledge.signal_detector import detect_signals
    result = detect_signals("송풍기 주문 500개 확인해주세요")
    assert len(result["entities"]) >= 1
    # Should find 송풍기 as a product entity
    products = [e for e in result["entities"] if e["entity_type"] == "product"]
    assert any(p["name"] == "송풍기" for p in products)


def test_detect_signals_idea_pattern():
    """Text with idea pattern should detect idea."""
    from src.knowledge.signal_detector import detect_signals
    text = "아이디어인데: 새로운 접근 방법으로 비용 절감 가능"
    result = detect_signals(text)
    assert len(result["ideas"]) >= 1


def test_detect_signals_operational_short():
    """Short operational messages are skipped."""
    from src.knowledge.signal_detector import detect_signals
    result = detect_signals("네")
    assert result["is_operational"] is True


def test_detect_signals_operational_keywords():
    """Known operational keywords marked as operational."""
    from src.knowledge.signal_detector import detect_signals
    for text in ("ok", "yes", "고마워", "알겠어"):
        result = detect_signals(text)
        assert result["is_operational"] is True, f"'{text}' should be operational"


def test_detect_signals_non_operational():
    """Business message should not be operational."""
    from src.knowledge.signal_detector import detect_signals
    text = "주문 500개 송풍기 발송 확인 부탁드립니다. 연락 주세요."
    result = detect_signals(text)
    assert result["is_operational"] is False


def test_detect_signals_summary_present():
    from src.knowledge.signal_detector import detect_signals
    result = detect_signals("테스트 메시지")
    assert "summary" in result
    assert "fast_score" in result["summary"]


def test_detect_signals_event_not_recorded_for_operational(tmp_path):
    """Operational messages should not append events."""
    from src.knowledge.signal_detector import detect_signals
    events_path = tmp_path / "events.jsonl"
    detect_signals("ok")
    if events_path.exists():
        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 0


# ── No external API calls ────────────────────────────────────────────────

def test_no_llm_api_calls():
    """Ensure fast_score_transcript never calls an external API."""
    from src.knowledge.signal_detector import fast_score_transcript
    with patch("urllib.request.urlopen") as mock_urlopen:
        fast_score_transcript("주문 확인 견적 발송")
        mock_urlopen.assert_not_called()

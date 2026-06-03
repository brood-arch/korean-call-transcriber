"""Tests for src.integrations.email_todo_extract — promo filter, exclusions, state, LLM mock."""

import json
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setenv("EMAIL_TODO_STATE", str(tmp_path / "email_todo_state.json"))
    monkeypatch.setenv("EMAIL_TODO_EXCLUSIONS", str(tmp_path / "email_todo_exclusions.json"))
    import src.integrations.email_todo_extract as ete
    monkeypatch.setattr(ete, "EXTRACT_STATE_PATH", tmp_path / "email_todo_state.json")
    monkeypatch.setattr(ete, "EXCLUSION_PATH", tmp_path / "email_todo_exclusions.json")


# ── Promotional detection ───────────────────────────────────────────────

def test_is_promotional_subject_tags():
    from src.integrations.email_todo_extract import is_promotional
    assert is_promotional("[광고] 특가 할인", "")
    assert is_promotional("[AD] Sale now", "")
    assert is_promotional("[EVENT] 이벤트 안내", "")


def test_is_promotional_body_keywords():
    from src.integrations.email_todo_extract import is_promotional
    assert is_promotional("정상 제목", "수신거부 안내입니다.")
    assert is_promotional("정상 제목", "Unsubscribe here")


def test_is_not_promotional():
    from src.integrations.email_todo_extract import is_promotional
    assert not is_promotional("견적서 확인 요청", "안녕하세요, 견적서 확인 부탁드립니다.")
    assert not is_promotional("회의 일정", "내일 미팅 확정")


# ── Exclusion list ──────────────────────────────────────────────────────

def test_load_exclusions_empty(tmp_path):
    from src.integrations.email_todo_extract import load_exclusions
    excl = load_exclusions()
    assert excl == {"senders": []}


def test_save_and_load_exclusions(tmp_path):
    from src.integrations.email_todo_extract import load_exclusions, save_exclusions
    excl = {"senders": ["spam@example.com"]}
    save_exclusions(excl)
    loaded = load_exclusions()
    assert "spam@example.com" in loaded["senders"]


def test_add_exclusion(tmp_path):
    from src.integrations.email_todo_extract import add_exclusion
    excl = add_exclusion("Spam@Example.COM")
    assert "spam@example.com" in excl["senders"]


def test_add_exclusion_no_duplicate(tmp_path):
    from src.integrations.email_todo_extract import add_exclusion
    add_exclusion("spam@example.com")
    excl = add_exclusion("spam@example.com")
    assert excl["senders"].count("spam@example.com") == 1


def test_remove_exclusion(tmp_path):
    from src.integrations.email_todo_extract import add_exclusion, remove_exclusion
    add_exclusion("spam@example.com")
    add_exclusion("ads@example.com")
    excl = remove_exclusion("spam@example.com")
    assert "spam@example.com" not in excl["senders"]
    assert "ads@example.com" in excl["senders"]


# ── State management ────────────────────────────────────────────────────

def test_load_state_empty(tmp_path):
    from src.integrations.email_todo_extract import load_state
    state = load_state()
    assert "extracted_uids" in state
    assert state["last_extraction"] is None


def test_save_and_load_state(tmp_path):
    from src.integrations.email_todo_extract import load_state, save_state
    state = {"extracted_uids": {"INBOX:1": {"status": "extracted"}}, "last_extraction": "2026-06-01"}
    save_state(state)
    loaded = load_state()
    assert "INBOX:1" in loaded["extracted_uids"]


# ── LLM extraction (mocked) ────────────────────────────────────────────

def test_call_llm_extract_mocked(tmp_path, monkeypatch):
    """Verify call_llm_extract parses a valid JSON response."""
    from src.integrations.email_todo_extract import call_llm_extract
    mock_response = json.dumps({
        "choices": [{
            "message": {
                "content": json.dumps({
                    "has_actionable_items": True,
                    "todos": [{"title": "Reply to client", "priority": "high"}],
                })
            }
        }]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = mock_response
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("src.integrations.email_todo_extract.urllib.request.urlopen", return_value=mock_resp):
        result = call_llm_extract("Email content here", "fake-key", "https://api.test.com/v1", "test-model")

    assert result is not None
    assert result["has_actionable_items"] is True
    assert len(result["todos"]) == 1


def test_call_llm_extract_failure_returns_none(monkeypatch):
    """If LLM call fails, return None."""
    from src.integrations.email_todo_extract import call_llm_extract
    with patch("src.integrations.email_todo_extract.urllib.request.urlopen", side_effect=Exception("API error")):
        result = call_llm_extract("content", "key", "https://api.test.com/v1", "model")
    assert result is None


# ── Full pipeline (extract_todos_from_emails) ──────────────────────────

def test_extract_todos_from_emails_dry_run():
    from src.integrations.email_todo_extract import extract_todos_from_emails
    result = extract_todos_from_emails([{"meta": {}, "staged": ""}], dry_run=True)
    assert result == []


def test_extract_todos_from_emails_no_api_key(monkeypatch):
    from src.integrations.email_todo_extract import extract_todos_from_emails
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    result = extract_todos_from_emails([{"meta": {}, "staged": ""}])
    assert result == []


def test_extract_todos_from_emails_skips_promo(tmp_path, monkeypatch):
    from src.integrations.email_todo_extract import extract_todos_from_emails
    monkeypatch.setenv("LLM_API_KEY", "fake-key")

    # Create staged file with promo content
    staged = tmp_path / "staged_email.txt"
    staged.write_text("수신거부 안내 메일입니다. [광고] 특가 할인", encoding="utf-8")

    rows = [{
        "meta": {
            "subject": "[광고] 특가 할인 안내",
            "from": "promo@spam.com",
            "uid": "42",
            "date": "2026-06-01",
        },
        "staged": str(staged),
        "folder": "INBOX",
    }]
    result = extract_todos_from_emails(rows)
    assert result == []


def test_extract_todos_from_emails_skips_sent(tmp_path, monkeypatch):
    from src.integrations.email_todo_extract import extract_todos_from_emails
    monkeypatch.setenv("LLM_API_KEY", "fake-key")

    staged = tmp_path / "sent.txt"
    staged.write_text("Sent mail content", encoding="utf-8")

    rows = [{
        "meta": {"subject": "Sent", "from": "me@example.com", "uid": "1"},
        "staged": str(staged),
        "folder": "Sent Messages",
    }]
    result = extract_todos_from_emails(rows)
    assert result == []

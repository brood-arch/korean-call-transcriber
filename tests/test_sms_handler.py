"""Tests for kct.integrations.sms_handler — payload normalization, TODO conversion, edge cases."""



# ── SMSMessage dataclass ────────────────────────────────────────────────

def test_sms_message_fields():
    from kct.integrations.sms_handler import SMSMessage
    sms = SMSMessage(
        sender="+82-10-1234-5678",
        message="Test message",
        timestamp="2026-06-03T10:00:00+09:00",
    )
    assert sms.sender == "+82-10-1234-5678"
    assert sms.message == "Test message"
    assert sms.thread_id == ""
    assert sms.raw_data == {}


# ── Payload normalization ────────────────────────────────────────────────

def test_normalize_sms_payload_standard():
    from kct.integrations.sms_handler import normalize_sms_payload
    payload = {
        "from": "+82-10-1234-5678",
        "body": "Meeting confirmed",
        "timestamp": "2026-06-03T10:00:00+09:00",
        "thread_id": "thread-123",
    }
    sms = normalize_sms_payload(payload)
    assert sms.sender == "+82-10-1234-5678"
    assert sms.message == "Meeting confirmed"
    assert sms.thread_id == "thread-123"
    assert sms.raw_data == payload


def test_normalize_sms_payload_alternative_keys():
    """Test with 'sender', 'text', and 'message' alternatives."""
    from kct.integrations.sms_handler import normalize_sms_payload
    # 'sender' instead of 'from'
    sms = normalize_sms_payload({"sender": "+82-10-9999", "body": "hi"})
    assert sms.sender == "+82-10-9999"

    # 'text' instead of 'body'
    sms = normalize_sms_payload({"from": "+82-10-1111", "text": "hello"})
    assert sms.message == "hello"

    # 'message' instead of 'body'
    sms = normalize_sms_payload({"from": "+82-10-2222", "message": "world"})
    assert sms.message == "world"


def test_normalize_sms_payload_missing_fields():
    from kct.integrations.sms_handler import normalize_sms_payload
    sms = normalize_sms_payload({})
    assert sms.sender == ""
    assert sms.message == ""
    assert sms.thread_id == ""


def test_normalize_sms_payload_preserves_raw():
    from kct.integrations.sms_handler import normalize_sms_payload
    payload = {"from": "A", "body": "B", "extra_key": "extra_value"}
    sms = normalize_sms_payload(payload)
    assert sms.raw_data["extra_key"] == "extra_value"


# ── TODO extraction (placeholder) ───────────────────────────────────────

def test_extract_sms_todos_returns_empty():
    """Default implementation returns empty list."""
    from kct.integrations.sms_handler import SMSMessage, extract_sms_todos
    sms = SMSMessage(sender="A", message="test", timestamp="2026-01-01T00:00:00")
    todos = extract_sms_todos(sms)
    assert isinstance(todos, list)
    assert len(todos) == 0


# ── SMS to TODO entry conversion ────────────────────────────────────────

def test_sms_to_todo_entry_enriches():
    from kct.integrations.sms_handler import SMSMessage, sms_to_todo_entry
    sms = SMSMessage(
        sender="+82-10-1234",
        message="test",
        timestamp="2026-06-03T10:00:00+09:00",
        thread_id="t1",
    )
    todos = [
        {"title": "Follow up", "priority": "high"},
        {"title": "Send quote", "priority": "medium"},
    ]
    entries = sms_to_todo_entry(sms, todos)
    assert len(entries) == 2
    for e in entries:
        assert e["source"] == "sms"
        assert e["source_phone"] == "+82-10-1234"
        assert e["sms_thread"] == "t1"
        assert e["sms_timestamp"] == "2026-06-03T10:00:00+09:00"


def test_sms_to_todo_entry_empty():
    from kct.integrations.sms_handler import SMSMessage, sms_to_todo_entry
    sms = SMSMessage(sender="A", message="test", timestamp="2026-01-01")
    entries = sms_to_todo_entry(sms, [])
    assert entries == []


def test_sms_to_todo_entry_preserves_original_fields():
    from kct.integrations.sms_handler import SMSMessage, sms_to_todo_entry
    sms = SMSMessage(sender="B", message="test", timestamp="2026-01-01")
    todos = [{"title": "Task", "priority": "low", "custom_field": "value"}]
    entries = sms_to_todo_entry(sms, todos)
    assert entries[0]["custom_field"] == "value"
    assert entries[0]["priority"] == "low"


# ── Edge cases ───────────────────────────────────────────────────────────

def test_normalize_sms_payload_none_values():
    from kct.integrations.sms_handler import normalize_sms_payload
    sms = normalize_sms_payload({"from": None, "body": None})
    # Should not crash; .get() with None returns None which becomes ""
    assert isinstance(sms.sender, (str, type(None)))
    assert isinstance(sms.message, (str, type(None)))


def test_sms_message_raw_data_default():
    from kct.integrations.sms_handler import SMSMessage
    sms = SMSMessage(sender="A", message="B", timestamp="T")
    assert sms.raw_data == {}

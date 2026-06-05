"""Tests for kct.integrations.naver_mail — message parsing, header decode, state I/O."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setenv("NAVER_MAIL_STATE_DIR", str(tmp_path / "naver_state"))


# ── Header decoding ──────────────────────────────────────────────────────

def test_decode_header_value_ascii():
    from kct.integrations.naver_mail import _decode_header_value
    assert _decode_header_value("Hello World") == "Hello World"


def test_decode_header_value_korean_encoded():
    from kct.integrations.naver_mail import _decode_header_value
    result = _decode_header_value("=?utf-8?B?7YWM7Iqk7Yq4?=")
    assert result == "테스트"


def test_decode_header_value_none():
    from kct.integrations.naver_mail import _decode_header_value
    assert _decode_header_value(None) == ""
    assert _decode_header_value("") == ""


def test_decode_header_value_mixed():
    from kct.integrations.naver_mail import _decode_header_value
    result = _decode_header_value("Re: =?utf-8?B?7YWM7Iqk7Yq4?= =?utf-8?B?7JWI64WV?=")
    assert "테스트" in result


# ── Body extraction ─────────────────────────────────────────────────────

def test_extract_body_plain_text():
    import email as email_lib

    from kct.integrations.naver_mail import _extract_body
    msg = email_lib.message_from_string(
        "Content-Type: text/plain; charset=utf-8\r\n\r\nHello plain text"
    )
    body = _extract_body(msg)
    assert "Hello plain text" in body


def test_extract_body_html_fallback():
    import email as email_lib

    from kct.integrations.naver_mail import _extract_body
    raw = (
        "MIME-Version: 1.0\r\n"
        "Content-Type: multipart/alternative; boundary=bnd\r\n"
        "\r\n"
        "--bnd\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "\r\n"
        "<p>HTML content</p>\r\n"
        "--bnd--\r\n"
    )
    msg = email_lib.message_from_string(raw)
    body = _extract_body(msg)
    assert "HTML content" in body


def test_extract_body_korean():
    from email.message import EmailMessage

    from kct.integrations.naver_mail import _extract_body
    msg = EmailMessage()
    msg.set_content("안녕하세요 반갑습니다", charset="utf-8")
    body = _extract_body(msg)
    assert "안녕하세요" in body


# ── Message parsing ─────────────────────────────────────────────────────

def test_parse_message_basic():
    from kct.integrations.naver_mail import parse_message
    raw = (
        b"From: sender@example.com\r\n"
        b"To: receiver@naver.com\r\n"
        b"Subject: Test Subject\r\n"
        b"Date: Wed, 01 Jun 2026 10:00:00 +0900\r\n"
        b"Message-ID: <12345@example.com>\r\n"
        b"\r\n"
        b"Email body content here"
    )
    result = parse_message(raw)
    assert result["subject"] == "Test Subject"
    assert "sender@example.com" in result["from"]
    assert "receiver@naver.com" in result["to"]
    assert result["date"] == "Wed, 01 Jun 2026 10:00:00 +0900"
    assert "Email body content" in result["body"]


def test_parse_message_korean_subject():
    import email as email_lib

    from kct.integrations.naver_mail import parse_message
    # Build message with Korean encoded subject
    msg = email_lib.message_from_string(
        "From: sender@example.com\r\n"
        "To: receiver@naver.com\r\n"
        "Subject: =?utf-8?B?7YWM7Iqk7Yq4?=\r\n"
        "Date: Wed, 01 Jun 2026 10:00:00 +0900\r\n"
        "\r\n"
        "Body text"
    )
    raw = msg.as_bytes()
    result = parse_message(raw)
    assert result["subject"] == "테스트"


# ── State load/save ─────────────────────────────────────────────────────

def test_load_state_empty(tmp_path):
    from kct.integrations.naver_mail import load_state
    state = load_state(tmp_path / "nonexistent")
    assert state == {}


def test_save_and_load_state(tmp_path):
    from kct.integrations.naver_mail import load_state, save_state
    state_dir = tmp_path / "state"
    original = {"INBOX": ["1", "2", "3"]}
    save_state(state_dir, original)
    loaded = load_state(state_dir)
    assert loaded == original


def test_save_state_creates_directory(tmp_path):
    from kct.integrations.naver_mail import save_state
    state_dir = tmp_path / "deep" / "nested" / "dir"
    save_state(state_dir, {"key": "val"})
    assert (state_dir / "processed_uids.json").exists()


# ── Credentials check ───────────────────────────────────────────────────

def test_credentials_missing_raises(monkeypatch):
    import kct.integrations.naver_mail as nm
    monkeypatch.setattr(nm, "NAVER_MAIL_ADDRESS", "")
    monkeypatch.setattr(nm, "NAVER_MAIL_PASSWORD", "")
    with pytest.raises(EnvironmentError):
        nm._get_credentials()


def test_credentials_present(monkeypatch):
    import kct.integrations.naver_mail as nm
    monkeypatch.setattr(nm, "NAVER_MAIL_ADDRESS", "user@naver.com")
    monkeypatch.setattr(nm, "NAVER_MAIL_PASSWORD", "secretpass")
    addr, pw = nm._get_credentials()
    assert addr == "user@naver.com"
    assert pw == "secretpass"


# ── fetch_messages mocked (no real IMAP) ────────────────────────────────

def test_fetch_messages_mocked(tmp_path, monkeypatch):
    """Test fetch_messages with mocked IMAP connection."""
    from kct.integrations.naver_mail import fetch_messages

    mock_imap = MagicMock()
    mock_imap.__enter__ = MagicMock(return_value=mock_imap)
    mock_imap.__exit__ = MagicMock(return_value=False)
    mock_imap.select.return_value = ("OK", None)
    mock_imap.uid.side_effect = [
        ("OK", [b"1 2"]),  # SEARCH
        ("OK", [(b"1", b"Subject: Test\r\nFrom: a@b.com\r\n\r\nBody1")]),  # FETCH 1
        ("OK", [(b"2", b"Subject: Test2\r\nFrom: c@d.com\r\n\r\nBody2")]),  # FETCH 2
    ]

    with patch("kct.integrations.naver_mail.imaplib.IMAP4_SSL", return_value=mock_imap):
        state = {}
        results = fetch_messages(
            account="user@naver.com",
            password="pass",
            folders=["INBOX"],
            limit=10,
            state=state,
        )
    assert len(results) == 2
    assert results[0]["subject"] == "Test"
    assert "INBOX" in state

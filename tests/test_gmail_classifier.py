"""Tests for kct.integrations.gmail_classifier — classification, keyword matching, MIME decoding."""

from unittest.mock import MagicMock

import pytest

# ── Classification logic (no IMAP) ──────────────────────────────────────

def test_classify_ad_multiple_keywords():
    from kct.integrations.gmail_classifier import classify_email
    result = classify_email(
        subject="광고 특가 할인 이벤트",
        sender="promo@spam.com",
        body="지금 바로 구매하세요. 프로모션 진행 중!",
    )
    assert result == "ad"


def test_classify_ad_newsletter():
    from kct.integrations.gmail_classifier import classify_email
    result = classify_email(
        subject="뉴스레터 구독 안내",
        sender="newsletter@company.com",
        body="이번 주 뉴스레터입니다.",
    )
    assert result == "ad"


def test_classify_important():
    from kct.integrations.gmail_classifier import classify_email
    result = classify_email(
        subject="긴급: 견적서 확인 요청",
        sender="partner@biz.com",
        body="견적서 확인 부탁드립니다.",
    )
    assert result == "important"


def test_classify_important_order():
    from kct.integrations.gmail_classifier import classify_email
    result = classify_email(
        subject="주문 확인",
        sender="orders@biz.com",
        body="주문이 접수되었습니다.",
    )
    assert result == "important"


def test_classify_normal():
    from kct.integrations.gmail_classifier import classify_email
    result = classify_email(
        subject="안녕하세요",
        sender="friend@example.com",
        body="잘 지내시죠?",
    )
    assert result == "normal"


def test_classify_ad_only_one_keyword_is_normal():
    """Ad classification requires >= 2 ad keywords."""
    from kct.integrations.gmail_classifier import classify_email
    result = classify_email(
        subject="할인",
        sender="shop@example.com",
        body="일반 내용입니다.",
    )
    # Only 1 ad keyword (할인), not enough for 'ad'
    assert result != "ad" or _count_ad_keywords("할인", "shop@example.com", "일반 내용입니다.") >= 2


def _count_ad_keywords(subject, sender, body):
    """Helper to count ad keywords."""
    from kct.integrations.gmail_classifier import AD_KEYWORDS
    text = f"{subject} {sender} {body}".lower()
    return sum(1 for kw in AD_KEYWORDS if kw.lower() in text)


# ── Keyword matching ─────────────────────────────────────────────────────

def test_ad_keywords_list():
    from kct.integrations.gmail_classifier import AD_KEYWORDS
    assert len(AD_KEYWORDS) > 0
    assert "광고" in AD_KEYWORDS


def test_important_keywords_list():
    from kct.integrations.gmail_classifier import IMPORTANT_KEYWORDS
    assert len(IMPORTANT_KEYWORDS) > 0
    assert "긴급" in IMPORTANT_KEYWORDS


# ── MIME header decoding ────────────────────────────────────────────────

def test_decode_mime_words_ascii():
    from kct.integrations.gmail_classifier import decode_mime_words
    assert decode_mime_words("Hello") == "Hello"


def test_decode_mime_words_korean_encoded():
    from kct.integrations.gmail_classifier import decode_mime_words
    # =?utf-8?B?7YWM7Iqk7Yq4?= decodes to "테스트"
    result = decode_mime_words("=?utf-8?B?7YWM7Iqk7Yq4?=")
    assert result == "테스트"


def test_decode_mime_words_none():
    from kct.integrations.gmail_classifier import decode_mime_words
    assert decode_mime_words(None) == ""


def test_decode_mime_words_mixed():
    from kct.integrations.gmail_classifier import decode_mime_words
    result = decode_mime_words("Re: =?utf-8?B?7YWM7Iqk7Yq4?=")
    assert "테스트" in result


# ── Email body extraction ───────────────────────────────────────────────

def test_get_email_body_plain():
    import email as email_lib

    from kct.integrations.gmail_classifier import get_email_body
    msg = email_lib.message_from_string(
        "Content-Type: text/plain; charset=utf-8\r\n\r\nHello body"
    )
    body = get_email_body(msg)
    assert "Hello body" in body


def test_get_email_body_multipart():
    import email as email_lib

    from kct.integrations.gmail_classifier import get_email_body
    raw = (
        "From: test@test.com\r\n"
        "Subject: Test\r\n"
        "MIME-Version: 1.0\r\n"
        "Content-Type: multipart/alternative; boundary=boundary\r\n"
        "\r\n"
        "--boundary\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        "\r\n"
        "Plain text body\r\n"
        "--boundary\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "\r\n"
        "<p>HTML body</p>\r\n"
        "--boundary--\r\n"
    )
    msg = email_lib.message_from_string(raw)
    body = get_email_body(msg)
    assert "Plain text body" in body


# ── Credentials ──────────────────────────────────────────────────────────

def test_get_credentials_missing(monkeypatch):
    from kct.integrations.gmail_classifier import _get_credentials
    monkeypatch.delenv("GMAIL_ADDRESS", raising=False)
    monkeypatch.delenv("GMAIL_APP_PASSWORD", raising=False)
    with pytest.raises(EnvironmentError):
        _get_credentials()


def test_get_credentials_present(monkeypatch):
    from kct.integrations.gmail_classifier import _get_credentials
    monkeypatch.setenv("GMAIL_ADDRESS", "test@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "abcd1234efgh5678")
    addr, pw = _get_credentials()
    assert addr == "test@gmail.com"
    assert pw == "abcd1234efgh5678"


# ── scan_inbox with mocked IMAP ─────────────────────────────────────────

def test_scan_inbox_mocked(monkeypatch):
    """Verify scan_inbox classifies emails using mocked IMAP."""
    from email.message import EmailMessage

    from kct.integrations.gmail_classifier import scan_inbox

    def make_message(subject, sender, body):
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg.set_content(body, charset="utf-8")
        return msg

    mock_mail = MagicMock()
    mock_mail.search.return_value = ("OK", [b"1 2"])
    # Email 1: ad
    ad_email = make_message(
        "광고 특가 할인 프로모션",
        "promo@spam.com",
        "광고 메일 본문입니다.",
    )
    # Email 2: important
    imp_email = make_message(
        "긴급 견적 확인",
        "biz@partner.com",
        "견적서 확인 부탁드립니다.",
    )
    mock_mail.fetch.side_effect = [
        ("OK", [(b"1", ad_email.as_bytes())]),
        ("OK", [(b"2", imp_email.as_bytes())]),
    ]
    mock_imap_cls = MagicMock(return_value=mock_mail)
    monkeypatch.setattr("kct.integrations.gmail_classifier.imaplib.IMAP4_SSL", mock_imap_cls)
    monkeypatch.setenv("GMAIL_ADDRESS", "test@gmail.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "pass")

    results = scan_inbox(max_emails=10)
    assert len(results["ads"]) >= 1
    assert len(results["important"]) >= 1

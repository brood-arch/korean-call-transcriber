from src.pipeline.redact import redact_sensitive_text


def test_redacts_phone_number():
    assert redact_sensitive_text("call 010-1234-5678") == "call 010-****-****"


def test_redacts_email_local_part():
    assert redact_sensitive_text("send to user.name@example.com") == "send to u***@example.com"


def test_redacts_api_key_token_prefixes():
    text = redact_sensitive_text("sk-abcdef123456 key-abcdef123456")
    assert "[REDACTED_TOKEN]" in text
    assert "abcdef123456" not in text


def test_redacts_key_query_param():
    text = redact_sensitive_text("token=abcdefabcdefabcdefabcdefabcdefabcdef")
    assert text == "token=[REDACTED_TOKEN]"


def test_redacts_windows_user_path():
    path = "C:" + "\\Users\\someone\\secret.txt"
    assert redact_sensitive_text(path) == "[REDACTED_PATH]"

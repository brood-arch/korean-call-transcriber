"""Tests for src.pipeline.redact — sensitive text redaction."""

from src.pipeline.redact import redact_sensitive_text


class TestPhoneRedaction:
    def test_mobile_with_hyphens(self):
        assert "010-1234-5678" not in redact_sensitive_text("Call 010-1234-5678")

    def test_mobile_no_hyphens(self):
        assert "01012345678" not in redact_sensitive_text("Call 01012345678 please")

    def test_mobile_with_spaces(self):
        assert "010 1234 5678" not in redact_sensitive_text("Call 010 1234 5678")

    def test_seoul_landline(self):
        result = redact_sensitive_text("Tel 02-123-4567")
        assert "02-123-4567" not in result

    def test_area_code(self):
        result = redact_sensitive_text("Tel 031-123-4567")
        assert "031-123-4567" not in result

    def test_toll_free(self):
        result = redact_sensitive_text("Call 1588-0000")
        assert "1588-0000" not in result

    def test_plus82_prefix(self):
        result = redact_sensitive_text("Call +82-10-1234-5678")
        assert "REDACTED" in result
        assert "10-1234-5678" not in result

    def test_plus82_with_hyphen(self):
        assert "REDACTED" in redact_sensitive_text("+82-10-1234-5678")

    def test_plus82_with_space(self):
        assert "REDACTED" in redact_sensitive_text("+82 10 1234 5678")

    def test_plus82_no_separator(self):
        assert "REDACTED" in redact_sensitive_text("+821012345678")

    def test_normal_number_not_redacted(self):
        text = "Model ABC-1234-5678"
        result = redact_sensitive_text(text)
        assert "ABC-1234-5678" in result


class TestEmailRedaction:
    def test_email_masked(self):
        result = redact_sensitive_text("user@example.com")
        assert "user@example.com" not in result
        assert "@example.com" in result

    def test_long_email(self):
        result = redact_sensitive_text("longuser.name+tag@corp.co.kr")
        assert "@corp.co.kr" in result


class TestTokenRedaction:
    def test_sk_prefix(self):
        result = redact_sensitive_text("key=sk-abc123def456gh")
        assert "sk-abc123" not in result

    def test_key_prefix(self):
        result = redact_sensitive_text("token=key-abcdefghijklmnop")
        assert "key-abcdefghijklmnop" not in result

    def test_bearer_token(self):
        result = redact_sensitive_text("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9")
        assert "eyJhbGci" not in result
        assert "REDACTED" in result


class TestKeyParamRedaction:
    def test_api_key_equals(self):
        result = redact_sensitive_text("api_key=abcdef1234567890abcdef")
        assert "abcdef1234567890" not in result

    def test_password_equals(self):
        result = redact_sensitive_text("password=mysecretpassword123")
        assert "mysecretpassword123" not in result

    def test_bearer_in_text(self):
        result = redact_sensitive_text("Bearer abc123def456ghi789jkl012")
        assert "abc123def456ghi789" not in result


class TestPathRedaction:
    def test_windows_user_path(self):
        result = redact_sensitive_text(r"C:\Users\brood\secrets\key.pem")
        assert r"C:\Users\brood" not in result

    def test_linux_path_not_redacted(self):
        result = redact_sensitive_text("/home/user/data/file.txt")
        assert "/home/user/data/file.txt" in result


class TestEdgeCases:
    def test_empty_string(self):
        assert redact_sensitive_text("") == ""

    def test_none_input(self):
        assert redact_sensitive_text(None) == ""

    def test_no_sensitive_content(self):
        text = "안녕하세요 오늘 회의 관련 통화입니다"
        assert redact_sensitive_text(text) == text

    def test_multiple_sensitive_items(self):
        text = "Call 010-1234-5678, email user@corp.com, key=sk-abc123def456"
        result = redact_sensitive_text(text)
        assert "010-1234-5678" not in result
        assert "user@corp.com" not in result
        assert "sk-abc123def456" not in result

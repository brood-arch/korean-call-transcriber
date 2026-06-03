"""Tests for src.pipeline.utils — atomic writes, compression, JSON helpers."""




class TestAtomicWriteText:
    def test_creates_file(self, tmp_path):
        from src.pipeline.utils import safe_write_text
        target = tmp_path / "test.txt"
        safe_write_text(target, "hello world")
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_creates_parent_dirs(self, tmp_path):
        from src.pipeline.utils import safe_write_text
        target = tmp_path / "deep" / "nested" / "file.txt"
        safe_write_text(target, "content")
        assert target.exists()

    def test_overwrites_existing(self, tmp_path):
        from src.pipeline.utils import safe_write_text
        target = tmp_path / "test.txt"
        safe_write_text(target, "first")
        safe_write_text(target, "second")
        assert target.read_text(encoding="utf-8") == "second"

    def test_no_tmp_file_left(self, tmp_path):
        from src.pipeline.utils import safe_write_text
        target = tmp_path / "test.txt"
        safe_write_text(target, "clean")
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestSafeSaveJson:
    def test_roundtrip(self, tmp_path):
        from src.pipeline.utils import safe_load_json, safe_save_json
        target = tmp_path / "data.json"
        data = {"key": "value", "number": 42, "nested": {"a": [1, 2, 3]}}
        safe_save_json(target, data)
        loaded = safe_load_json(target)
        assert loaded == data

    def test_corrupt_file_returns_default(self, tmp_path):
        from src.pipeline.utils import safe_load_json
        target = tmp_path / "bad.json"
        target.write_text("{broken json", encoding="utf-8")
        result = safe_load_json(target, default={"fallback": True})
        assert result == {"fallback": True}

    def test_missing_file_returns_default(self, tmp_path):
        from src.pipeline.utils import safe_load_json
        target = tmp_path / "missing.json"
        result = safe_load_json(target, default={})
        assert result == {}

    def test_unicode_preserved(self, tmp_path):
        from src.pipeline.utils import safe_load_json, safe_save_json
        target = tmp_path / "korean.json"
        data = {"text": "전표 처리", "company": "오일프라자"}
        safe_save_json(target, data)
        loaded = safe_load_json(target)
        assert loaded["text"] == "전표 처리"


class TestCompressTranscript:
    def test_shortens_long_transcript(self):
        from src.pipeline.utils import compress_transcript
        text = "이것은 테스트입니다. " * 200  # ~4000 chars
        result = compress_transcript(text, budget=500)
        assert len(result) <= 600  # Allow some slack for header/footer

    def test_preserves_short_transcript(self):
        from src.pipeline.utils import compress_transcript
        text = "짧은 텍스트"
        result = compress_transcript(text, budget=5000)
        assert result == text

    def test_empty_input(self):
        from src.pipeline.utils import compress_transcript
        assert compress_transcript("") == ""


class TestFallbackSummary:
    def test_generates_basic_summary(self):
        from src.pipeline.utils import fallback_summary
        text = "안녕하세요. 오늘 회의 관련 통화입니다. 견적 부탁드립니다. 감사합니다."
        result = fallback_summary(text)
        assert isinstance(result, dict)
        assert "summary" in result
        assert "one_line" in result["summary"]

    def test_empty_input(self):
        from src.pipeline.utils import fallback_summary
        result = fallback_summary("")
        assert isinstance(result, dict)
        assert "summary" in result

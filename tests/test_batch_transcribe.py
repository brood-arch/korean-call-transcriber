"""Tests for src.transcribe.batch_transcribe — blacklist, pending selection, speaker mapping."""

import json
from pathlib import Path
from unittest.mock import patch


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_audio(path: Path, size: int = 2048) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


# ---------------------------------------------------------------------------
# Blacklist handling
# ---------------------------------------------------------------------------

class TestBlacklist:
    @patch("src.transcribe.batch_transcribe.BLACKLIST_FILE")
    def test_load_empty_returns_default(self, mock_bl, tmp_path):
        from src.transcribe.batch_transcribe import _load_blacklist
        bl_file = tmp_path / "blacklist.json"
        mock_bl.__str__ = lambda s: str(bl_file)
        # BLACKLIST_FILE is a Path constant, patch it properly
        with patch("src.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
            result = _load_blacklist()
        assert isinstance(result, dict)

    def test_load_with_entries(self, tmp_path):
        from src.transcribe.batch_transcribe import _load_blacklist
        bl_file = tmp_path / "blacklist.json"
        _write_json(bl_file, {"bad_file1": {"failures": 3, "blacklisted_at": "2026-06-03T18:00:00"}})
        with patch("src.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
            result = _load_blacklist()
        assert "bad_file1" in result

    def test_is_blacklisted(self, tmp_path):
        from src.transcribe.batch_transcribe import is_blacklisted
        bl_file = tmp_path / "blacklist.json"
        _write_json(bl_file, {"noise_file": {"failures": 3, "blacklisted_at": "2026-06-03T18:00:00"}})
        with patch("src.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
            assert is_blacklisted("noise_file") is True
            assert is_blacklisted("clean_file") is False


# ---------------------------------------------------------------------------
# Pending file selection
# ---------------------------------------------------------------------------

class TestPendingSelection:
    def test_selects_untranscribed_files(self, tmp_path):
        audio_dir = tmp_path / "audio"
        trans_dir = tmp_path / "transcripts"
        audio_dir.mkdir()
        trans_dir.mkdir()

        _write_audio(audio_dir / "call_001.m4a")
        _write_audio(audio_dir / "call_002.m4a")
        (trans_dir / "call_001.txt").write_text("existing transcript", encoding="utf-8")

        bl_file = tmp_path / "blacklist.json"
        _write_json(bl_file, {})

        from src.transcribe.batch_transcribe import get_pending
        with patch("src.transcribe.batch_transcribe.SOURCE_DIR", audio_dir), \
             patch("src.transcribe.batch_transcribe.OUTPUT_DIR", trans_dir), \
             patch("src.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
            pending = get_pending()

        stems = [f.stem for f in pending]
        assert "call_002" in stems
        assert "call_001" not in stems

    def test_blacklisted_files_excluded(self, tmp_path):
        audio_dir = tmp_path / "audio"
        trans_dir = tmp_path / "transcripts"
        audio_dir.mkdir()
        trans_dir.mkdir()

        _write_audio(audio_dir / "call_good.m4a")
        _write_audio(audio_dir / "call_blocked.m4a")

        bl_file = tmp_path / "blacklist.json"
        _write_json(bl_file, {"call_blocked": {"failures": 3, "blacklisted_at": "2026-06-03T18:00:00"}})

        from src.transcribe.batch_transcribe import get_pending
        with patch("src.transcribe.batch_transcribe.SOURCE_DIR", audio_dir), \
             patch("src.transcribe.batch_transcribe.OUTPUT_DIR", trans_dir), \
             patch("src.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
            pending = get_pending()

        stems = [f.stem for f in pending]
        assert "call_good" in stems
        assert "call_blocked" not in stems


# ---------------------------------------------------------------------------
# Speaker mapping
# ---------------------------------------------------------------------------

class TestSpeakerMapping:
    def test_basic_mapping(self):
        from src.transcribe.batch_transcribe import map_speakers

        segments = [
            {"start": 0.0, "end": 5.0, "text": "안녕하세요", "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "text": "네 반갑습니다", "speaker": "SPEAKER_01"},
        ]
        result = map_speakers(segments, caller_name="홍길동", caller_phone="010-1234-5678")
        assert len(result) == 2
        # One should be tagged as the caller
        callers = [s for s in result if s.get("speaker") == "홍길동"]
        assert len(callers) >= 1

    def test_no_caller_info(self):
        from src.transcribe.batch_transcribe import map_speakers

        segments = [
            {"start": 0.0, "end": 5.0, "text": "대화 내용", "speaker": "SPEAKER_00"},
        ]
        result = map_speakers(segments, caller_name=None, caller_phone=None)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# parse_caller_info
# ---------------------------------------------------------------------------

class TestParseCallerInfo:
    def test_parses_phone_number(self):
        from src.transcribe.batch_transcribe import parse_caller_info
        name, phone = parse_caller_info("통화_01012345678_20260603150000")
        assert phone is not None
        assert "010" in phone

    def test_no_match_returns_none(self):
        from src.transcribe.batch_transcribe import parse_caller_info
        name, phone = parse_caller_info("random_file_name")
        assert name is None
        assert phone is None

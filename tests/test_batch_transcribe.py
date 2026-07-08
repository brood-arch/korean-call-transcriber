"""Tests for kct.transcribe.batch_transcribe — blacklist, pending selection, speaker mapping."""

import argparse
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
    @patch("kct.transcribe.batch_transcribe.BLACKLIST_FILE")
    def test_load_empty_returns_default(self, mock_bl, tmp_path):
        from kct.transcribe.batch_transcribe import _load_blacklist
        bl_file = tmp_path / "blacklist.json"
        mock_bl.__str__ = lambda s: str(bl_file)
        # BLACKLIST_FILE is a Path constant, patch it properly
        with patch("kct.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
            result = _load_blacklist()
        assert isinstance(result, dict)

    def test_load_with_entries(self, tmp_path):
        from kct.transcribe.batch_transcribe import _load_blacklist
        bl_file = tmp_path / "blacklist.json"
        _write_json(bl_file, {"bad_file1": {"failures": 3, "blacklisted_at": "2026-06-03T18:00:00"}})
        with patch("kct.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
            result = _load_blacklist()
        assert "bad_file1" in result

    def test_is_blacklisted(self, tmp_path):
        from kct.transcribe.batch_transcribe import is_blacklisted
        bl_file = tmp_path / "blacklist.json"
        _write_json(bl_file, {"noise_file": {"failures": 3, "blacklisted_at": "2026-06-03T18:00:00"}})
        with patch("kct.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
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

        from kct.transcribe.batch_transcribe import get_pending
        with patch("kct.transcribe.batch_transcribe.SOURCE_DIR", audio_dir), \
             patch("kct.transcribe.batch_transcribe.OUTPUT_DIR", trans_dir), \
             patch("kct.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
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

        from kct.transcribe.batch_transcribe import get_pending
        with patch("kct.transcribe.batch_transcribe.SOURCE_DIR", audio_dir), \
             patch("kct.transcribe.batch_transcribe.OUTPUT_DIR", trans_dir), \
             patch("kct.transcribe.batch_transcribe.BLACKLIST_FILE", bl_file):
            pending = get_pending()

        stems = [f.stem for f in pending]
        assert "call_good" in stems
        assert "call_blocked" not in stems


# ---------------------------------------------------------------------------
# Speaker mapping
# ---------------------------------------------------------------------------

class TestSpeakerMapping:
    def test_basic_mapping(self):
        from kct.transcribe.batch_transcribe import map_speakers

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
        from kct.transcribe.batch_transcribe import map_speakers

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
        from kct.transcribe.batch_transcribe import parse_caller_info
        name, phone = parse_caller_info("통화_01012345678_20260603150000")
        assert phone is not None
        assert "010" in phone

    def test_no_match_returns_none(self):
        from kct.transcribe.batch_transcribe import parse_caller_info
        name, phone = parse_caller_info("random_file_name")
        assert name is None
        assert phone is None


def test_process_single_audio_success_path_does_not_pass_flush_to_logger(tmp_path, monkeypatch):
    """Regression: logging.Logger.info does not accept flush=True."""
    from kct.transcribe import batch_transcribe as bt

    audio = tmp_path / "call_001.m4a"
    _write_audio(audio)
    output_dir = tmp_path / "transcripts"
    output_dir.mkdir()
    out_path = output_dir / "call_001.txt"

    monkeypatch.setattr(bt, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(bt, "get_audio_duration", lambda _path: 30.0)
    monkeypatch.setattr(
        bt,
        "_perform_transcription",
        lambda *_args: ([{"start": 0.0, "end": 1.0, "text": "안녕하세요 " * 80}], 30.0, {}, False),
    )
    monkeypatch.setattr(bt, "apply_corrections", lambda text, source="": (text, []))
    monkeypatch.setattr(bt, "log_quality", lambda *args, **kwargs: None)

    assert bt._process_single_audio(audio, out_path, argparse.Namespace(), 1, 1) is True
    assert out_path.exists()

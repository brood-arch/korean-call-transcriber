"""Tests for src.extract.extract_all — pipeline orchestration (mocked API)."""

from pathlib import Path
from unittest.mock import patch

import pytest


def _write_transcript(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_args(tmp_path, **overrides):
    """Create a minimal args namespace for IntegratedPipeline."""
    import argparse
    defaults = dict(
        base_dir=str(tmp_path / "transcripts"),
        state_dir=str(tmp_path / "state"),
        batch_size=5,
        api_delay=0.0,
        json=False,
        today=False,
        start_batch=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

class TestFileSelection:
    def test_finds_txt_files(self, tmp_path):
        from src.extract.extract_all import IntegratedPipeline

        trans_dir = tmp_path / "transcripts"
        _write_transcript(trans_dir / "call_20260603_150000.txt", "안녕하세요 오늘 회의 관련 통화입니다." * 10)
        _write_transcript(trans_dir / "call_20260603_160000.txt", "네 알겠습니다 내일 미팅 잡아주세요." * 10)

        args = _make_args(tmp_path)
        pipe = IntegratedPipeline(args)
        files = pipe.get_transcription_files()
        assert len(files) == 2

    def test_skips_empty_files(self, tmp_path):
        from src.extract.extract_all import IntegratedPipeline

        trans_dir = tmp_path / "transcripts"
        _write_transcript(trans_dir / "empty.txt", "")
        _write_transcript(trans_dir / "tiny.txt", "x")
        _write_transcript(trans_dir / "good.txt", "충분히 긴 내용입니다." * 10)

        args = _make_args(tmp_path)
        pipe = IntegratedPipeline(args)
        files = pipe.get_transcription_files()
        assert len(files) == 1
        assert files[0].stem == "good"

    def test_skips_already_processed(self, tmp_path):
        from src.extract.extract_all import IntegratedPipeline
        from src.extract.state import compute_file_hash, save_processed_index

        trans_dir = tmp_path / "transcripts"
        _write_transcript(trans_dir / "call_20260603.txt", "이미 처리된 통화 내용입니다." * 10)

        # Create pipeline and manually set its state dir
        args = _make_args(tmp_path)
        pipe = IntegratedPipeline(args)

        # Compute hash and save as processed in the pipeline's index file
        fhash = compute_file_hash(trans_dir / "call_20260603.txt")
        pipe.state_dir.mkdir(parents=True, exist_ok=True)
        save_processed_index(pipe.processed_index_file, {"call_20260603": fhash})

        files = pipe.get_transcription_files()
        assert len(files) == 0

    def test_today_mode_filters(self, tmp_path):
        from src.extract.extract_all import IntegratedPipeline

        trans_dir = tmp_path / "transcripts"
        _write_transcript(trans_dir / "call_20260603_150000.txt", "오늘 통화" * 10)
        _write_transcript(trans_dir / "call_20260602_150000.txt", "어제 통화" * 10)

        args = _make_args(tmp_path, today=True)
        with patch("src.extract.extract_all.datetime") as mock_dt:
            from datetime import datetime as real_dt
            mock_dt.now.side_effect = lambda tz=None: real_dt(2026, 6, 3, 15, 0, 0)
            mock_dt.side_effect = lambda *a, **k: real_dt(*a, **k)
            pipe = IntegratedPipeline(args)
            files = pipe.get_transcription_files()

        # Should only include today's file
        assert any("20260603" in f.stem for f in files)


# ---------------------------------------------------------------------------
# Pipeline run with mocked API
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_dry_run_finds_files_but_no_api(self, tmp_path):
        """Verify file selection works without calling the LLM API."""
        from src.extract.extract_all import IntegratedPipeline

        trans_dir = tmp_path / "transcripts"
        _write_transcript(trans_dir / "call_001.txt", "테스트 통화 내용입니다." * 10)
        _write_transcript(trans_dir / "call_002.txt", "또 다른 통화 내용입니다." * 10)

        args = _make_args(tmp_path)
        pipe = IntegratedPipeline(args)
        files = pipe.get_transcription_files()
        assert len(files) == 2

    @patch("src.extract.extract_all.call_llm_extract")
    def test_single_file_extraction(self, mock_llm, tmp_path):
        from src.extract.extract_all import IntegratedPipeline

        mock_llm.return_value = {
            "summary": {"one_line": "테스트 통화 요약"},
            "todos": [{"title": "견적 보내기", "owner": "me", "priority": "high"}],
            "appointments": [],
            "entities": [],
            "products": [],
            "money": [],
            "risks": [],
            "corrections": [],
        }

        trans_dir = tmp_path / "transcripts"
        _write_transcript(trans_dir / "call_001.txt", "테스트 통화 내용입니다." * 10)

        args = _make_args(tmp_path, batch_size=1)
        pipe = IntegratedPipeline(args)

        # Mock fast_score to always process
        with patch("src.extract.extract_all._get_fast_score", return_value=lambda t: {"should_process": True, "score": 1.0, "band": "definite_keep"}):
            with patch("src.extract.extract_all.setup_langfuse", return_value=None):
                with patch("src.extract.extract_all.get_llm_config", return_value={"api_key": "test-key"}):
                    pipe._lf_available = False
                    with pytest.raises(SystemExit) as exc:
                        pipe.run()
                    assert exc.value.code == 0

        assert mock_llm.called
        assert pipe.stats["summary"] == 1
        assert pipe.stats["todos"] == 1

    @patch("src.extract.extract_all.call_llm_extract")
    def test_fallback_on_api_failure(self, mock_llm, tmp_path):
        from src.extract.extract_all import IntegratedPipeline

        mock_llm.return_value = None  # API failure

        trans_dir = tmp_path / "transcripts"
        _write_transcript(trans_dir / "call_fail.txt", "API 실패 테스트 통화 내용입니다." * 10)

        args = _make_args(tmp_path, batch_size=1)
        pipe = IntegratedPipeline(args)

        with patch("src.extract.extract_all._get_fast_score", return_value=lambda t: {"should_process": True}):
            with patch("src.extract.extract_all.setup_langfuse", return_value=None):
                with patch("src.extract.extract_all.get_llm_config", return_value={"api_key": "test-key"}):
                    pipe._lf_available = False
                    with pytest.raises(SystemExit) as exc:
                        pipe.run()
                    assert exc.value.code == 0

        assert pipe.stats.get("fallbacks", 0) == 1

    @patch("src.extract.extract_all.call_llm_extract")
    def test_fast_score_skip(self, mock_llm, tmp_path):
        from src.extract.extract_all import IntegratedPipeline

        trans_dir = tmp_path / "transcripts"
        _write_transcript(trans_dir / "short_call.txt", "짧은 통화" * 10)

        args = _make_args(tmp_path, batch_size=1)
        pipe = IntegratedPipeline(args)

        # fast_score says "skip"
        with patch("src.extract.extract_all._get_fast_score", return_value=lambda t: {"should_process": False, "score": 0.1, "band": "definite_drop"}):
            with patch("src.extract.extract_all.setup_langfuse", return_value=None):
                with patch("src.extract.extract_all.get_llm_config", return_value={"api_key": "test-key"}):
                    pipe._lf_available = False
                    with pytest.raises(SystemExit) as exc:
                        pipe.run()
                    assert exc.value.code == 0

        # LLM should NOT be called for dropped files
        assert not mock_llm.called
        assert pipe.stats.get("fast_score_dropped", 0) == 1

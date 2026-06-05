"""Tests for kct.transcribe modules — align_worker, batch_diarize, worker."""


class TestAlignWorker:
    def test_import_and_main_exists(self):
        import pytest

        pytest.importorskip("torch", reason="align_worker requires torch")
        from kct.transcribe import align_worker
        assert hasattr(align_worker, "main")
        assert hasattr(align_worker, "subprocess")


class TestBatchDiarize:
    def test_import_and_main_exists(self):
        from kct.transcribe import batch_diarize
        assert hasattr(batch_diarize, "main")
        assert hasattr(batch_diarize, "find_missing")

    def test_is_wsl_callable(self):
        from kct.pipeline.paths import is_wsl
        result = is_wsl()
        assert isinstance(result, bool)


class TestWorker:
    def test_import_and_main_exists(self):
        from kct.transcribe import worker
        assert hasattr(worker, "main")
        assert hasattr(worker, "transcribe_file")

    def test_get_audio_duration_missing(self):
        from kct.transcribe.worker import get_audio_duration
        result = get_audio_duration("/nonexistent/file.m4a")
        assert result is None or result == 0

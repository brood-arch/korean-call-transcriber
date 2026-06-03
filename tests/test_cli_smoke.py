"""Smoke tests for documented module entry points."""

import subprocess
import sys


def run_help(module: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_extract_all_help_runs():
    result = run_help("src.extract.extract_all")
    assert result.returncode == 0, result.stderr
    assert "Integrated LLM Extraction Pipeline" in result.stdout


def test_batch_transcribe_help_runs():
    result = run_help("src.transcribe.batch_transcribe")
    assert result.returncode == 0, result.stderr
    assert "Batch transcription with WhisperX" in result.stdout


def test_email_todo_extract_help_runs():
    result = run_help("src.integrations.email_todo_extract")
    assert result.returncode == 0, result.stderr
    assert "Email TODO extraction" in result.stdout

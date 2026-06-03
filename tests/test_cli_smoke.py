"""Smoke tests for documented module entry points."""

import subprocess
import sys
import tomllib
from pathlib import Path


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


def test_pyproject_exposes_console_scripts():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    version = data["project"]["version"]
    assert version, "version must be set in pyproject.toml"
    assert tuple(int(x) for x in version.split(".")) >= (0, 4, 0), f"version {version} < 0.4.0"
    assert scripts["kct-transcribe"] == "src.transcribe.batch_transcribe:main"
    assert scripts["kct-extract"] == "src.extract.extract_all:main"
    assert scripts["kct-health"] == "src.queue.gap_analyzer:main"
    assert scripts["kct-sync-obsidian"] == "src.sync.sync_obsidian:main"

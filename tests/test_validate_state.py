"""Tests for kct.pipeline.validate_state — file existence, staleness, integrity checks."""

import json
import time

import pytest


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setenv("KCT_STATE_DIR", str(tmp_path))
    import kct.pipeline.validate_state as vs
    monkeypatch.setattr(vs, "_STATE_DIR", tmp_path)


# ── check_file ───────────────────────────────────────────────────────────

def test_check_file_missing(tmp_path):
    from kct.pipeline.validate_state import check_file
    result = check_file("pipeline_state.json", state_dir=tmp_path)
    assert result["exists"] is False
    assert result["stale"] is True


def test_check_file_exists_and_fresh(tmp_path):
    from kct.pipeline.validate_state import check_file
    fpath = tmp_path / "pipeline_state.json"
    fpath.write_text(json.dumps({"key": "val"}), encoding="utf-8")
    result = check_file("pipeline_state.json", state_dir=tmp_path)
    assert result["exists"] is True
    assert result["stale"] is False
    assert result["parse_ok"] is True


def test_check_file_stale(tmp_path):
    from kct.pipeline.validate_state import check_file
    fpath = tmp_path / "pipeline_state.json"
    fpath.write_text(json.dumps({"key": "val"}), encoding="utf-8")
    # Set mtime to 100 hours ago
    old_time = time.time() - 100 * 3600
    import os
    os.utime(fpath, (old_time, old_time))
    result = check_file("pipeline_state.json", state_dir=tmp_path)
    assert result["exists"] is True
    assert result["stale"] is True
    assert result["age_hours"] > 72


def test_check_file_corrupt_json(tmp_path):
    from kct.pipeline.validate_state import check_file
    fpath = tmp_path / "persistent_todos.json"
    fpath.write_text("{invalid json content", encoding="utf-8")
    result = check_file("persistent_todos.json", state_dir=tmp_path)
    assert result["exists"] is True
    assert result["parse_ok"] is False


def test_check_file_null_bytes(tmp_path):
    from kct.pipeline.validate_state import check_file
    fpath = tmp_path / "persistent_todos.json"
    fpath.write_bytes(b'{"key": "val\x00ue"}')
    result = check_file("persistent_todos.json", state_dir=tmp_path)
    assert result["exists"] is True
    assert result["parse_ok"] is False


def test_check_file_empty_json_ok(tmp_path):
    from kct.pipeline.validate_state import check_file
    fpath = tmp_path / "pipeline_state.json"
    fpath.write_text("{}", encoding="utf-8")
    result = check_file("pipeline_state.json", state_dir=tmp_path)
    assert result["parse_ok"] is True


# ── check_all ────────────────────────────────────────────────────────────

def test_check_all_missing_files(tmp_path):
    from kct.pipeline.validate_state import check_all
    report = check_all(state_dir=tmp_path)
    assert report["ok"] is False
    assert len(report["files"]) > 0
    assert all(not f["exists"] for f in report["files"])


def test_check_all_fresh_files(tmp_path):
    from kct.pipeline.validate_state import EXPECTED_FILES, check_all
    for name in EXPECTED_FILES:
        fpath = tmp_path / name
        if name.endswith(".json"):
            fpath.write_text(json.dumps({"key": "val"}), encoding="utf-8")
        else:
            fpath.write_text("line1\nline2\n", encoding="utf-8")
    report = check_all(state_dir=tmp_path)
    assert report["ok"] is True
    assert all(f["exists"] for f in report["files"])


def test_check_all_has_timestamp(tmp_path):
    from kct.pipeline.validate_state import check_all
    report = check_all(state_dir=tmp_path)
    assert "checked_at" in report
    assert "202" in report["checked_at"]  # ISO timestamp starts with year


# ── Staleness thresholds ────────────────────────────────────────────────

def test_stale_threshold_persistent_todos(tmp_path):
    from kct.pipeline.validate_state import STALE_THRESHOLDS
    assert STALE_THRESHOLDS["persistent_todos.json"] == 48


def test_stale_threshold_events(tmp_path):
    from kct.pipeline.validate_state import STALE_THRESHOLDS
    assert STALE_THRESHOLDS["events.jsonl"] == 168  # 1 week


# ── Non-JSON files ──────────────────────────────────────────────────────

def test_check_jsonl_file(tmp_path):
    from kct.pipeline.validate_state import check_file
    fpath = tmp_path / "events.jsonl"
    fpath.write_text('{"event": "one"}\n{"event": "two"}\n', encoding="utf-8")
    result = check_file("events.jsonl", state_dir=tmp_path)
    assert result["exists"] is True
    # jsonl files don't get JSON parse check (only .json files do)
    assert result.get("parse_ok", True) is True


# ── Fix mode ────────────────────────────────────────────────────────────

def test_check_file_with_fix_flag(tmp_path):
    """Fix mode currently only flags; verify it doesn't crash."""
    from kct.pipeline.validate_state import check_file
    fpath = tmp_path / "pipeline_state.json"
    fpath.write_text(json.dumps({"ok": True}), encoding="utf-8")
    result = check_file("pipeline_state.json", state_dir=tmp_path, fix=True)
    assert result["exists"] is True

"""Regression tests for Hermes active runner lock and dependency behavior."""

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_active_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "hermes_transcription_active_run.py"
    spec = importlib.util.spec_from_file_location("hermes_transcription_active_run", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acquire_lock_reclaims_stale_lock_with_atomic_tombstone(tmp_path, monkeypatch):
    ar = _load_active_runner()
    state_dir = tmp_path / "state"
    lock_dir = state_dir / "hermes_transcription_pipeline.lock"
    summary_path = state_dir / "active_run_summary.json"
    lock_dir.mkdir(parents=True)
    old = datetime.now(timezone(timedelta(hours=9))) - timedelta(hours=2)
    (lock_dir / "owner.json").write_text(
        json.dumps({"run_id": "old", "pid": 999999, "started_at": old.isoformat(timespec="seconds")}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ar, "STATE_DIR", state_dir)
    monkeypatch.setattr(ar, "LOCK_DIR", lock_dir)
    monkeypatch.setattr(ar, "SUMMARY_PATH", summary_path)

    assert ar.acquire_lock("new", stale_seconds=1) is True
    owner = json.loads((lock_dir / "owner.json").read_text(encoding="utf-8"))
    assert owner["run_id"] == "new"
    assert owner["reclaimed_stale"]["run_id"] == "old"
    assert not list(state_dir.glob("*.stale.*"))


def test_main_blocks_downstream_after_upstream_failure(tmp_path, monkeypatch):
    ar = _load_active_runner()
    monkeypatch.setattr(ar, "acquire_lock", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(ar, "release_lock", lambda: None)
    monkeypatch.setattr(ar, "get_windows_task_states", lambda _run_id: {name: "Disabled" for name in ar.ACTIVE_WINDOWS_TASKS})
    monkeypatch.setattr(ar, "emit_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(ar, "append_log", lambda row: None)
    monkeypatch.setattr(ar, "safe_write_json", lambda path, payload: None)
    monkeypatch.setattr(ar, "_mem_snapshot", lambda: {"vram_pct": 0})
    monkeypatch.setattr(ar, "_record_step_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(ar, "_check_error_pause", lambda _name: None)

    called = []

    def fake_run_cmd(name, argv, *, timeout, run_id):
        called.append(name)
        return {
            "step": name,
            "returncode": 1 if name == "call_recordings_automation" else 0,
            "duration_seconds": 0,
            "stdout_tail": "",
            "stderr_tail": "boom" if name == "call_recordings_automation" else "",
        }

    monkeypatch.setattr(ar, "run_cmd", fake_run_cmd)

    rc = ar.main(["--skip-shared-events", "--skip-mempalace", "--skip-email-archive"])

    assert rc == 1
    assert "call_recordings_automation" in called
    assert "batch_diarize" in called
    assert "build_chroma_index" not in called
    assert "extract_all_today" not in called
    assert "sync_transcripts_to_obsidian" not in called

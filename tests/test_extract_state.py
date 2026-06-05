"""Tests for kct.extract.state — processed index, checkpoints, TODO sync."""

import json
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Processed index
# ---------------------------------------------------------------------------

class TestProcessedIndex:
    def test_load_empty_returns_empty_dict(self, tmp_path):
        from kct.extract.state import load_processed_index

        idx_file = tmp_path / "processed.json"
        assert load_processed_index(idx_file) == {}

    def test_load_existing_index(self, tmp_path):
        from kct.extract.state import load_processed_index

        idx_file = tmp_path / "processed.json"
        _write_json(idx_file, {"file_a.txt": "hash1", "file_b.txt": "hash2"})
        result = load_processed_index(idx_file)
        assert result == {"file_a.txt": "hash1", "file_b.txt": "hash2"}

    def test_load_corrupt_json_returns_empty(self, tmp_path):
        from kct.extract.state import load_processed_index

        idx_file = tmp_path / "processed.json"
        idx_file.write_text("{bad json", encoding="utf-8")
        result = load_processed_index(idx_file)
        assert result == {}

    def test_save_and_reload(self, tmp_path):
        from kct.extract.state import load_processed_index, save_processed_index

        idx_file = tmp_path / "processed.json"
        data = {"file_x.txt": "hash_x"}
        save_processed_index(idx_file, data)
        assert load_processed_index(idx_file) == data


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_load_no_checkpoint_returns_zero(self, tmp_path):
        from kct.extract.state import load_checkpoint

        cp_file = tmp_path / "checkpoint.json"
        assert load_checkpoint(cp_file) == 0

    def test_load_returns_next_batch(self, tmp_path):
        from kct.extract.state import load_checkpoint

        cp_file = tmp_path / "checkpoint.json"
        _write_json(cp_file, {"last_completed_batch": 3, "last_updated": "2026-06-03T12:00:00+09:00"})
        assert load_checkpoint(cp_file) == 4

    def test_today_only_resets_if_stale(self, tmp_path):
        from kct.extract.state import load_checkpoint

        cp_file = tmp_path / "checkpoint.json"
        # Stale: yesterday's checkpoint
        _write_json(cp_file, {"last_completed_batch": 5, "last_updated": "2026-06-02T12:00:00+09:00"})
        with patch("kct.extract.state.datetime") as mock_dt:
            from datetime import datetime as real_dt
            mock_dt.now.return_value = real_dt(2026, 6, 3, 10, 0, 0)
            mock_dt.side_effect = lambda *a, **k: real_dt(*a, **k)
            result = load_checkpoint(cp_file, today_only=True)
        assert result == 0

    def test_today_only_continues_if_fresh(self, tmp_path):
        from kct.extract.state import load_checkpoint

        cp_file = tmp_path / "checkpoint.json"
        _write_json(cp_file, {"last_completed_batch": 2, "last_updated": "2026-06-03T08:00:00+09:00"})
        with patch("kct.extract.state.datetime") as mock_dt:
            from datetime import datetime as real_dt
            mock_dt.now.return_value = real_dt(2026, 6, 3, 10, 0, 0)
            mock_dt.side_effect = lambda *a, **k: real_dt(*a, **k)
            result = load_checkpoint(cp_file, today_only=True)
        assert result == 3

    def test_save_and_load_roundtrip(self, tmp_path):
        from kct.extract.state import load_checkpoint, save_checkpoint

        cp_file = tmp_path / "checkpoint.json"
        save_checkpoint(cp_file, batch_idx=7, total=10, stats={"ok": 5}, run_id="r1")
        assert load_checkpoint(cp_file) == 8


# ---------------------------------------------------------------------------
# Batch result
# ---------------------------------------------------------------------------

class TestBatchResult:
    def test_save_batch_result_creates_file(self, tmp_path):
        from kct.extract.state import save_batch_result

        save_batch_result(
            tmp_path, batch_idx=0,
            batch_files=[Path("a.txt"), Path("b.txt")],
            results=[{"file": "a.txt", "status": "ok"}],
            errors=[], status="complete", run_id="r1",
        )
        result_file = tmp_path / "batch_0000.json"
        assert result_file.exists()
        data = _read_json(result_file)
        assert data["batch_index"] == 0
        assert data["status"] == "complete"
        assert len(data["files"]) == 2


# ---------------------------------------------------------------------------
# Notification state
# ---------------------------------------------------------------------------

class TestNotificationState:
    def test_load_empty_creates_buckets(self, tmp_path):
        from kct.extract.state import load_notification_state

        ns_file = tmp_path / "notif_state.json"
        state = load_notification_state(ns_file)
        assert "notified_todos" in state
        assert "notified_appointments" in state
        assert "calendar_drafts" in state

    def test_save_and_reload(self, tmp_path):
        from kct.extract.state import load_notification_state, save_notification_state

        ns_file = tmp_path / "notif_state.json"
        state = load_notification_state(ns_file)
        state["notified_todos"]["key1"] = {"title": "test"}
        save_notification_state(ns_file, state)
        reloaded = load_notification_state(ns_file)
        assert "key1" in reloaded["notified_todos"]


# ---------------------------------------------------------------------------
# Appointment key dedup
# ---------------------------------------------------------------------------

class TestAppointmentKey:
    def test_key_format(self):
        from kct.extract.state import appointment_key

        key = appointment_key({"title": "미팅", "date": "2026-06-05", "time": "14:00", "source": "통화_20260603"})
        assert "미팅" in key
        assert "2026-06-05" in key

    def test_key_differs_by_date(self):
        from kct.extract.state import appointment_key

        k1 = appointment_key({"title": "미팅", "date": "2026-06-05", "time": "14:00"})
        k2 = appointment_key({"title": "미팅", "date": "2026-06-06", "time": "14:00"})
        assert k1 != k2

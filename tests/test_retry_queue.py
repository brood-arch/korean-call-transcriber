from datetime import timedelta, timezone
from pathlib import Path

import pytest

KST = timezone(timedelta(hours=9))


def _report(tmp_path: Path):
    audio = tmp_path / "audio"
    transcripts = tmp_path / "transcripts"
    audio.mkdir()
    transcripts.mkdir()
    stems = {
        "missing": "missing_call_20260512010101",
        "diarize": "diarize_call_20260512020202",
        "entity": "entity_call_20260512030303",
        "rag": "index_call_20260512040404",
        "obsidian": "sync_call_20260512050505",
        "blacklisted": "excluded_call_20260512060606",
    }
    for stem in stems.values():
        (audio / f"{stem}.m4a").write_bytes(b"x" * 2048)
        (transcripts / f"{stem}.txt").write_text("전사본", encoding="utf-8")
    return {
        "schema_version": 1,
        "generated_at": "omitted_for_deterministic_reruns",
        "workspace": str(tmp_path),
        "source_dir": str(audio),
        "transcript_dir": str(transcripts),
        "cause_files": {
            "missing_transcript": [
                {
                    "file": f"{stems['missing']}.m4a",
                    "stem": stems["missing"],
                    "reason": "missing_transcript",
                    "priority": "P0",
                    "source_path": str(
                        audio / f"{stems['missing']}.m4a"
                    ),
                }
            ],
            "transcription_failed": [],
            "diarization_failed": [
                {
                    "file": f"{stems['diarize']}.m4a",
                    "stem": stems["diarize"],
                    "reason": "diarization_failed",
                    "priority": "P2",
                }
            ],
            "entity_pending": [
                {
                    "file": f"{stems['entity']}.txt",
                    "stem": stems["entity"],
                    "reason": "entity_pending",
                    "priority": "P2",
                }
            ],
            "rag_pending": [
                {"file": f"{stems['rag']}.txt", "stem": stems["rag"], "reason": "rag_pending", "priority": "P2"}
            ],
            "obsidian_pending": [
                {
                    "file": f"{stems['obsidian']}.txt",
                    "stem": stems["obsidian"],
                    "reason": "obsidian_pending",
                    "priority": "P2",
                }
            ],
            "blacklisted": [
                {
                    "file": f"{stems['blacklisted']}.m4a",
                    "stem": stems["blacklisted"],
                    "reason": "blacklisted",
                    "priority": "P3",
                }
            ],
            "derived_excluded": [],
            "missing_sync": [],
        },
    }


def test_build_queue_from_gap_report_maps_actionable_reasons_and_excludes_holds(tmp_path):
    from src.queue import retry_queue as q

    queue = q.build_queue_from_report(_report(tmp_path), now="2026-05-12T10:00:00+09:00")

    assert [entry["next_action"] for entry in queue] == ["transcribe", "diarize", "entity", "rag", "obsidian"]
    assert all(entry["status"] == "pending" for entry in queue)
    assert all(entry["attempts"] == 0 for entry in queue)
    assert all(entry["last_error"] is None for entry in queue)
    assert all(entry["terminal_failure"] is False for entry in queue)
    assert all(entry["schema_version"] == 1 for entry in queue)
    assert {entry["reason"] for entry in queue}.isdisjoint({"blacklisted", "derived_excluded"})
    assert queue[0]["queue_id"].startswith("rq_")


def test_jsonl_round_trip_and_atomic_rewrite_preserves_entries_on_failure(tmp_path, monkeypatch):
    from src.queue import retry_queue as q

    path = tmp_path / "transcription_retry_queue.jsonl"
    entries = q.build_queue_from_report(_report(tmp_path), now="2026-05-12T10:00:00+09:00")[:2]
    q.write_queue(path, entries)
    assert q.load_queue(path) == entries

    original = path.read_text(encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        q.write_queue(path, entries[:1])
    assert path.read_text(encoding="utf-8") == original


def test_dry_run_worker_reports_commands_without_mutating_queue_or_backing_up(tmp_path):
    from src.queue import retry_queue as q

    queue_path = tmp_path / "transcription_retry_queue.jsonl"
    entries = q.build_queue_from_report(_report(tmp_path), now="2026-05-12T10:00:00+09:00")
    q.write_queue(queue_path, entries)
    before = queue_path.read_text(encoding="utf-8")

    result = q.run_worker(
        queue_path=queue_path, workspace=tmp_path,
        dry_run=True, limit=3,
        now="2026-05-12T10:05:00+09:00",
    )

    assert result["dry_run"] is True
    assert result["selected"] == 3
    argv_text = " ".join(result["commands"][0]["argv"])
    assert "src.transcribe.batch_transcribe" in argv_text
    assert "--file" in argv_text
    assert queue_path.read_text(encoding="utf-8") == before
    assert not list((tmp_path / "backup").glob("**/*transcription_retry_queue*"))


def test_command_for_entry_supports_native_and_wsl_shapes(tmp_path):
    from src.queue import retry_queue as q

    entry = q.build_queue_from_report(_report(tmp_path), now="2026-05-12T10:00:00+09:00")[0]

    native = q.command_for_entry(entry, tmp_path, running_on_wsl=False)
    assert native[1:4] == ["-m", "src.transcribe.batch_transcribe", "--file"]

    wsl = q.command_for_entry(entry, tmp_path, running_on_wsl=True)
    wsl_text = " ".join(wsl)
    assert wsl[:2] == ["/mnt/c/Windows/System32/cmd.exe", "/c"]
    assert "src.transcribe.batch_transcribe" in wsl_text
    assert "--file" in wsl_text


def test_success_history_redacts_sensitive_output():
    from src.queue import retry_queue as q

    entry = {"history": []}
    q.mark_success(
        entry,
        now="2026-05-12T10:00:00+09:00",
        argv=["python"],
        stdout="sent to person@example.com with token=abc123456789012345",
        stderr="call 010-1234-5678 failed",
    )

    history = entry["history"][-1]
    assert "person@example.com" not in history["stdout_tail"]
    assert "abc123456789012345" not in history["stdout_tail"]
    assert "010-1234-5678" not in history["stderr_tail"]
    assert "p***@example.com" in history["stdout_tail"]
    assert "REDACTED" in history["stderr_tail"]


def test_worker_records_failure_attempt_backoff_log_and_backup(tmp_path):
    from src.queue import retry_queue as q

    queue_path = tmp_path / "transcription_retry_queue.jsonl"
    entries = q.build_queue_from_report(_report(tmp_path), now="2026-05-12T10:00:00+09:00")[:1]
    q.write_queue(queue_path, entries)

    def fake_runner(argv, cwd, timeout, text, capture_output):
        class Result:
            returncode = 7
            stdout = "out"
            stderr = "err"
        return Result()

    result = q.run_worker(
        queue_path=queue_path, workspace=tmp_path,
        dry_run=False, limit=1,
        now="2026-05-12T10:10:00+09:00",
        runner=fake_runner,
    )

    updated = q.load_queue(queue_path)[0]
    assert result["failed"] == 1
    assert updated["attempts"] == 1
    assert updated["status"] == "pending"
    assert updated["last_error"] == "exit 7: err"
    assert updated["next_retry_at"] > "2026-05-12T10:10:00+09:00"
    assert updated["terminal_failure"] is False
    assert updated["history"][-1]["outcome"] == "failed"
    assert list((tmp_path / "backup").glob("retry_queue_*/transcription_retry_queue.jsonl"))
    assert (tmp_path / "logs" / "transcription_retry_worker.jsonl").exists()


def test_worker_marks_terminal_failure_after_max_attempts(tmp_path):
    from src.queue import retry_queue as q

    queue_path = tmp_path / "transcription_retry_queue.jsonl"
    entry = q.build_queue_from_report(_report(tmp_path), now="2026-05-12T10:00:00+09:00")[0]
    entry["attempts"] = entry["max_attempts"] - 1
    q.write_queue(queue_path, [entry])

    def fake_runner(argv, cwd, timeout, text, capture_output):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "still broken"
        return Result()

    q.run_worker(
        queue_path=queue_path, workspace=tmp_path,
        dry_run=False, limit=1,
        now="2026-05-12T10:10:00+09:00",
        runner=fake_runner,
    )
    updated = q.load_queue(queue_path)[0]
    assert updated["status"] == "terminal_failure"
    assert updated["terminal_failure"] is True
    assert updated["next_retry_at"] is None




#!/usr/bin/env python3
"""Retry queue generator and worker for transcription pipeline recovery.

This module converts ``transcription_gap_analyzer.py`` JSON reports into an
atomic JSONL retry queue and processes due entries by ``next_action``.

Queue file: ``memory/state/transcription_retry_queue.jsonl``
Log file:   ``logs/transcription_retry_worker.jsonl``

Schema v1 JSONL entry (one object per line):
  schema_version: 1
  queue_id: deterministic rq_<sha256[:16]> for stem/reason/action
  created_at / updated_at: ISO-8601 KST timestamp
  stem, file, source_path, transcript_path
  reason: gap analyzer cause
  next_action: transcribe | diarize | entity | rag | obsidian
  priority: P0..P4
  status: pending | running | succeeded | terminal_failure
  attempts, max_attempts, last_error, next_retry_at, terminal_failure
  history: append-only attempt outcomes

The worker is intentionally conservative: dry-run is default in the CLI unless
``--execute`` is supplied.  Actual execution always writes a timestamped backup
of the queue before launching the first subprocess and writes JSONL audit logs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from kct.config import LOG_DIR, STATE_DIR, WORKSPACE
from kct.pipeline.paths import is_wsl
from kct.pipeline.utils import redact_sensitive_text, safe_write_text

KST = timezone(timedelta(hours=9))

log = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = STATE_DIR / "transcription_retry_queue.jsonl"
DEFAULT_REPORT_PATH = WORKSPACE / "reports" / "transcription_health_taxonomy.json"
DEFAULT_LOG_PATH = LOG_DIR / "transcription_retry_worker.jsonl"
DEFAULT_BACKUP_ROOT = STATE_DIR / "backups"
DEFAULT_LOCK_PATH = STATE_DIR / "transcription_retry_queue.lock"

ACTIONABLE_REASON_TO_ACTION = {
    "missing_transcript": "transcribe",
    "transcription_failed": "transcribe",
    "diarization_failed": "diarize",
    "entity_pending": "entity",
    "rag_pending": "rag",
    "obsidian_pending": "obsidian",
}

ACTION_ORDER = {"transcribe": 0, "diarize": 1, "entity": 2, "rag": 3, "obsidian": 4}
STATUS_TERMINAL = {"succeeded", "terminal_failure"}


class QueueError(RuntimeError):
    pass


def now_iso() -> str:
    """현재 KST 시간을 ISO-8601 문자열로 반환."""
    return datetime.now(KST).isoformat(timespec="seconds")


def parse_iso(value: str) -> datetime:
    """ISO-8601 문자열을 datetime 객체로 파싱."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def coerce_now(value: str | None = None) -> str:
    """값이 없으면 현재 시간을, 있으면 그대로 반환."""
    return value or now_iso()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """JSONL 파일에 행을 추가."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def queue_id_for(stem: str, reason: str, next_action: str) -> str:
    """stem/reason/action 조합으로 결정론적 큐 ID를 생성."""
    raw = f"{stem}\0{reason}\0{next_action}".encode("utf-8")
    return "rq_" + hashlib.sha256(raw).hexdigest()[:16]


def _path_from_row(row: dict[str, Any], report: dict[str, Any], *, kind: str) -> str | None:
    key = "source_path" if kind == "audio" else "transcript_path"
    if row.get(key):
        return str(row[key])
    base_key = "source_dir" if kind == "audio" else "transcript_dir"
    suffix = ".m4a" if kind == "audio" else ".txt"
    base = report.get(base_key)
    stem = row.get("stem")
    if base and stem:
        return str(Path(str(base)) / f"{stem}{suffix}")
    return None


def build_entry(row: dict[str, Any], report: dict[str, Any], *, now: str) -> dict[str, Any] | None:
    """간 리포트 행에서 재시도 큐 엔트리를 생성."""
    reason = str(row.get("reason") or "")
    next_action = ACTIONABLE_REASON_TO_ACTION.get(reason)
    if not next_action:
        return None
    stem = str(row.get("stem") or Path(str(row.get("file", ""))).stem)
    file_name = str(row.get("file") or f"{stem}.txt")
    priority = str(row.get("priority") or "P3")
    source_path = _path_from_row(row, report, kind="audio")
    transcript_path = _path_from_row(row, report, kind="transcript")
    return {
        "schema_version": 1,
        "queue_id": queue_id_for(stem, reason, next_action),
        "created_at": now,
        "updated_at": now,
        "stem": stem,
        "file": file_name,
        "source_path": source_path,
        "transcript_path": transcript_path,
        "reason": reason,
        "next_action": next_action,
        "priority": priority,
        "status": "pending",
        "attempts": 0,
        "max_attempts": 3,
        "last_error": None,
        "next_retry_at": None,
        "terminal_failure": False,
        "history": [],
        "source_report": {
            "generated_at": report.get("generated_at"),
            "health": report.get("health"),
        },
    }


def build_queue_from_report(report: dict[str, Any], *, now: str | None = None) -> list[dict[str, Any]]:
    """간 분석 리포트에서 재시도 큐 엔트리 목록을 생성."""
    ts = coerce_now(now)
    entries: dict[str, dict[str, Any]] = {}
    cause_files = report.get("cause_files") or {}
    if not isinstance(cause_files, dict):
        raise QueueError("gap report missing object cause_files")
    for reason in ACTIONABLE_REASON_TO_ACTION:
        for row in cause_files.get(reason, []) or []:
            if not isinstance(row, dict):
                continue
            entry = build_entry(row, report, now=ts)
            if entry:
                entries[entry["queue_id"]] = entry
    return sorted(
        entries.values(),
        key=lambda e: (str(e["priority"]), ACTION_ORDER[e["next_action"]], str(e["stem"]), str(e["reason"])),
    )


def load_queue(path: Path) -> list[dict[str, Any]]:
    """JSONL 큐 파일을 로드."""
    if not path.exists():
        return []
    entries = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QueueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
        entries.append(row)
    return entries


def write_queue(path: Path, entries: Iterable[dict[str, Any]]) -> None:
    """엔트리 목록을 JSONL 파일로 저장."""
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in entries]
    safe_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def acquire_queue_lock(lock_path: Path, *, now: str, stale_seconds: int = 60 * 60 * 8) -> bool:
    """Acquire a process-level queue lock using atomic mkdir.

    The lock is intentionally coarse-grained: only one worker may mutate or run
    queue entries at a time.  Stale locks are reclaimed with an atomic rename so
    two contenders cannot both delete/create the live lock directory.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_path.mkdir()
        safe_write_text(lock_path / "owner.json", json.dumps({"pid": os.getpid(), "started_at": now}, sort_keys=True) + "\n")
        return True
    except FileExistsError:
        owner_file = lock_path / "owner.json"
        stale = False
        try:
            owner = json.loads(owner_file.read_text(encoding="utf-8"))
            started = parse_iso(str(owner.get("started_at")))
            stale = (parse_iso(now) - started).total_seconds() > stale_seconds
        except Exception:
            stale = True
        if not stale:
            return False
        tombstone = lock_path.with_name(f"{lock_path.name}.stale.{uuid.uuid4().hex}")
        try:
            lock_path.rename(tombstone)
        except FileNotFoundError:
            return acquire_queue_lock(lock_path, now=now, stale_seconds=stale_seconds)
        except OSError:
            return False
        try:
            lock_path.mkdir()
            safe_write_text(lock_path / "owner.json", json.dumps({"pid": os.getpid(), "started_at": now}, sort_keys=True) + "\n")
            return True
        finally:
            shutil.rmtree(tombstone, ignore_errors=True)


def release_queue_lock(lock_path: Path) -> None:
    shutil.rmtree(lock_path, ignore_errors=True)


def merge_queues(existing: list[dict[str, Any]], generated: list[dict[str, Any]], *, now: str) -> list[dict[str, Any]]:
    """기존 큐와 새로 생성된 엔트리를 병합."""
    by_id = {str(e["queue_id"]): e for e in existing}
    for new in generated:
        old = by_id.get(str(new["queue_id"]))
        if old and old.get("status") in STATUS_TERMINAL:
            continue
        if old:
            # Preserve retry state while refreshing paths/reason metadata from latest analysis.
            preserved = {
                k: old.get(k)
                for k in (
                    "created_at", "status", "attempts",
                    "last_error", "next_retry_at",
                    "terminal_failure", "history",
                )
            }
            old.update(new)
            old.update({k: v for k, v in preserved.items() if v is not None or k in {"last_error", "next_retry_at"}})
            old["updated_at"] = now
        else:
            by_id[str(new["queue_id"])] = new
    return sorted(
        by_id.values(),
        key=lambda e: (
            str(e.get("priority", "P9")),
            ACTION_ORDER.get(str(e.get("next_action")), 9),
            str(e.get("stem")),
        ),
    )


def backup_queue(queue_path: Path, workspace: Path, *, now: str) -> Path | None:
    """큐 파일의 타임스탬프 백업을 생성."""
    if not queue_path.exists():
        return None
    stamp = now.replace(":", "").replace("-", "").replace("+", "_")
    dest_dir = workspace / "backup" / f"retry_queue_{stamp}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / queue_path.name
    shutil.copy2(queue_path, dest)
    return dest


def entry_due(entry: dict[str, Any], *, now: str) -> bool:
    """엔트리가 지금 실행 가능한지 확인."""
    if entry.get("terminal_failure") or entry.get("status") in STATUS_TERMINAL:
        return False
    if entry.get("status") == "running":
        lease_expires_at = entry.get("lease_expires_at")
        return bool(lease_expires_at and parse_iso(str(lease_expires_at)) <= parse_iso(now))
    retry_at = entry.get("next_retry_at")
    if retry_at:
        return parse_iso(str(retry_at)) <= parse_iso(now)
    return entry.get("status", "pending") in {"pending", "failed"}


def _wsl_to_win_path(wsl_path: str) -> str:
    """Convert /mnt/X/... WSL path to X:\\... Windows path for cmd.exe."""
    if len(wsl_path) >= 6 and wsl_path.startswith("/mnt/"):
        drive = wsl_path[5].upper()
        rest = wsl_path[6:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return wsl_path


def _cmd_transcribe(entry, win_py, win_source_path, running_on_wsl):
    """Build command for transcribe action."""
    cmd = "/mnt/c/Windows/System32/cmd.exe"
    if running_on_wsl:
        argv = [cmd, "/c", f"{win_py} -m kct.transcribe.batch_transcribe --file \"{win_source_path}\""]
    else:
        argv = [win_py, "-m", "kct.transcribe.batch_transcribe", "--file", win_source_path]
    if entry.get("reason") == "transcription_failed":
        argv.append("--force")
    return argv


def _cmd_diarize(entry, win_py, win_source_path, running_on_wsl):
    """Build command for diarize action."""
    cmd = "/mnt/c/Windows/System32/cmd.exe"
    if running_on_wsl:
        return [cmd, "/c", f"{win_py} -m kct.transcribe.batch_transcribe --file \"{win_source_path}\" --force"]
    return [win_py, "-m", "kct.transcribe.batch_transcribe", "--file", win_source_path, "--force"]


def command_for_entry(entry: dict[str, Any], workspace: Path, running_on_wsl: bool | None = None) -> list[str]:
    """엔트리의 next_action에 맞는 실행 명령어를 생성."""
    action = str(entry.get("next_action"))
    source_path = entry.get("source_path")
    win_py = os.environ.get("KCT_WINDOWS_PYTHON", sys.executable)
    win_source_path = _wsl_to_win_path(str(source_path)) if source_path else ""
    if running_on_wsl is None:
        running_on_wsl = is_wsl()

    if action == "transcribe":
        if not source_path:
            raise QueueError(f"{entry.get('queue_id')}: missing source_path for transcribe")
        return _cmd_transcribe(entry, win_py, win_source_path, running_on_wsl)
    if action == "diarize":
        if not source_path:
            raise QueueError(f"{entry.get('queue_id')}: missing source_path for diarize")
        return _cmd_diarize(entry, win_py, win_source_path, running_on_wsl)
    if action == "entity":
        return [
            sys.executable, "-m", "kct.extract.extract_all",
            "--base-dir",
            str(Path(str(entry.get("transcript_path") or workspace)).parent),
        ]
    if action == "rag":
        return [sys.executable, "-m", "kct.queue.gap_analyzer", "--json"]
    if action == "obsidian":
        return [sys.executable, "-m", "kct.sync.sync_obsidian"]
    raise QueueError(f"unsupported next_action: {action}")


def backoff_next_retry(now: str, attempts: int) -> str:
    """지수 백오프로 다음 재시도 시간을 계산."""
    base = parse_iso(now)
    minutes = min(60 * 24, 15 * (2 ** max(0, attempts - 1)))
    return (base + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def summarize_error(returncode: int, stderr: str, stdout: str) -> str:
    """서브프로세스 에러 출력을 요약."""
    detail = redact_sensitive_text(stderr or stdout or "").strip().splitlines()
    tail = detail[-1] if detail else "no output"
    return f"exit {returncode}: {tail}"[:500]


def mark_success(entry: dict[str, Any], *, now: str, argv: list[str], stdout: str = "", stderr: str = "") -> None:
    """엔트리를 성공 상태로 표시."""
    entry["attempts"] = int(entry.get("attempts") or 0) + 1
    entry["status"] = "succeeded"
    entry["last_error"] = None
    entry["next_retry_at"] = None
    entry["terminal_failure"] = False
    entry["updated_at"] = now
    entry.setdefault("history", []).append({
        "at": now,
        "outcome": "succeeded",
        "argv": argv,
        "stdout_tail": redact_sensitive_text(stdout, limit=1000),
        "stderr_tail": redact_sensitive_text(stderr, limit=1000),
    })


def mark_failure(entry: dict[str, Any], *, now: str, argv: list[str], error: str) -> None:
    """엔트리를 실패 상태로 표시하고 재시도 또는 종료 실패를 결정."""
    attempts = int(entry.get("attempts") or 0) + 1
    max_attempts = int(entry.get("max_attempts") or 3)
    entry["attempts"] = attempts
    entry["last_error"] = error
    entry["updated_at"] = now
    if attempts >= max_attempts:
        entry["status"] = "terminal_failure"
        entry["terminal_failure"] = True
        entry["next_retry_at"] = None
    else:
        entry["status"] = "pending"
        entry["terminal_failure"] = False
        entry["next_retry_at"] = backoff_next_retry(now, attempts)
    entry.setdefault("history", []).append({"at": now, "outcome": "failed", "argv": argv, "error": error})


def run_worker(
    *,
    queue_path: Path = DEFAULT_QUEUE_PATH,
    workspace: Path = WORKSPACE,
    dry_run: bool = True,
    limit: int = 10,
    now: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: int = 60 * 60 * 8,
) -> dict[str, Any]:
    """재시도 큐에서 만료된 항목을 실행한다."""
    ts = coerce_now(now)
    entries = load_queue(queue_path)
    due = [e for e in entries if entry_due(e, now=ts)]
    due = sorted(
        due,
        key=lambda e: (
            str(e.get("priority", "P9")),
            ACTION_ORDER.get(str(e.get("next_action")), 9),
            str(e.get("stem")),
        ),
    )[: max(0, limit)]

    # On native Windows, detect if we need env var for Unicode argv handling
    _needs_env = (
        sys.platform == "win32"
        and runner is subprocess.run
    )

    def _build_cmd(entry):
        argv = command_for_entry(entry, workspace)
        if _needs_env:
            # On Windows, pass Unicode file path via env var to avoid argv encoding issues
            env = os.environ.copy()
            env["TRANSCRIBE_FILE"] = (
                _wsl_to_win_path(str(entry.get("source_path", "")))
                if entry.get("source_path") else ""
            )
            if entry.get("reason") == "transcription_failed" or entry.get("next_action") == "diarize":
                env["TRANSCRIBE_FORCE"] = "1"
            return (argv, env)
        return (argv, None)

    commands = [
        {
            "queue_id": e.get("queue_id"),
            "next_action": e.get("next_action"),
            "argv": _build_cmd(e)[0],
            "env": _build_cmd(e)[1],
        }
        for e in due
    ]
    result = {
        "dry_run": dry_run, "selected": len(due),
        "commands": commands,
        "succeeded": 0, "failed": 0, "backup": None,
        "locked": False,
    }
    if dry_run or not due:
        return result

    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    if not acquire_queue_lock(lock_path, now=ts, stale_seconds=timeout_seconds + 60):
        result.update({"selected": 0, "commands": [], "locked": True})
        return result
    run_token = f"rqrun_{uuid.uuid4().hex}"
    lease_expires_at = (parse_iso(ts) + timedelta(seconds=timeout_seconds + 60)).isoformat(timespec="seconds")

    try:
        backup = backup_queue(queue_path, workspace, now=ts)
        result["backup"] = str(backup) if backup else None
        append_jsonl(
        DEFAULT_LOG_PATH
        if workspace == WORKSPACE
        else workspace / "logs" / "transcription_retry_worker.jsonl",
        {
            "at": ts, "event": "worker_start",
            "queue_path": str(queue_path),
            "backup": result["backup"],
            "selected": len(due),
        },
        )
        by_id = {str(e.get("queue_id")): e for e in entries}
        log_path = DEFAULT_LOG_PATH if workspace == WORKSPACE else workspace / "logs" / "transcription_retry_worker.jsonl"
        for item in commands:
            entry = by_id[str(item["queue_id"])]
            argv = item["argv"]
            env = item.get("env")
            entry["status"] = "running"
            entry["locked_at"] = ts
            entry["lease_expires_at"] = lease_expires_at
            entry["run_token"] = run_token
            entry["updated_at"] = ts
            write_queue(queue_path, entries)
            append_jsonl(
                log_path,
                {
                    "at": ts, "event": "attempt_start",
                    "queue_id": entry.get("queue_id"),
                    "next_action": entry.get("next_action"),
                    "argv": argv,
                },
            )
            try:
                if env:
                    cp = runner(argv, cwd=str(workspace), timeout=timeout_seconds, text=True, capture_output=True, env=env)
                else:
                    cp = runner(argv, cwd=str(workspace), timeout=timeout_seconds, text=True, capture_output=True)
                rc = int(getattr(cp, "returncode", 1))
                stdout = str(getattr(cp, "stdout", "") or "")
                stderr = str(getattr(cp, "stderr", "") or "")
                if rc == 0:
                    mark_success(entry, now=ts, argv=argv, stdout=stdout, stderr=stderr)
                    entry.pop("locked_at", None)
                    entry.pop("lease_expires_at", None)
                    entry.pop("run_token", None)
                    result["succeeded"] += 1
                    append_jsonl(
                        log_path,
                        {
                            "at": ts, "event": "attempt_succeeded",
                            "queue_id": entry.get("queue_id"),
                            "returncode": rc,
                        },
                    )
                else:
                    err = summarize_error(rc, stderr, stdout)
                    mark_failure(entry, now=ts, argv=argv, error=err)
                    entry.pop("locked_at", None)
                    entry.pop("lease_expires_at", None)
                    entry.pop("run_token", None)
                    result["failed"] += 1
                    append_jsonl(
                        log_path,
                        {
                            "at": ts, "event": "attempt_failed",
                            "queue_id": entry.get("queue_id"),
                            "returncode": rc,
                            "error": redact_sensitive_text(err, limit=500),
                        },
                    )
            except (OSError, subprocess.SubprocessError) as exc:  # noqa: BLE001
                mark_failure(entry, now=ts, argv=argv, error=redact_sensitive_text(repr(exc), limit=500))
                entry.pop("locked_at", None)
                entry.pop("lease_expires_at", None)
                entry.pop("run_token", None)
                result["failed"] += 1
                append_jsonl(
                    log_path,
                    {
                        "at": ts, "event": "attempt_exception",
                        "queue_id": entry.get("queue_id"),
                        "error": redact_sensitive_text(repr(exc), limit=500),
                    },
                )
            write_queue(queue_path, entries)
        return result
    finally:
        release_queue_lock(lock_path)


def generate_queue(report_path: Path, queue_path: Path, *, merge: bool, now: str | None = None) -> dict[str, Any]:
    """간 분석 리포트에서 재시도 큐를 생성하거나 기존 큐에 병합."""
    ts = coerce_now(now)
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    generated = build_queue_from_report(report, now=ts)
    entries = merge_queues(load_queue(queue_path), generated, now=ts) if merge else generated
    write_queue(queue_path, entries)
    return {"queue_path": str(queue_path), "generated": len(generated), "written": len(entries)}


def parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI 인자를 파싱."""
    p = argparse.ArgumentParser(description="Generate/process transcription_retry_queue.jsonl")
    sub = p.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="Build queue JSONL from gap analyzer JSON")
    gen.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    gen.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)
    gen.add_argument("--replace", action="store_true", help="Replace queue instead of merging existing retry state")
    gen.add_argument("--print-summary", action="store_true")

    work = sub.add_parser("worker", help="Process due queue entries")
    work.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)
    work.add_argument("--workspace", type=Path, default=WORKSPACE)
    work.add_argument("--limit", type=int, default=10)
    work.add_argument("--execute", action="store_true", help="Actually run subprocesses; default is dry-run")
    work.add_argument("--timeout-seconds", type=int, default=60 * 60 * 8)
    work.add_argument("--print-json", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.cmd == "generate":
        result = generate_queue(args.report, args.queue, merge=not args.replace)
        if args.print_summary:
            log.info(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            log.info(f"queue={result['queue_path']} generated={result['generated']} written={result['written']}")
        return 0
    if args.cmd == "worker":
        result = run_worker(
            queue_path=args.queue, workspace=args.workspace,
            dry_run=not args.execute, limit=args.limit,
            timeout_seconds=args.timeout_seconds,
        )
        if args.print_json:
            log.info(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            log.info(
                "dry_run=%s selected=%s succeeded=%s "
                "failed=%s backup=%s",
                result['dry_run'], result['selected'],
                result['succeeded'], result['failed'],
                result['backup'],
            )
            for command in result["commands"]:
                log.info(f"- {command['queue_id']} {command['next_action']}: {' '.join(command['argv'])}")
        return 1 if result.get("failed") else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


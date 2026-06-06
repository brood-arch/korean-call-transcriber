#!/usr/bin/env python3
"""Hermes-owned active runner for the transcription pipeline cutover.

Runs the production pipeline from WSL while keeping Windows Task Scheduler as
rollback-only. The runner is intentionally conservative:
- atomic mkdir lock prevents overlap
- skips if Windows active owner tasks are not Disabled
- processes at most a small batch of new recordings
- uses Windows Python for Windows/G: drive dependent scripts
- emits shared events and writes a JSON summary for health/monitoring
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))

# ── paths.py loader ──────────────────────────────────────────────────────
import importlib.util as _importlib_util  # noqa: E402,I001
_SCRIPT_DIR = Path(__file__).resolve().parent
_paths_spec = _importlib_util.spec_from_file_location("paths", _SCRIPT_DIR / "paths.py")
if _paths_spec and _paths_spec.loader:
    _paths_mod = _importlib_util.module_from_spec(_paths_spec)
    _paths_spec.loader.exec_module(_paths_mod)
    _p = _paths_mod.p
else:
    import sys as _sys
    _WORKSPACE = _SCRIPT_DIR.parent
    if str(_WORKSPACE) not in _sys.path:
        _sys.path.insert(0, str(_WORKSPACE))
    from paths import Paths as _Paths
    _p = _Paths()

WORKSPACE = _p.root
SCRIPTS = WORKSPACE / "scripts"
STATE_DIR = _p.state_dir
LOG_DIR = _p.log_dir
LOCK_DIR = STATE_DIR / "hermes_transcription_pipeline.lock"
SUMMARY_PATH = STATE_DIR / _p.pipeline.active_run_summary
CMD_EXE = "/mnt/c/Windows/System32/cmd.exe"
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
WIN_WORKSPACE = str(_p.win.workspace)
WIN_PYTHON = str(_p.win.python)
MEMPALACE_PY = Path(os.environ.get("MEMPALACE_PY", "/home/brood38/.local/share/uv/tools/mempalace/bin/python"))
VENV_PYTHON = str(WORKSPACE / ".venv" / "bin" / "python")
ACTIVE_WINDOWS_TASKS = ["OpenClaw-CallRecordingsAutomation", "OpenClaw-Pipeline"]


def now_kst() -> datetime:
    return datetime.now(KST)


def iso_now() -> str:
    return now_kst().isoformat(timespec="seconds")


def safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_log(row: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "hermes_transcription_active_run.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def emit_event(event_type: str, summary: str, *, run_id: str, extra: dict[str, Any] | None = None, risks: list[str] | None = None) -> None:
    try:
        sys.path.insert(0, str(SCRIPTS))
        from emit_event import emit_event as _emit_event  # type: ignore
        _emit_event(
            agent="hermes",
            event_type=event_type,
            summary=summary,
            source="hermes_cutover",
            project="transcription_pipeline",
            risks=risks or [],
            extra=extra or {},
            idempotency_key=f"hermes:transcription_active:{run_id}:{event_type}",
        )
    except Exception as exc:  # event bus must not break the pipeline
        append_log({"at": iso_now(), "run_id": run_id, "event": "emit_event_failed", "error": repr(exc)})


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running (Linux/WSL)."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it (owned by another user) — treat as alive
        return True
    except Exception:
        return False


def acquire_lock(run_id: str, stale_seconds: int) -> bool:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        LOCK_DIR.mkdir()
        safe_write_json(LOCK_DIR / "owner.json", {"run_id": run_id, "pid": os.getpid(), "started_at": iso_now()})
        return True
    except FileExistsError:
        owner_file = LOCK_DIR / "owner.json"
        stale = False
        owner: dict[str, Any] = {}
        try:
            owner = json.loads(owner_file.read_text(encoding="utf-8"))
            started = datetime.fromisoformat(str(owner.get("started_at")))
            stale = (now_kst() - started).total_seconds() > stale_seconds
            # Also reclaim if the owning PID is no longer running (crash/OOM/segfault)
            if not stale:
                owner_pid = owner.get("pid")
                if owner_pid is not None and not _pid_alive(int(owner_pid)):
                    stale = True
        except Exception:
            stale = True
        if not stale:
            safe_write_json(SUMMARY_PATH, {"run_id": run_id, "status": "skipped_locked", "lock_owner": owner, "ended_at": iso_now()})
            return False
        shutil.rmtree(LOCK_DIR, ignore_errors=True)
        LOCK_DIR.mkdir()
        safe_write_json(LOCK_DIR / "owner.json", {"run_id": run_id, "pid": os.getpid(), "started_at": iso_now(), "reclaimed_stale": owner})
        return True


def release_lock() -> None:
    shutil.rmtree(LOCK_DIR, ignore_errors=True)


def run_cmd(name: str, argv: list[str], *, timeout: int, run_id: str) -> dict[str, Any]:
    start = time.time()
    append_log({"at": iso_now(), "run_id": run_id, "event": "step_start", "step": name, "argv": argv})
    try:
        cp = subprocess.run(argv, cwd=str(WORKSPACE), text=True, capture_output=True, timeout=timeout, encoding="utf-8", errors="replace")
        row = {
            "step": name,
            "returncode": int(cp.returncode),
            "duration_seconds": round(time.time() - start, 3),
            "stdout_tail": (cp.stdout or "")[-4000:],
            "stderr_tail": (cp.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        row = {
            "step": name,
            "returncode": 124,
            "duration_seconds": round(time.time() - start, 3),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": f"timeout after {timeout}s",
        }
    append_log({"at": iso_now(), "run_id": run_id, "event": "step_end", **row})
    return row


def win_py_step(script: str, args: list[str]) -> list[str]:
    # The configured paths contain no spaces. Avoid nested quotes here because
    # WSL subprocess(list)->cmd.exe can preserve escaped quotes literally.
    cmd = f"cd /d {WIN_WORKSPACE} && {WIN_PYTHON} -u scripts\\{script}"
    if args:
        cmd += " " + " ".join(args)
    return [CMD_EXE, "/c", cmd]


def get_windows_task_states(run_id: str) -> dict[str, str]:
    names = ",".join([f"'{n}'" for n in ACTIVE_WINDOWS_TASKS])
    ps = (
        f"Get-ScheduledTask -TaskName {names} | "
        "ForEach-Object { [PSCustomObject]@{ TaskName = $_.TaskName; State = $_.State.ToString() } } | "
        "ConvertTo-Json -Compress"
    )
    row = run_cmd("windows_task_state", [POWERSHELL, "-NoProfile", "-Command", ps], timeout=60, run_id=run_id)
    if row["returncode"] != 0:
        raise RuntimeError(f"could not read scheduled task state: {row.get('stderr_tail') or row.get('stdout_tail')}")
    raw = (row.get("stdout_tail") or "").strip()
    data = json.loads(raw) if raw else []
    if isinstance(data, dict):
        data = [data]
    return {str(item.get("TaskName")): str(item.get("State")) for item in data}


def _mem_snapshot() -> dict:
    """Capture current RAM/VRAM usage as a flat dict."""
    snap = {}
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS / "mem_guard.py"), "--json"],
            capture_output=True, text=True, timeout=15, encoding="utf-8", errors="replace"
        )
        data = json.loads(r.stdout)
        snap["ram_used_mb"] = data.get("ram_mb", {}).get("used_mb", -1)
        snap["ram_pct"] = data.get("ram_mb", {}).get("pct", 0)
        snap["vram_used_mb"] = data.get("vram_mb", {}).get("used_mb", -1)
        snap["vram_pct"] = data.get("vram_mb", {}).get("pct", 0)
        snap["vram_gpu"] = data.get("vram_mb", {}).get("gpu", "")
    except Exception as exc:
        snap["error"] = repr(exc)
    return snap


def _mem_reclaim() -> None:
    """Force GPU/RAM memory reclaim."""
    try:
        subprocess.run(
            ["python3", str(SCRIPTS / "mem_guard.py"), "--reclaim"],
            capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace"
        )
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hermes active transcription pipeline runner")
    p.add_argument("--transcribe-limit", type=int, default=3)
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--stale-lock-seconds", type=int, default=1800)
    p.add_argument("--ignore-owner-guard", action="store_true", help="Run even if Windows active tasks are not Disabled")
    p.add_argument("--skip-transcribe", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-obsidian", action="store_true")
    p.add_argument("--skip-shared-events", action="store_true")
    p.add_argument("--skip-mempalace", action="store_true", help="Skip MemPalace transcript archive/KG indexing")
    p.add_argument("--skip-email-archive", action="store_true", help="Skip Naver email → MemPalace archive")
    p.add_argument("--print-json", action="store_true")
    p.add_argument("--diarize-only", action="store_true", help="Shorthand: --skip-extract --skip-obsidian --skip-shared-events")
    p.add_argument("--obsidian-only", action="store_true", help="Shorthand: --skip-transcribe --skip-extract --skip-shared-events")
    p.add_argument("--health-only", action="store_true", help="Shorthand: --skip-transcribe --skip-extract --skip-obsidian --skip-shared-events")
    args = p.parse_args(argv)

    # Expand convenience flags
    if args.diarize_only:
        args.skip_extract = True
        args.skip_obsidian = True
        args.skip_shared_events = True
    if args.obsidian_only:
        args.skip_transcribe = True
        args.skip_extract = True
        args.skip_shared_events = True
    if args.health_only:
        args.skip_transcribe = True
        args.skip_extract = True
        args.skip_obsidian = True
        args.skip_shared_events = True
        args.skip_mempalace = True
        args.skip_email_archive = True

    run_id = f"hcut_{now_kst().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    summary: dict[str, Any] = {"run_id": run_id, "started_at": iso_now(), "status": "running", "steps": []}
    if not acquire_lock(run_id, args.stale_lock_seconds):
        print(json.dumps({"run_id": run_id, "status": "skipped_locked"}, ensure_ascii=False))
        return 0

    try:
        states = get_windows_task_states(run_id)
        summary["windows_task_states"] = states
        active = {name: state for name, state in states.items() if state != "Disabled"}
        if active and not args.ignore_owner_guard:
            summary.update({"status": "skipped_duplicate_owner", "ended_at": iso_now(), "active_windows_tasks": active})
            safe_write_json(SUMMARY_PATH, summary)
            emit_event("blocked", "Hermes active run skipped: Windows active owner tasks are not Disabled", run_id=run_id, extra={"active_windows_tasks": active}, risks=["duplicate owner guard tripped"])
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        emit_event("task_started", "Hermes active transcription pipeline run started", run_id=run_id, extra={"windows_task_states": states})

        # ── Memory guard: check before pipeline ──────────────────────────
        summary["mem_before"] = _mem_snapshot()
        if summary["mem_before"].get("vram_pct", 0) >= 80:
            append_log({"at": iso_now(), "run_id": run_id, "event": "mem_reclaim_before", "vram_pct": summary["mem_before"]["vram_pct"]})
            _mem_reclaim()

        if not args.skip_transcribe:
            summary["steps"].append(run_cmd(
                "call_recordings_automation",
                win_py_step("call_recordings_automation.py", ["--transcribe-limit", str(args.transcribe_limit), "--days", str(args.days), "--transcribe-only"]),
                timeout=7200,
                run_id=run_id,
            ))
        if not args.skip_transcribe:
            summary["steps"].append(run_cmd(
                "batch_diarize",
                [VENV_PYTHON, str(SCRIPTS / "batch_diarize.py"), "--days", str(args.days)],
                timeout=3600,
                run_id=run_id,
            ))
        summary["steps"].append(run_cmd("build_chroma_index", win_py_step("build_chroma_index.py", []), timeout=3600, run_id=run_id))
        if not args.skip_extract:
            summary["steps"].append(run_cmd("extract_all_today", win_py_step("extract_all.py", ["--today"]), timeout=3600, run_id=run_id))
        if not args.skip_obsidian:
            # WSL-native obsidian sync — avoids WSL→Windows→9p double-hop
            summary["steps"].append(run_cmd(
                "sync_transcripts_to_obsidian",
                ["python3", str(SCRIPTS / "obsidian_sync_wsl.py"), "transcripts"],
                timeout=1800,
                run_id=run_id,
            ))
        if not args.skip_shared_events:
            summary["steps"].append(run_cmd("index_shared_events", ["bash", str(SCRIPTS / "index_shared_events_wsl.sh")], timeout=1800, run_id=run_id))
        if not args.skip_mempalace:
            summary["steps"].append(run_cmd(
                "mempalace_business_archive",
                [str(MEMPALACE_PY), str(SCRIPTS / "mempalace_business_archive.py"), "--days", str(args.days), "--limit", "100"],
                timeout=3600,
                run_id=run_id,
            ))
        if not args.skip_email_archive:
            summary["steps"].append(run_cmd(
                "mempalace_email_archive",
                [str(MEMPALACE_PY), str(SCRIPTS / "mempalace_email_archive.py"), "--account", "brood38@naver.com", "--days", "14", "--limit", "0"],
                timeout=1800,
                run_id=run_id,
            ))
        summary["steps"].append(run_cmd("pipeline_health_check", win_py_step("pipeline_health_check.py", []), timeout=900, run_id=run_id))

        failed = [s for s in summary["steps"] if int(s.get("returncode", 1)) != 0]
        summary["status"] = "failed" if failed else "succeeded"

        # ── Memory guard: check + reclaim after pipeline ─────────────────
        summary["mem_after"] = _mem_snapshot()
        if summary["mem_after"].get("vram_pct", 0) >= 70:
            append_log({"at": iso_now(), "run_id": run_id, "event": "mem_reclaim_after", "vram_pct": summary["mem_after"]["vram_pct"]})
            _mem_reclaim()
            summary["mem_final"] = _mem_snapshot()

        summary["ended_at"] = iso_now()
        safe_write_json(SUMMARY_PATH, summary)
        if failed:
            emit_event("error", "Hermes active transcription pipeline run failed", run_id=run_id, extra={"failed_steps": [s["step"] for s in failed]}, risks=["rollback may be needed if failures repeat"])
        else:
            emit_event("task_completed", "Hermes active transcription pipeline run succeeded", run_id=run_id, extra={"steps": [s["step"] for s in summary["steps"]]})
        if args.print_json:
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1 if failed else 0
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())

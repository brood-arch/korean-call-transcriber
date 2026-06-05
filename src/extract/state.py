"""State file management for the integrated extraction pipeline.

Handles processed-file tracking, checkpoints, batch results,
and persistent TODO synchronization.
"""

import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import WORKSPACE
from src.pipeline.redact import redact_sensitive_text
from src.pipeline.utils import (
    normalize_source,
    normalize_title,
    parse_call_context,
    safe_load_json,
    safe_save_json,
)
from src.todo.persistent_store import merge_todos, todo_key

KST = timezone(timedelta(hours=9))
log = logging.getLogger(__name__)


def compute_file_hash(path: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_processed_index(processed_index_file: Path) -> dict:
    """Load {filename: file_hash} of already processed files."""
    if processed_index_file.exists():
        try:
            return json.loads(processed_index_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.debug("Failed to load processed index %s: %s", processed_index_file, redact_sensitive_text(repr(exc)))
    return {}


def save_processed_index(processed_index_file: Path, index: dict):
    """Save processed files index."""
    safe_save_json(processed_index_file, index, origin="integrated_pipeline")


def load_checkpoint(checkpoint_file: Path, today_only: bool = False) -> int:
    """Load checkpoint and return the next batch index to process."""
    if checkpoint_file.exists():
        try:
            cp = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            if today_only:
                today_str = datetime.now(KST).strftime("%Y%m%d")
                cp_date = cp.get("last_updated", "")[:10].replace("-", "")
                if cp_date != today_str:
                    log.info("Checkpoint stale (from %s), resetting for today", cp.get('last_updated', ''))
                    return 0
            return cp.get("last_completed_batch", -1) + 1
        except Exception as exc:
            log.debug("Failed to load checkpoint %s: %s", checkpoint_file, redact_sensitive_text(repr(exc)))
    return 0


def save_checkpoint(checkpoint_file: Path, batch_idx: int, total: int, stats: dict, run_id: str):
    """Save pipeline checkpoint."""
    data = {
        "last_completed_batch": batch_idx,
        "total_batches": total,
        "last_updated": datetime.now(KST).isoformat(),
        "run_id": run_id,
        "stats": stats,
    }
    safe_save_json(checkpoint_file, data, origin="integrated_pipeline")


def save_batch_result(
    state_dir: Path,
    batch_idx: int,
    batch_files: list,
    results: list,
    errors: list,
    status: str,
    run_id: str,
):
    """Save batch result to a JSON file."""
    output = {
        "batch_index": batch_idx,
        "run_id": run_id,
        "timestamp": datetime.now(KST).isoformat(),
        "files": [f.stem for f in batch_files],
        "results": results,
        "errors": errors,
        "status": status,
    }
    batch_file = state_dir / f"batch_{batch_idx:04d}.json"
    safe_save_json(batch_file, output, origin="integrated_pipeline")


# --- Persistent TODO sync ---

PERSISTENT_TODO_FILE = WORKSPACE / "memory" / "state" / "persistent_todos.json"


def load_persistent_todos() -> dict:
    """Load persistent_todos.json, creating if missing."""
    return safe_load_json(PERSISTENT_TODO_FILE, default={"todos": {}}) or {"todos": {}}


def save_persistent_todos(data: dict):
    """Save persistent_todos.json."""
    safe_save_json(PERSISTENT_TODO_FILE, data)


def sync_todos_to_persistent(batch_ok_results: list[dict], run_id: str) -> tuple[int, list]:
    """Sync new TODOs from extraction results to persistent_todos.json.

    Only syncs owner="me" tasks.
    Returns (count_of_new_todos, list_of_new_todos).
    """
    MY_OWNERS = {"me"}
    if not batch_ok_results:
        return 0, []

    store = load_persistent_todos()
    before = json.dumps(store, ensure_ascii=False, sort_keys=True, default=str)
    candidates = []

    for result in batch_ok_results:
        stem = result.get("file", "")
        stem_base = normalize_source(stem)
        ctx = parse_call_context(stem_base)
        for todo in result.get("todos", []):
            owner = todo.get("owner", "me")
            if owner not in MY_OWNERS:
                continue
            title = todo.get("title", "").strip()
            if not title:
                continue
            todo_entry = {
                "title": title,
                "owner": owner,
                "source": stem_base,
                "counterparty": ctx["caller"],
                "phone": ctx["phone"],
                "called_at": ctx["called_at"],
                "priority": todo.get("priority", "medium"),
                "status": "active",
                "details": todo.get("context", ""),
                "due_date": todo.get("due_date") or None,
                "created_at": datetime.now(KST).isoformat(),
                "run_id": run_id,
            }
            candidates.append(todo_entry)

    new_todos = merge_todos(store, candidates)
    after = json.dumps(store, ensure_ascii=False, sort_keys=True, default=str)
    if after != before:
        save_persistent_todos(store)

    return len(new_todos), new_todos


# --- Notification state ---

def appointment_key(appt: dict) -> str:
    """Generate a dedup key for an appointment."""
    source = normalize_source(appt.get("source", ""))
    return f"{appt.get('title','').strip()}|{appt.get('date')}|{appt.get('time')}|{source}"


def load_notification_state(notification_state_file: Path) -> dict:
    """Load extraction notification state and ensure notification buckets exist."""
    state = {}
    if notification_state_file.exists():
        try:
            state = json.loads(notification_state_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.debug(
                "Failed to load notification state %s: %s",
                notification_state_file,
                redact_sensitive_text(repr(exc)),
            )
            state = {}
    state.setdefault("notified_todos", {})
    state.setdefault("notified_appointments", {})
    state.setdefault("calendar_drafts", {})
    state.setdefault("last_summary", {})
    return state


def save_notification_state(notification_state_file: Path, state: dict):
    """Save notification state atomically."""
    safe_save_json(notification_state_file, state, origin="integrated_pipeline")


def collect_new_appointments(batch_results: list[dict], notification_state: dict) -> list:
    """Collect appointments not yet notified."""
    prev_notified = set(notification_state.get("notified_appointments", {}).keys())
    new_appointments = []
    for result in batch_results:
        if result.get("status") != "ok":
            continue
        stem = normalize_source(result.get("file", ""))
        ctx = parse_call_context(stem)
        for appt in result.get("appointments", []):
            appt_entry = {
                **appt,
                "source": stem,
                "counterparty": ctx.get("caller", ""),
                "phone": ctx.get("phone", ""),
                "called_at": ctx.get("called_at", ""),
            }
            if appointment_key(appt_entry) not in prev_notified:
                new_appointments.append(appt_entry)
    return new_appointments


def track_notified(
    notification_state_file: Path,
    new_todos: list,
    new_appointments: list,
    notifications: list | None = None,
):
    """Persist notified_todos/notified_appointments in shared state."""
    try:
        state = load_notification_state(notification_state_file)
        now_iso = datetime.now(KST).isoformat()

        for a in new_appointments or []:
            source = normalize_source(a.get("source", ""))
            appt = {**a, "source": source}
            state.setdefault("notified_appointments", {})[appointment_key(appt)] = {
                "title": appt.get("title"),
                "date": appt.get("date"),
                "updated_at": now_iso,
            }

        for t in new_todos or []:
            source = normalize_source(t.get("source", ""))
            todo = {**t, "source": source}
            state.setdefault("notified_todos", {})[todo_key(todo)] = {
                "title": todo.get("title"),
                "source": source,
                "updated_at": now_iso,
            }

        state["last_run"] = now_iso
        state["last_summary"] = {
            **state.get("last_summary", {}),
            "new_todos": len(new_todos or []),
            "new_appointments": len(new_appointments or []),
            "notifications": notifications or [],
            "source": "extract_all",
        }
        save_notification_state(notification_state_file, state)
    except Exception as e:
        log.warning(f"notification state tracking failed: {redact_sensitive_text(str(e))}")


# --- Telegram notification ---

def send_telegram(text: str):
    """Send a Telegram message using environment variables + curl subprocess."""
    try:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN not set")

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        return subprocess.run(
            [
                "curl", "-sS", "-X", "POST", url,
                "-d", f"chat_id={chat_id}",
                "--data-urlencode", f"text={text}",
                "-d", "disable_web_page_preview=true",
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20,
        )
    except Exception as e:
        class _Result:
            returncode = 1
            stdout = ""
            stderr = redact_sensitive_text(str(e))
        return _Result()


def _build_todo_msg_lines(todos, _fmt_phone, _fmt_date):
    """Build notification lines for a list of TODOs."""
    lines = []
    for t in todos:
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(t.get("priority"), "")
        title = t.get("title", "")
        cp = t.get("counterparty") or ""
        phone = t.get("phone") or ""
        called = t.get("called_at") or ""
        src_parts = []
        if cp and cp != "알 수 없음":
            src_parts.append(f"{cp} ({_fmt_phone(phone)})" if phone else cp)
        if called:
            src_parts.append(_fmt_date(called))
        src = " / ".join(src_parts)
        lines.append(f"  {priority_icon} {title}")
        if src:
            lines.append(f"    • 통화: {src}")
    return lines


def _build_appointment_msg_lines(new_appointments):
    """Build notification lines for appointments."""
    lines = ["", "📅 새 스케줄"]
    for a in new_appointments[:5]:
        date_str = a.get("date") or "미정"
        time_str = a.get("time") or ""
        cp = a.get("counterparty") or ""
        src = f" / {cp}" if cp and cp != "알 수 없음" else ""
        lines.append(f"  - {a.get('title')} / {date_str} {time_str}{src}")
    return lines


def _build_notification_body(new_todos, new_appointments, active_todos, _fmt_phone, _fmt_date):
    """Build the full notification message lines."""
    msg_lines = []

    if new_todos:
        msg_lines.append("🆕 새 TODO")
        msg_lines.extend(_build_todo_msg_lines(new_todos[:8], _fmt_phone, _fmt_date))

    if new_todos and active_todos:
        new_titles = {normalize_title(t.get("title", "")) for t in new_todos}
        existing = [t for t in active_todos if normalize_title(t.get("title", "")) not in new_titles]
        if existing:
            msg_lines.append("")
            msg_lines.append(f"📋 미완료 TODO ({len(existing)}건)")
            msg_lines.extend(_build_todo_msg_lines(existing[:10], _fmt_phone, _fmt_date))
            if len(existing) > 10:
                msg_lines.append(f"  ... 외 {len(existing) - 10}건")

    if new_appointments:
        msg_lines.extend(_build_appointment_msg_lines(new_appointments))

    return msg_lines


def notify_new_items(new_todos: list, new_appointments: list | None = None, active_todos: list | None = None) -> list:
    """Notify about new TODOs/appointments; include active TODO backlog with new TODO alerts."""
    def _fmt_phone(p):
        p = str(p or "")
        if not p or len(p) < 10:
            return p
        return f"{p[:3]}-{p[3:7]}-{p[7:]}"

    def _fmt_date(d):
        if not d:
            return ""
        try:
            dt = datetime.fromisoformat(str(d))
            return f"{dt.month}/{dt.day} {dt.hour:02d}:{dt.minute:02d}"
        except Exception as exc:
            log.debug("Failed to parse notification date %r: %s", d, redact_sensitive_text(repr(exc)))
            return str(d)[:16]

    new_appointments = new_appointments or []
    active_todos = active_todos or []

    msg_lines = _build_notification_body(new_todos, new_appointments, active_todos, _fmt_phone, _fmt_date)
    if not msg_lines:
        return []

    result = send_telegram("\n".join(msg_lines))
    ok = result.returncode == 0
    notes = [{"kind": "new_items", "ok": ok}]
    if not ok:
        stderr_or_stdout = (
            getattr(result, 'stderr', '')
            or getattr(result, 'stdout', '')
        )
        log.warning(
            f"telegram notification failed:"
            f" {redact_sensitive_text(stderr_or_stdout)}"
        )
    return notes


def print_todo_alert():
    """Print full active TODO report for immediate notification."""
    # NOTE: requires scripts/todo_report.py in WORKSPACE (not in public repo)
    try:
        result = subprocess.run(
            [sys.executable, str(WORKSPACE / "scripts" / "todo_report.py"), "--status", "active"],
            capture_output=True, text=True, timeout=10, cwd=str(WORKSPACE),
        )
        if result.stdout:
            log.info(f"\n{'='*50}")
            log.info("🚨 신규 TODO 발생 — 전체 활성 할 일:")
            log.info(f"{'='*50}")
            log.info(result.stdout)
    except Exception as e:
        log.warning(f"todo alert failed: {redact_sensitive_text(str(e))}")

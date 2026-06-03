#!/usr/bin/env python3
"""
Minions Job Queue — Postgres-backed durable job queue.

Features:
- Shell jobs (deterministic script execution)
- Subagent jobs (LLM tasks, extensible via backends)
- Crash recovery: state persisted in Postgres → resume on restart
- Parent-child DAG, priority, idempotency, progress tracking
- Fan-out parallel execution with aggregator pattern
- Job steering: send messages to running jobs

Environment variables:
    MINIONS_DB_HOST — Postgres host (default: localhost)
    MINIONS_DB_PORT — Postgres port (default: 5432)
    MINIONS_DB_NAME — Database name (default: minions)
    MINIONS_DB_USER — Database user (default: minions)
    MINIONS_DB_PASS — Database password (REQUIRED)

Usage:
    from src.pipeline.minions_queue import MinionsQueue

    mq = MinionsQueue()
    job_id = mq.submit("sync_transcripts", {"cmd": "python extract_all.py"})
    result = mq.get_job(job_id)
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import time
from datetime import timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    psycopg2 = None  # type: ignore

from src.config import (
    KCT_ENABLE_SHELL_JOBS,
    MINIONS_DB_HOST,
    MINIONS_DB_NAME,
    MINIONS_DB_PASS,
    MINIONS_DB_PORT,
    MINIONS_DB_URL,
    MINIONS_DB_USER,
)
from src.pipeline.utils import redact_sensitive_text

KST = timezone(timedelta(hours=9))

# Protected job names (MCP/agent cannot submit these directly)
PROTECTED_JOB_NAMES = {"shell", "subagent", "subagent_aggregator"}

# Maximum concurrent jobs
MAX_CONCURRENT = 3


def _db_config() -> dict:
    """Build Postgres connection config from environment variables."""
    if MINIONS_DB_URL:
        return {"dsn": MINIONS_DB_URL}
    if not MINIONS_DB_PASS:
        raise EnvironmentError(
            "MINIONS_DB_PASS environment variable is required. "
            "Set it to your Postgres password for the minions database."
        )
    return {
        "host": MINIONS_DB_HOST,
        "port": int(MINIONS_DB_PORT),
        "dbname": MINIONS_DB_NAME,
        "user": MINIONS_DB_USER,
        "password": MINIONS_DB_PASS,
    }


def _get_conn():
    """Get a Postgres connection."""
    if psycopg2 is None:
        raise ImportError(
            "psycopg2 is required for MinionsQueue. "
            "Install with: pip install psycopg2-binary"
        )
    cfg = _db_config()
    return psycopg2.connect(cfg["dsn"]) if "dsn" in cfg else psycopg2.connect(**cfg)


def _shell_jobs_enabled() -> bool:
    return KCT_ENABLE_SHELL_JOBS.lower() in {"1", "true", "yes"}


class MinionsQueue:
    """Postgres-backed durable job queue.

    Supports shell jobs, subagent jobs, fan-out parallel execution,
    and job steering via messages. All state is persisted in Postgres
    for crash recovery.
    """

    def __init__(self, db_config: dict | None = None):
        self.db_config = db_config or _db_config()
        self._test_connection()

    def _test_connection(self) -> None:
        """Verify database connectivity."""
        conn = psycopg2.connect(self.db_config["dsn"]) if "dsn" in self.db_config else psycopg2.connect(**self.db_config)
        conn.close()

    def _conn(self):
        return psycopg2.connect(self.db_config["dsn"]) if "dsn" in self.db_config else psycopg2.connect(**self.db_config)

    # ── Submit ──────────────────────────────────────────

    def submit(
        self,
        name: str,
        payload: dict,
        *,
        priority: int = 0,
        queue: str = "default",
        max_attempts: int = 3,
        timeout_ms: int = 300000,
        idempotency_key: str | None = None,
        parent_id: int | None = None,
        scheduled_at: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Submit a new job to the queue.

        Args:
            name: Job name (e.g., 'sync_transcripts', 'shell').
            payload: Job payload dict.
            priority: Higher = runs first.
            queue: Queue name for partitioning.
            max_attempts: Max retry count.
            timeout_ms: Timeout in milliseconds.
            idempotency_key: Prevents duplicate submissions.
            parent_id: Parent job ID for DAG.
            scheduled_at: ISO timestamp for deferred execution.
            metadata: Additional metadata dict.

        Returns:
            Job ID (int).
        """
        if name == "shell" and not _shell_jobs_enabled():
            raise RuntimeError("Shell jobs are disabled by default. Set KCT_ENABLE_SHELL_JOBS=1 for trusted local automation.")
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO jobs (name, status, priority, queue, payload,
                                  max_attempts, timeout_ms, idempotency_key,
                                  parent_id, scheduled_at, metadata)
                VALUES (%(name)s, 'pending', %(priority)s, %(queue)s,
                        %(payload)s, %(max_attempts)s, %(timeout_ms)s,
                        %(idempotency_key)s, %(parent_id)s,
                        COALESCE(%(scheduled_at)s::timestamptz, NOW()),
                        %(metadata)s)
                ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL
                DO NOTHING
                RETURNING id
                """,
                {
                    "name": name,
                    "priority": priority,
                    "queue": queue,
                    "payload": json.dumps(payload, ensure_ascii=False),
                    "max_attempts": max_attempts,
                    "timeout_ms": timeout_ms,
                    "idempotency_key": idempotency_key,
                    "parent_id": parent_id,
                    "scheduled_at": scheduled_at,
                    "metadata": json.dumps(metadata or {}, ensure_ascii=False),
                },
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "SELECT id FROM jobs WHERE idempotency_key = %s",
                    (idempotency_key,),
                )
                row = cur.fetchone()
            conn.commit()
            return row[0]
        finally:
            conn.close()

    # ── Query ───────────────────────────────────────────

    def get_job(self, job_id: int) -> Optional[dict]:
        """Get job details by ID."""
        conn = self._conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
            return cur.fetchone()
        finally:
            conn.close()

    def list_jobs(
        self, *, status: str | None = None, queue: str | None = None, limit: int = 20
    ) -> list[dict]:
        """List jobs with optional filtering."""
        conn = self._conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            query = "SELECT * FROM jobs WHERE 1=1"
            params: list = []
            if status:
                query += " AND status = %s"
                params.append(status)
            if queue:
                query += " AND queue = %s"
                params.append(queue)
            query += " ORDER BY priority DESC, created_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            return cur.fetchall()
        finally:
            conn.close()

    def get_progress(self, job_id: int) -> Optional[dict]:
        """Get job progress info."""
        conn = self._conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                SELECT id, name, status, progress, attempts, max_attempts,
                       started_at, completed_at, created_at
                FROM jobs WHERE id = %s
                """,
                (job_id,),
            )
            return cur.fetchone()
        finally:
            conn.close()

    def stats(self) -> dict:
        """Get queue statistics."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT status, count(*) FROM jobs GROUP BY status ORDER BY status")
            status_counts = dict(cur.fetchall())
            return {
                "statuses": status_counts,
                "total": sum(status_counts.values()),
                "queues": self._queue_stats(cur),
            }
        finally:
            conn.close()

    def _queue_stats(self, cur) -> dict:
        cur.execute(
            "SELECT queue, status, count(*) FROM jobs "
            "GROUP BY queue, status ORDER BY queue, status"
        )
        result: dict[str, dict] = {}
        for queue, status, count in cur.fetchall():
            result.setdefault(queue, {})[status] = count
        return result

    # ── Lifecycle ───────────────────────────────────────

    def cancel(self, job_id: int) -> bool:
        """Cancel a pending or active job."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE jobs SET status = 'cancelled', completed_at = NOW() "
                "WHERE id = %s AND status IN ('pending', 'active') RETURNING id",
                (job_id,),
            )
            conn.commit()
            return cur.fetchone() is not None
        finally:
            conn.close()

    def pause(self, job_id: int) -> bool:
        """Pause an active job."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE jobs SET status = 'paused' "
                "WHERE id = %s AND status = 'active' RETURNING id",
                (job_id,),
            )
            conn.commit()
            return cur.fetchone() is not None
        finally:
            conn.close()

    def resume(self, job_id: int) -> bool:
        """Resume a paused job."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE jobs SET status = 'pending' "
                "WHERE id = %s AND status = 'paused' RETURNING id",
                (job_id,),
            )
            conn.commit()
            return cur.fetchone() is not None
        finally:
            conn.close()

    def replay(self, job_id: int, data_overrides: dict | None = None) -> Optional[int]:
        """Create a new job as a replay of an existing one."""
        original = self.get_job(job_id)
        if not original:
            return None
        payload = original["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        if data_overrides:
            payload.update(data_overrides)
        return self.submit(
            name=original["name"],
            payload=payload,
            priority=original["priority"],
            queue=original["queue"],
            max_attempts=original["max_attempts"],
            timeout_ms=original["timeout_ms"],
            parent_id=original["parent_id"],
        )

    # ── Steering ────────────────────────────────────────

    def send_message(self, job_id: int, payload: dict, sender: str = "operator") -> int:
        """Send a message to a running job (for steering)."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO job_messages (job_id, sender, payload) VALUES (%s, %s, %s) RETURNING id",
                (job_id, sender, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()
            return cur.fetchone()[0]
        finally:
            conn.close()

    def read_messages(self, job_id: int) -> list[dict]:
        """Read and mark unread messages for a job."""
        conn = self._conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "UPDATE job_messages SET read = TRUE "
                "WHERE job_id = %s AND read = FALSE "
                "RETURNING id, sender, payload, created_at",
                (job_id,),
            )
            messages = cur.fetchall()
            conn.commit()
            return messages
        finally:
            conn.close()

    # ── Logging ─────────────────────────────────────────

    def log(self, job_id: int, level: str, message: str, metadata: dict | None = None) -> None:
        """Write a log entry for a job."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO job_logs (job_id, level, message, metadata) VALUES (%s, %s, %s, %s)",
                (job_id, level, message, json.dumps(metadata or {}, ensure_ascii=False)),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Worker ──────────────────────────────────────────

    def claim_next(self, queue: str = "default") -> Optional[dict]:
        """Atomically claim the next pending job (SKIP LOCKED)."""
        conn = self._conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                """
                UPDATE jobs SET status = 'active',
                                started_at = COALESCE(started_at, NOW()),
                                attempts = attempts + 1
                WHERE id = (
                    SELECT id FROM jobs
                    WHERE status = 'pending'
                      AND queue = %s
                      AND scheduled_at <= NOW()
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING *
                """,
                (queue,),
            )
            conn.commit()
            return cur.fetchone()
        finally:
            conn.close()

    def complete(self, job_id: int, result: dict) -> None:
        """Mark a job as completed."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE jobs SET status = 'completed', completed_at = NOW(), result = %s WHERE id = %s",
                (json.dumps(result, ensure_ascii=False), job_id),
            )
            conn.commit()
        finally:
            conn.close()

    def fail(self, job_id: int, error: str) -> str:
        """Mark a job as failed (or re-queue if retries remain).

        Returns:
            The resulting status ('pending' for retry, 'failed' for permanent failure).
        """
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs SET
                    error = %s,
                    status = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'pending' END,
                    completed_at = CASE WHEN attempts >= max_attempts THEN NOW() ELSE completed_at END
                WHERE id = %s
                RETURNING status
                """,
                (error, job_id),
            )
            conn.commit()
            return cur.fetchone()[0]
        finally:
            conn.close()

    def update_progress(self, job_id: int, progress: dict) -> None:
        """Update job progress data."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE jobs SET progress = %s WHERE id = %s",
                (json.dumps(progress, ensure_ascii=False), job_id),
            )
            conn.commit()
        finally:
            conn.close()

    # ── Shell Job Executor ──────────────────────────────

    def execute_shell(self, job: dict) -> dict:
        """Execute a shell/subprocess job.

        Preferred payload format (v0.7+):
            {"argv": ["python", "-m", "src.extract.extract_all", "--today"],
             "cwd": "/path/to/workspace",
             "env": {"KCT_WORKSPACE": "/path/to/workspace"}}

        Legacy format (deprecated, removal in v0.8):
            {"cmd": "python script.py", "cwd": "/path", "env": {...}}
        """
        payload = job["payload"]
        if isinstance(payload, str):
            payload = json.loads(payload)

        cmd = payload.get("cmd")
        argv = payload.get("argv")
        cwd = payload.get("cwd", os.getcwd())
        env_overrides = payload.get("env", {})
        timeout = job.get("timeout_ms", 300000) / 1000

        # Redact any sensitive values in env overrides before logging
        safe_env_preview = {k: redact_sensitive_text(str(v)) for k, v in env_overrides.items()}
        log.debug("Shell job: argv=%s cmd=%s cwd=%s env=%s", argv, cmd, cwd, safe_env_preview)

        run_env = {**os.environ, **env_overrides}

        try:
            # Prefer argv (structured, safe) over cmd (string, risky)
            if argv:
                proc = subprocess.run(
                    argv, cwd=cwd, env=run_env,
                    capture_output=True, text=True, timeout=timeout,
                )
            elif cmd:
                if not _shell_jobs_enabled():
                    return {
                        "exit_code": 2,
                        "error": "Shell command payloads are disabled by default. "
                                 "Set KCT_ENABLE_SHELL_JOBS=1 for trusted local automation.",
                    }
                log.warning("Legacy 'cmd' payload used; switch to 'argv' format. cmd=%s", cmd[:100])
                try:
                    parsed_argv = shlex.split(cmd, posix=(os.name != "nt"))
                except ValueError as exc:
                    return {"exit_code": 2, "error": f"Invalid cmd payload: {exc}"}
                proc = subprocess.run(
                    parsed_argv, cwd=cwd, env=run_env,
                    capture_output=True, text=True, timeout=timeout,
                )
            else:
                return {"exit_code": 1, "error": "No argv or cmd in payload"}

            return {
                "exit_code": proc.returncode,
                "stdout": redact_sensitive_text(proc.stdout or "", limit=2000),
                "stderr": redact_sensitive_text(proc.stderr or "", limit=2000),
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": -1, "error": f"Timeout ({timeout}s)"}
        except Exception as e:
            return {"exit_code": -1, "error": redact_sensitive_text(str(e))}

    # ── Fan-out ────────────────────────────────────────

    def submit_fanout(
        self,
        children: list[dict],
        aggregator: dict | None = None,
        *,
        queue: str = "default",
        priority: int = 0,
    ) -> dict:
        """Fan-out parallel execution: N child jobs + 1 optional aggregator.

        Args:
            children: List of {"name": str, "payload": dict} dicts.
            aggregator: Optional {"name": str, "payload": dict} for result aggregation.
            queue: Queue name.
            priority: Priority for all jobs.

        Returns:
            {"parent_id": int, "children": [int, ...], "aggregator_id": int|None}
        """
        parent_id = self.submit(
            name="fanout_parent",
            payload={"child_count": len(children)},
            priority=priority,
            queue=queue,
        )

        child_ids = []
        for child in children:
            cid = self.submit(
                name=child["name"],
                payload=child["payload"],
                priority=priority,
                queue=queue,
                parent_id=parent_id,
            )
            child_ids.append(cid)

        agg_id = None
        if aggregator:
            agg_id = self.submit(
                name=aggregator.get("name", "aggregator"),
                payload=aggregator.get("payload", {}),
                priority=priority,
                queue=queue,
                parent_id=parent_id,
                scheduled_at="9999-12-31T23:59:59",
            )

        return {"parent_id": parent_id, "children": child_ids, "aggregator_id": agg_id}

    def check_children_complete(self, parent_id: int) -> dict:
        """Check if all children of a parent job are complete.

        Returns:
            {"all_complete": bool, "completed": int, "total": int, "results": [dict, ...]}
        """
        conn = self._conn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT id, name, status, result, error FROM jobs WHERE parent_id = %s ORDER BY id",
                (parent_id,),
            )
            children = cur.fetchall()
            completed = [
                c for c in children if c["status"] in ("completed", "failed", "cancelled")
            ]
            all_complete = len(completed) == len(children) and len(children) > 0
            return {
                "all_complete": all_complete,
                "completed": len(completed),
                "total": len(children),
                "results": [dict(c) for c in children],
            }
        finally:
            conn.close()

    def activate_aggregator(self, parent_id: int) -> Optional[int]:
        """Activate the aggregator job for a parent once children are done."""
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE jobs SET scheduled_at = NOW()
                WHERE parent_id = %s
                  AND name IN ('aggregator', 'subagent_aggregator')
                  AND status = 'pending'
                  AND scheduled_at > NOW()
                RETURNING id
                """,
                (parent_id,),
            )
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
        finally:
            conn.close()

    # ── Worker Loop ─────────────────────────────────────

    def work_loop(self, queue: str = "default", interval: float = 5.0, once: bool = False) -> None:
        """Main worker loop: claim and execute pending jobs.

        Args:
            queue: Queue to poll.
            interval: Seconds between polls when no job is available.
            once: Run one iteration and exit.
        """
        print(f"Minions worker started (queue={queue}, interval={interval}s)")

        while True:
            job = self.claim_next(queue)
            if job:
                job_id = job["id"]
                job_name = job["name"]
                print(f"[{job_id}] Running: {job_name}")

                if job_name == "shell":
                    result = self.execute_shell(job)
                    if result.get("exit_code", -1) == 0:
                        self.complete(job_id, result)
                        log.info(f"[{job_id}] Completed")
                    else:
                        status = self.fail(job_id, result.get("error", "Unknown"))
                        log.error(f"[{job_id}] {'Will retry' if status == 'pending' else 'Failed permanently'}")

                elif job_name in ("aggregator", "subagent_aggregator"):
                    parent_id = job.get("parent_id")
                    if parent_id:
                        children = self.check_children_complete(parent_id)
                        self.complete(job_id, children)
                        print(f"[{job_id}] Aggregated {children['completed']}/{children['total']}")
                    else:
                        self.complete(job_id, {"note": "No parent, standalone"})

                elif job_name == "fanout_parent":
                    children = self.check_children_complete(job_id)
                    if children["all_complete"]:
                        self.activate_aggregator(job_id)
                        self.complete(job_id, children)
                        print(f"[{job_id}] All children done")
                    else:
                        conn = self._conn()
                        try:
                            cur = conn.cursor()
                            cur.execute(
                                "UPDATE jobs SET status = 'pending', attempts = attempts - 1 WHERE id = %s",
                                (job_id,),
                            )
                            conn.commit()
                        finally:
                            conn.close()
                    if once:
                        break
                    continue

                else:
                    # Custom job types fall back to shell execution
                    result = self.execute_shell(job)
                    if result.get("exit_code", -1) == 0:
                        self.complete(job_id, result)
                        log.info(f"[{job_id}] Completed")
                    else:
                        status = self.fail(job_id, result.get("error", "Unknown"))
                        log.error(f"[{job_id}] {'Will retry' if status == 'pending' else 'Failed permanently'}")

                # Check parent after child completion
                parent_id = job.get("parent_id")
                if parent_id:
                    children = self.check_children_complete(parent_id)
                    if children["all_complete"]:
                        self.activate_aggregator(parent_id)

            if once:
                break
            time.sleep(interval)

    # ── Cleanup ─────────────────────────────────────────

    def cleanup(self, days: int = 30) -> int:
        """Delete completed/failed/cancelled jobs older than N days.

        Returns:
            Number of jobs deleted.
        """
        conn = self._conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM jobs WHERE status IN ('completed', 'failed', 'cancelled') "
                "AND completed_at < NOW() - INTERVAL '%s days'",
                (days,),
            )
            target_ids = [r[0] for r in cur.fetchall()]
            if not target_ids:
                return 0
            placeholders = ",".join(["%s"] * len(target_ids))
            cur.execute(f"DELETE FROM job_logs WHERE job_id IN ({placeholders})", target_ids)
            cur.execute(f"DELETE FROM job_messages WHERE job_id IN ({placeholders})", target_ids)
            cur.execute(
                f"UPDATE jobs SET parent_id = NULL WHERE parent_id IN ({placeholders})",
                target_ids,
            )
            cur.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", target_ids)
            deleted = cur.rowcount
            conn.commit()
            return deleted
        finally:
            conn.close()


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    """Command-line interface for the job queue."""
    if psycopg2 is None:
        log.error("psycopg2 is required. Install with: pip install psycopg2-binary")
        sys.exit(1)

    mq = MinionsQueue()

    if len(sys.argv) < 2:
        print("Usage: minions_queue.py <command> [args]")
        print("  submit <name> <json_payload>   - Submit a job")
        print("  list [status]                  - List jobs")
        print("  get <job_id>                   - Get job details")
        print("  cancel <job_id>                - Cancel a job")
        print("  stats                          - Queue statistics")
        print("  work [queue]                   - Start worker")
        print("  cleanup [days]                 - Clean old jobs")
        print("  fanout <children_json> [agg]   - Fan-out parallel execution")
        print("  children <parent_id>           - Check child status")
        return

    cmd = sys.argv[1]

    if cmd == "submit" and len(sys.argv) > 3:
        name = sys.argv[2]
        payload = json.loads(sys.argv[3])
        job_id = mq.submit(name, payload)
        print(f"Job #{job_id} submitted")

    elif cmd == "list":
        status = sys.argv[2] if len(sys.argv) > 2 else None
        jobs = mq.list_jobs(status=status)
        for j in jobs:
            print(f"  #{j['id']} {j['name']} [{j['status']}] pri={j['priority']} created={j['created_at']}")

    elif cmd == "get" and len(sys.argv) > 2:
        job = mq.get_job(int(sys.argv[2]))
        print(json.dumps(dict(job) if job else {}, ensure_ascii=False, indent=2, default=str))

    elif cmd == "cancel" and len(sys.argv) > 2:
        ok = mq.cancel(int(sys.argv[2]))
        print(f"{'Cancelled' if ok else 'Not found or not cancellable'}")

    elif cmd == "stats":
        s = mq.stats()
        print(json.dumps(s, ensure_ascii=False, indent=2))

    elif cmd == "work":
        queue = sys.argv[2] if len(sys.argv) > 2 else "default"
        mq.work_loop(queue=queue)

    elif cmd == "cleanup":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        deleted = mq.cleanup(days)
        print(f"Cleaned up {deleted} old jobs")

    elif cmd == "fanout" and len(sys.argv) > 2:
        children = json.loads(sys.argv[2])
        aggregator = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
        result = mq.submit_fanout(children, aggregator)
        print(
            f"Fan-out: parent=#{result['parent_id']} "
            f"children={[f'#{c}' for c in result['children']]} "
            f"agg=#{result.get('aggregator_id')}"
        )

    elif cmd == "children" and len(sys.argv) > 2:
        children = mq.check_children_complete(int(sys.argv[2]))
        print(json.dumps(children, ensure_ascii=False, indent=2, default=str))

    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()

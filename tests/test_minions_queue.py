"""Tests for src.pipeline.minions_queue — job submission, lifecycle, stats (mocked DB)."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Mock psycopg2 globally for all tests ─────────────────────────────────

@pytest.fixture(autouse=True)
def _mock_psycopg2(monkeypatch):
    """Mock psycopg2 so no real PostgreSQL is needed."""
    mock_psycopg2 = MagicMock()
    mock_extras = MagicMock()

    # RealDictCursor returns plain dicts for easier testing
    class FakeRealDictCursor:
        def __init__(self, *a, **kw):
            pass

    mock_extras.RealDictCursor = FakeRealDictCursor
    mock_psycopg2.extras = mock_extras
    mock_psycopg2.connect = MagicMock()

    monkeypatch.setitem(__import__("sys").modules, "psycopg2", mock_psycopg2)
    monkeypatch.setitem(__import__("sys").modules, "psycopg2.extras", mock_extras)

    # Also patch the module-level reference in minions_queue
    import src.pipeline.minions_queue as mq
    monkeypatch.setattr(mq, "psycopg2", mock_psycopg2)

    return mock_psycopg2


@pytest.fixture(autouse=True)
def _mock_db_pass(monkeypatch):
    monkeypatch.setenv("MINIONS_DB_PASS", "test_password")


def _make_queue(mock_psycopg2):
    """Create a MinionsQueue with mocked connection."""
    from src.pipeline.minions_queue import MinionsQueue

    mock_conn = MagicMock()
    mock_psycopg2.connect.return_value = mock_conn
    mq = MinionsQueue(db_config={"host": "localhost", "port": 5432, "dbname": "test", "user": "test", "password": "test"})
    return mq, mock_conn


# ── Submit ───────────────────────────────────────────────────────────────

def test_submit_returns_job_id(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (42,)

    job_id = mq.submit("test_job", {"cmd": "echo hello"})
    assert job_id == 42
    mock_conn.commit.assert_called()


def test_submit_with_idempotency_key_existing(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    # First INSERT returns None (ON CONFLICT DO NOTHING)
    # Then SELECT returns the existing id
    mock_cursor.fetchone.side_effect = [None, (99,)]

    job_id = mq.submit("test_job", {"cmd": "echo"}, idempotency_key="unique-key")
    assert job_id == 99


# ── Get job ──────────────────────────────────────────────────────────────

def test_get_job(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = {"id": 1, "name": "test", "status": "pending"}

    job = mq.get_job(1)
    assert job is not None
    assert job["name"] == "test"


def test_get_job_not_found(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = None

    job = mq.get_job(999)
    assert job is None


# ── Status transitions ──────────────────────────────────────────────────

def test_cancel_pending(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (1,)

    result = mq.cancel(1)
    assert result is True
    mock_conn.commit.assert_called()


def test_cancel_not_cancellable(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = None

    result = mq.cancel(1)
    assert result is False


def test_pause_active(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (1,)

    result = mq.pause(1)
    assert result is True


def test_resume_paused(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (1,)

    result = mq.resume(1)
    assert result is True


# ── Complete / fail ─────────────────────────────────────────────────────

def test_complete(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mq.complete(1, {"exit_code": 0, "stdout": "done"})
    mock_conn.commit.assert_called()


def test_fail_with_retry(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    # attempts < max_attempts → status = 'pending' (retry)
    mock_cursor.fetchone.return_value = ("pending",)

    status = mq.fail(1, "timeout")
    assert status == "pending"


def test_fail_permanent(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    # attempts >= max_attempts → status = 'failed'
    mock_cursor.fetchone.return_value = ("failed",)

    status = mq.fail(1, "exhausted retries")
    assert status == "failed"


# ── Stats ───────────────────────────────────────────────────────────────

def test_stats(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    # Status counts
    mock_cursor.fetchall.side_effect = [
        [("pending", 5), ("completed", 10)],  # status counts
        [("default", "pending", 5)],  # queue stats
    ]

    stats = mq.stats()
    assert stats["total"] == 15
    assert stats["statuses"]["pending"] == 5
    assert stats["statuses"]["completed"] == 10


# ── List jobs ───────────────────────────────────────────────────────────

def test_list_jobs(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        {"id": 1, "name": "a", "status": "pending"},
        {"id": 2, "name": "b", "status": "completed"},
    ]

    jobs = mq.list_jobs(status="pending")
    assert len(jobs) == 2


# ── Shell execution ─────────────────────────────────────────────────────

def test_execute_shell_cmd(_mock_psycopg2, monkeypatch):
    monkeypatch.setenv("KCT_ENABLE_SHELL_JOBS", "1")
    mq, _ = _make_queue(_mock_psycopg2)
    job = {
        "payload": json.dumps({"cmd": f'{sys.executable} -c "print(\'hello\')"'}),
        "timeout_ms": 5000,
    }
    completed = MagicMock(returncode=0, stdout="hello\n", stderr="")
    with patch("src.pipeline.minions_queue.subprocess.run", return_value=completed) as run:
        result = mq.execute_shell(job)

    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
    assert "shell" not in run.call_args.kwargs


def test_execute_shell_cmd_disabled_by_default(_mock_psycopg2, monkeypatch):
    monkeypatch.delenv("KCT_ENABLE_SHELL_JOBS", raising=False)
    mq, _ = _make_queue(_mock_psycopg2)
    job = {
        "payload": json.dumps({"cmd": "echo hello"}),
        "timeout_ms": 5000,
    }
    result = mq.execute_shell(job)
    assert result["exit_code"] == 2
    assert "disabled by default" in result["error"]


def test_execute_shell_argv(_mock_psycopg2, monkeypatch):
    monkeypatch.setenv("KCT_ENABLE_SHELL_JOBS", "1")
    mq, _ = _make_queue(_mock_psycopg2)
    job = {
        "payload": json.dumps({"argv": ["python", "-c", "print(42)"]}),
        "timeout_ms": 5000,
    }
    result = mq.execute_shell(job)
    assert result["exit_code"] == 0
    assert "42" in result["stdout"]


def test_execute_shell_no_cmd(_mock_psycopg2, monkeypatch):
    monkeypatch.setenv("KCT_ENABLE_SHELL_JOBS", "1")
    mq, _ = _make_queue(_mock_psycopg2)
    job = {
        "payload": json.dumps({}),
        "timeout_ms": 5000,
    }
    result = mq.execute_shell(job)
    assert result["exit_code"] == 1
    assert "error" in result


def test_submit_shell_disabled_by_default(_mock_psycopg2, monkeypatch):
    monkeypatch.delenv("KCT_ENABLE_SHELL_JOBS", raising=False)
    mq, _ = _make_queue(_mock_psycopg2)
    with pytest.raises(RuntimeError, match="disabled by default"):
        mq.submit("shell", {"cmd": "echo hello"})


# ── Fan-out ─────────────────────────────────────────────────────────────

def test_submit_fanout(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    # Parent, child1, child2, aggregator
    mock_cursor.fetchone.side_effect = [(1,), (2,), (3,), (4,)]

    result = mq.submit_fanout(
        children=[
            {"name": "child1", "payload": {"x": 1}},
            {"name": "child2", "payload": {"x": 2}},
        ],
        aggregator={"name": "agg", "payload": {}},
    )
    assert result["parent_id"] == 1
    assert result["children"] == [2, 3]
    assert result["aggregator_id"] == 4


# ── Check children complete ─────────────────────────────────────────────

def test_check_children_complete(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        {"id": 2, "name": "c1", "status": "completed", "result": None, "error": None},
        {"id": 3, "name": "c2", "status": "completed", "result": None, "error": None},
    ]

    result = mq.check_children_complete(1)
    assert result["all_complete"] is True
    assert result["completed"] == 2
    assert result["total"] == 2


def test_check_children_incomplete(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        {"id": 2, "name": "c1", "status": "completed", "result": None, "error": None},
        {"id": 3, "name": "c2", "status": "pending", "result": None, "error": None},
    ]

    result = mq.check_children_complete(1)
    assert result["all_complete"] is False


# ── Logging / messages ──────────────────────────────────────────────────

def test_log(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mq.log(1, "info", "Job started")
    mock_conn.commit.assert_called()


def test_send_message(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = (1,)

    msg_id = mq.send_message(1, {"action": "stop"})
    assert msg_id == 1


def test_read_messages(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        {"id": 1, "sender": "op", "payload": '{"action":"stop"}', "created_at": "2026-06-01"},
    ]

    msgs = mq.read_messages(1)
    assert len(msgs) == 1


# ── Cleanup ─────────────────────────────────────────────────────────────

def test_cleanup_no_old_jobs(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = []

    deleted = mq.cleanup(days=30)
    assert deleted == 0


def test_cleanup_with_old_jobs(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [(1,), (2,)]
    mock_cursor.rowcount = 2

    deleted = mq.cleanup(days=30)
    assert deleted == 2


# ── Replay ──────────────────────────────────────────────────────────────

def test_replay(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    # get_job returns original
    mock_cursor.fetchone.side_effect = [
        {"id": 1, "name": "test", "status": "completed",
         "payload": json.dumps({"x": 1}), "priority": 0,
         "queue": "default", "max_attempts": 3, "timeout_ms": 300000,
         "parent_id": None},
        (2,),  # submit returns new id
    ]

    new_id = mq.replay(1)
    assert new_id == 2


def test_replay_nonexistent(_mock_psycopg2):
    mq, mock_conn = _make_queue(_mock_psycopg2)
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = None

    result = mq.replay(999)
    assert result is None


# ── DB config ───────────────────────────────────────────────────────────

def test_db_config_from_env(monkeypatch):
    from src.pipeline.minions_queue import _db_config
    monkeypatch.setenv("MINIONS_DB_HOST", "myhost")
    monkeypatch.setenv("MINIONS_DB_PORT", "5433")
    monkeypatch.setenv("MINIONS_DB_NAME", "mydb")
    monkeypatch.setenv("MINIONS_DB_USER", "myuser")
    monkeypatch.setenv("MINIONS_DB_PASS", "mypass")

    config = _db_config()
    assert config["host"] == "myhost"
    assert config["port"] == 5433
    assert config["dbname"] == "mydb"


def test_db_config_missing_password(monkeypatch):
    from src.pipeline.minions_queue import _db_config
    monkeypatch.delenv("MINIONS_DB_PASS", raising=False)
    with pytest.raises(EnvironmentError):
        _db_config()

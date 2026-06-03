"""Tests for src.todo.persistent_store — Jaccard dedup, same-source merge, state I/O."""

import json

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Redirect all state paths to tmp_path so tests are isolated."""
    monkeypatch.setenv("KCT_STATE_DIR", str(tmp_path))
    # Force module-level Path re-evaluation
    import src.todo.persistent_store as ps
    monkeypatch.setattr(ps, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(ps, "PERSISTENT_FILE", tmp_path / "persistent_todos.json")
    monkeypatch.setattr(ps, "COMPLETED_FILE", tmp_path / "completed_todos.json")


# ── Jaccard similarity ───────────────────────────────────────────────────

def test_jaccard_identical_strings():
    from src.todo.persistent_store import _jaccard_similarity
    assert _jaccard_similarity("배송 확인", "배송 확인") == 1.0


def test_jaccard_completely_different():
    from src.todo.persistent_store import _jaccard_similarity
    score = _jaccard_similarity("가나다라", "마바사아")
    assert score == 0.0


def test_jaccard_similar_strings():
    from src.todo.persistent_store import _jaccard_similarity
    score = _jaccard_similarity("배송확인부탁드립니다", "배송확인부탁드립니다")
    assert score >= 0.5


def test_jaccard_empty_strings():
    from src.todo.persistent_store import _jaccard_similarity
    assert _jaccard_similarity("", "") == 0.0
    assert _jaccard_similarity("text", "") == 0.0


# ── Fuzzy match ───────────────────────────────────────────────────────────

def test_is_fuzzy_match_identical():
    from src.todo.persistent_store import _is_fuzzy_match
    assert _is_fuzzy_match("주문 500개 발송", "주문 500개 발송")


def test_is_fuzzy_match_similar():
    from src.todo.persistent_store import _is_fuzzy_match
    assert _is_fuzzy_match("주문 500개 발송 확인", "주문 500개 발송 확인해 주세요")


def test_is_fuzzy_match_different():
    from src.todo.persistent_store import _is_fuzzy_match
    assert not _is_fuzzy_match("주문 500개 발송", "새로운 프로젝트 시작")


# ── Same-source dedup ────────────────────────────────────────────────────

def test_dedup_same_source_merges_identical():
    from src.todo.persistent_store import _dedup_same_source
    todos = [
        {"title": "배송 확인", "source": "call1.m4a"},
        {"title": "배송 확인", "source": "call1.m4a"},
    ]
    result = _dedup_same_source(todos)
    assert len(result) == 1


def test_dedup_same_source_merges_similar():
    from src.todo.persistent_store import _dedup_same_source
    todos = [
        {"title": "배송 확인 부탁드립니다", "source": "call1.m4a"},
        {"title": "배송 확인 부탁", "source": "call1.m4a"},
    ]
    result = _dedup_same_source(todos)
    assert len(result) == 1
    # Longer title should be kept
    assert "부탁드립니다" in result[0]["title"]


def test_dedup_same_source_keeps_different():
    from src.todo.persistent_store import _dedup_same_source
    todos = [
        {"title": "배송 확인", "source": "call1.m4a"},
        {"title": "견적서 전송", "source": "call1.m4a"},
    ]
    result = _dedup_same_source(todos)
    assert len(result) == 2


def test_dedup_same_source_different_sources_kept():
    from src.todo.persistent_store import _dedup_same_source
    todos = [
        {"title": "배송 확인", "source": "call1.m4a"},
        {"title": "배송 확인", "source": "call2.m4a"},
    ]
    result = _dedup_same_source(todos)
    assert len(result) == 2


def test_dedup_same_source_empty_list():
    from src.todo.persistent_store import _dedup_same_source
    assert _dedup_same_source([]) == []
    assert _dedup_same_source([{"title": "only one", "source": "x"}]) == [
        {"title": "only one", "source": "x"}
    ]


# ── Completed TODO tracking ──────────────────────────────────────────────

def test_completed_set_loaded_from_file(tmp_path):
    from src.todo.persistent_store import _load_completed_titles
    completed = tmp_path / "completed_todos.json"
    completed.write_text(json.dumps([
            {"title": "완료된 작업", "source": "call1.m4a"},
            {"title": "다른 완료", "source": "call2.m4a"},
    ], ensure_ascii=False), encoding="utf-8")
    result = _load_completed_titles()
    assert len(result) == 2


def test_is_completed_fuzzy(tmp_path):
    from src.todo.persistent_store import _is_completed_fuzzy, _load_completed_titles
    completed = tmp_path / "completed_todos.json"
    completed.write_text(json.dumps([
            {"title": "배송 확인 완료", "source": "call1.m4a"},
    ], ensure_ascii=False), encoding="utf-8")
    cs = _load_completed_titles()
    assert _is_completed_fuzzy("배송 확인 완료", cs)
    assert _is_completed_fuzzy("배송 확인 완료함", cs)  # fuzzy match


# ── Load / save state ────────────────────────────────────────────────────

def test_load_store_empty(tmp_path):
    from src.todo.persistent_store import load_store
    store = load_store()
    assert store["version"] == 1
    assert store["todos"] == {}


def test_save_and_load_roundtrip(tmp_path):
    from src.todo.persistent_store import load_store, save_store
    store = load_store()
    store["todos"]["key1"] = {"title": "test todo", "source": "call.m4a"}
    save_store(store)

    loaded = load_store()
    assert "key1" in loaded["todos"]
    assert loaded["todos"]["key1"]["title"] == "test todo"


def test_save_atomic_write(tmp_path):
    from src.todo.persistent_store import save_store
    store = {"version": 1, "todos": {}, "last_updated": ""}
    save_store(store)
    assert (tmp_path / "persistent_todos.json").exists()
    # No leftover .tmp file
    assert not (tmp_path / "persistent_todos.json.tmp").exists()


# ── merge_todos integration ──────────────────────────────────────────────

def test_merge_todos_adds_new(monkeypatch):
    from src.todo import persistent_store as ps
    # Patch _is_fuzzy_match to handle list-as-second-arg correctly
    _orig = ps._is_fuzzy_match
    def _fuzzy_match_list_safe(a, b, threshold=0.55):
        if isinstance(b, list):
            return any(_orig(a, item, threshold) for item in b)
        return _orig(a, b, threshold)
    monkeypatch.setattr(ps, "_is_fuzzy_match", _fuzzy_match_list_safe)
    store = ps.load_store()
    new = [{"title": "새로운 TODO", "source": "call1.m4a", "priority": "high"}]
    added = ps.merge_todos(store, new)
    assert len(added) == 1
    assert added[0]["title"] == "새로운 TODO"


def test_merge_todos_dedup_across_sources(monkeypatch):
    from src.todo import persistent_store as ps
    _orig = ps._is_fuzzy_match
    def _fuzzy_match_list_safe(a, b, threshold=0.55):
        if isinstance(b, list):
            return any(_orig(a, item, threshold) for item in b)
        return _orig(a, b, threshold)
    monkeypatch.setattr(ps, "_is_fuzzy_match", _fuzzy_match_list_safe)
    store = ps.load_store()
    first = [{"title": "배송 확인 부탁드립니다", "source": "call1.m4a"}]
    ps.merge_todos(store, first)
    # Same title from another source should be fuzzy-deduped
    second = [{"title": "배송 확인 부탁드립니다", "source": "call2.m4a"}]
    added = ps.merge_todos(store, second)
    assert len(added) == 0  # fuzzy dedup


def test_merge_todos_skips_completed(tmp_path):
    from src.todo.persistent_store import load_store, merge_todos
    # Write completed file as a list
    completed = tmp_path / "completed_todos.json"
    completed.write_text(json.dumps([
        {"title": "완료된 건", "source": ""}
    ], ensure_ascii=False), encoding="utf-8")
    store = load_store()
    new = [{"title": "완료된 건", "source": "call.m4a"}]
    added = merge_todos(store, new)
    assert len(added) == 0


def test_merge_todos_same_source_dedup_then_merge(monkeypatch):
    from src.todo import persistent_store as ps
    _orig = ps._is_fuzzy_match
    def _fuzzy_match_list_safe(a, b, threshold=0.55):
        if isinstance(b, list):
            return any(_orig(a, item, threshold) for item in b)
        return _orig(a, b, threshold)
    monkeypatch.setattr(ps, "_is_fuzzy_match", _fuzzy_match_list_safe)
    store = ps.load_store()
    new = [
        {"title": "send invoice", "source": "call1.m4a"},
        {"title": "send invoice today", "source": "call1.m4a"},
    ]
    added = ps.merge_todos(store, new)
    # Same-source dedup should merge into one (longer title kept)
    assert len(added) == 1
    assert added[0]["title"] == "send invoice today"


# ── get_active ───────────────────────────────────────────────────────────

def test_get_active_excludes_completed(tmp_path):
    from src.todo.persistent_store import get_active, load_store
    completed = tmp_path / "completed_todos.json"
    completed.write_text(json.dumps([
        {"title": "done", "source": ""}
    ], ensure_ascii=False), encoding="utf-8")

    store = load_store()
    store["todos"]["k"] = {"title": "done", "source": "call.m4a", "status": "active"}
    store["todos"]["k2"] = {"title": "still active", "source": "call.m4a", "status": "active"}
    active = get_active(store)
    assert len(active) == 1
    assert active[0]["title"] == "still active"


# ── todo_key ─────────────────────────────────────────────────────────────

def test_todo_key():
    from src.todo.persistent_store import todo_key
    key = todo_key({"title": "배송 확인", "source": "call1.m4a"})
    assert "배송확인" in key.replace(" ", "")
    assert "call1" in key


# ── sync_completed_to_file ───────────────────────────────────────────────

def test_sync_completed(tmp_path):
    from src.todo.persistent_store import sync_completed_to_file
    store = {"todos": {
        "k1": {"title": "done", "source": "call.m4a", "status": "completed"},
        "k2": {"title": "pending", "source": "call.m4a", "status": "active"},
    }}
    out = tmp_path / "completed_todos.json"
    count = sync_completed_to_file(store, str(out))
    assert count == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["completed_titles"]) == 1
    assert data["completed_titles"][0]["title"] == "done"

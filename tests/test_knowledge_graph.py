"""Tests for src.knowledge.graph — nodes, edges, traversal, persistence."""

import json

import pytest


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    monkeypatch.setenv("KCT_STATE_DIR", str(tmp_path))
    import src.knowledge.graph as g
    monkeypatch.setattr(g, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(g, "GRAPH_FILE", tmp_path / "knowledge_graph.json")


@pytest.fixture
def graph(tmp_path):
    from src.knowledge.graph import KnowledgeGraph
    return KnowledgeGraph(graph_path=tmp_path / "test_graph.json")


# ── Node operations ──────────────────────────────────────────────────────

def test_add_node(graph):
    graph.add_node("cp:Acme", "counterparty", "Acme Corp")
    assert "cp:Acme" in graph.nodes
    assert graph.nodes["cp:Acme"]["type"] == "counterparty"
    assert graph.nodes["cp:Acme"]["label"] == "Acme Corp"


def test_add_node_with_meta(graph):
    graph.add_node("prod:fan1", "product", "Industrial Fan", price=50000)
    assert graph.nodes["prod:fan1"]["meta"]["price"] == 50000


def test_add_node_updates_existing_meta(graph):
    graph.add_node("cp:Acme", "counterparty", "Acme Corp")
    graph.add_node("cp:Acme", "counterparty", "Acme Corp", region="US")
    assert graph.nodes["cp:Acme"]["meta"]["region"] == "US"
    # Label should not change on update
    assert graph.nodes["cp:Acme"]["label"] == "Acme Corp"


def test_query_missing_node(graph):
    assert "nonexistent" not in graph.nodes


def test_delete_node(graph):
    graph.add_node("cp:Acme", "counterparty", "Acme Corp")
    del graph.nodes["cp:Acme"]
    assert "cp:Acme" not in graph.nodes


# ── Edge operations ──────────────────────────────────────────────────────

def test_add_edge(graph):
    graph.add_node("cp:Acme", "counterparty", "Acme Corp")
    graph.add_node("todo:Ship", "todo", "Ship 500 units")
    graph.add_edge("cp:Acme", "todo:Ship", "has_todo", "call.m4a", "2026-06-01")
    assert len(graph.edges) == 1
    assert graph.edges[0]["from"] == "cp:Acme"
    assert graph.edges[0]["to"] == "todo:Ship"
    assert graph.edges[0]["relation"] == "has_todo"


def test_add_edge_dedup(graph):
    graph.add_node("a", "x", "A")
    graph.add_node("b", "y", "B")
    graph.add_edge("a", "b", "mentions", "s", "2026-01-01")
    graph.add_edge("a", "b", "mentions", "s", "2026-01-01")
    assert len(graph.edges) == 1


def test_different_relations_not_deduped(graph):
    graph.add_node("a", "x", "A")
    graph.add_node("b", "y", "B")
    graph.add_edge("a", "b", "mentions", "s", "2026-01-01")
    graph.add_edge("a", "b", "supplies", "s", "2026-01-01")
    assert len(graph.edges) == 2


# ── Traversal ────────────────────────────────────────────────────────────

def test_get_related_depth1(graph):
    graph.add_node("a", "x", "A")
    graph.add_node("b", "y", "B")
    graph.add_node("c", "z", "C")
    graph.add_edge("a", "b", "mentions", "s", "d")
    graph.add_edge("b", "c", "mentions", "s", "d")

    result = graph.get_related("a", depth=1)
    assert "a" in result["nodes"]
    assert "b" in result["nodes"]
    assert "c" not in result["nodes"]  # depth 1 only


def test_get_related_depth2(graph):
    graph.add_node("a", "x", "A")
    graph.add_node("b", "y", "B")
    graph.add_node("c", "z", "C")
    graph.add_edge("a", "b", "mentions", "s", "d")
    graph.add_edge("b", "c", "mentions", "s", "d")

    result = graph.get_related("a", depth=2)
    assert "a" in result["nodes"]
    assert "b" in result["nodes"]
    assert "c" in result["nodes"]


def test_get_related_nonexistent(graph):
    result = graph.get_related("nonexistent", depth=1)
    assert len(result["nodes"]) == 0
    assert len(result["edges"]) == 0


# ── Persistence ──────────────────────────────────────────────────────────

def test_save_and_load(tmp_path):
    from src.knowledge.graph import KnowledgeGraph
    gpath = tmp_path / "graph.json"

    g1 = KnowledgeGraph(graph_path=gpath)
    g1.add_node("cp:Test", "counterparty", "Test Corp")
    g1.add_node("todo:Task", "todo", "A task")
    g1.add_edge("cp:Test", "todo:Task", "has_todo", "call.m4a", "2026-06-01")
    g1.save()

    g2 = KnowledgeGraph(graph_path=gpath)
    assert "cp:Test" in g2.nodes
    assert "todo:Task" in g2.nodes
    assert len(g2.edges) == 1


def test_load_nonexistent(tmp_path):
    from src.knowledge.graph import KnowledgeGraph
    g = KnowledgeGraph(graph_path=tmp_path / "nonexistent.json")
    assert len(g.nodes) == 0
    assert len(g.edges) == 0


# ── Graph statistics ─────────────────────────────────────────────────────

def test_empty_graph_stats():
    import tempfile

    from src.knowledge.graph import KnowledgeGraph
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        g = KnowledgeGraph(graph_path=Path(td) / "empty.json")
        assert len(g.nodes) == 0
        assert len(g.edges) == 0


def test_node_type_counts(graph):
    graph.add_node("cp:A", "counterparty", "A")
    graph.add_node("cp:B", "counterparty", "B")
    graph.add_node("prod:X", "product", "X")
    types = {n["type"] for n in graph.nodes.values()}
    assert types == {"counterparty", "product"}


# ── build_from_todos ────────────────────────────────────────────────────

def test_build_from_todos(tmp_path):
    from src.knowledge.graph import KnowledgeGraph, build_from_todos
    todos_data = {
        "todos": {
            "key1": {
                "title": "Ship 500 units",
                "source": "call.m4a",
                "counterparty": "Acme",
            }
        }
    }
    (tmp_path / "persistent_todos.json").write_text(
        json.dumps(todos_data, ensure_ascii=False), encoding="utf-8"
    )
    g = KnowledgeGraph(graph_path=tmp_path / "g.json")
    build_from_todos(g, state_dir=tmp_path)
    assert "cp:Acme" in g.nodes
    assert any(n.startswith("todo:") for n in g.nodes)

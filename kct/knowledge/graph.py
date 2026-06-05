#!/usr/bin/env python3
"""
Knowledge Graph — Extract and maintain entity relationships.

Builds a simple relationship graph from:
- Call transcripts (counterparty ↔ product ↔ TODO)
- Calendar events (counterparty ↔ event)
- Completed TODOs (counterparty ↔ done items)

The graph is stored as nodes + edges in a JSON file, supporting
traversal queries up to configurable depth.

Relation types:
    - mentions:   entity mentions entity
    - supplies:   counterparty supplies product
    - has_todo:   counterparty has pending TODO
    - involved_in: person involved in project/event
    - decided:    decision made about entity

Environment variables:
    KCT_STATE_DIR — Base state directory (default: state)

Usage:
    from kct.knowledge.graph import KnowledgeGraph

    graph = KnowledgeGraph()
    graph.add_node("cp:AcmeCorp", "counterparty", "Acme Corp")
    graph.add_edge("cp:AcmeCorp", "todo:Ship 500 units", "has_todo", "call.m4a", "2026-06-01")
    graph.save()
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from kct.config import STATE_DIR
from kct.pipeline.utils import safe_save_json

KST = timezone(timedelta(hours=9))
log = logging.getLogger(__name__)

_STATE_DIR = Path(os.environ.get("KCT_STATE_DIR", str(STATE_DIR)))
GRAPH_FILE = _STATE_DIR / "knowledge_graph.json"

# Relation types
REL_MENTIONS = "mentions"
REL_SUPPLIES = "supplies"
REL_HAS_TODO = "has_todo"
REL_INVOLVED = "involved_in"
REL_DECIDED = "decided"


class KnowledgeGraph:
    """In-memory knowledge graph with JSON persistence.

    Nodes are keyed by ID (e.g., 'cp:AcmeCorp', 'todo:Ship items').
    Edges are stored as a flat list with deduplication.
    """

    def __init__(self, graph_path: Optional[Path] = None):
        self.graph_path = graph_path or GRAPH_FILE
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self._load()

    def _load(self) -> None:
        """Load graph from disk if it exists."""
        if self.graph_path.exists():
            try:
                data = json.loads(self.graph_path.read_text(encoding="utf-8"))
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", [])
            except (json.JSONDecodeError, OSError) as exc:  # noqa: BLE001
                log.debug("Failed to load knowledge graph %s: %s", self.graph_path, exc)

    def save(self) -> None:
        """Save graph to disk atomically."""
        self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated": datetime.now(KST).isoformat(),
            "nodes": self.nodes,
            "edges": self.edges,
            "stats": {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
            },
        }
        safe_save_json(self.graph_path, data, origin="knowledge_graph")

    def add_node(self, node_id: str, node_type: str, label: str, **meta) -> None:
        """Add or update a node in the graph.

        Args:
            node_id: Unique node identifier (e.g., 'cp:CompanyName').
            node_type: Type label (counterparty, todo, event, product, etc.).
            label: Human-readable display name.
            **meta: Additional metadata stored in node's 'meta' dict.
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "type": node_type,
                "label": label,
                "created": datetime.now(KST).isoformat(),
                "meta": meta,
            }
        else:
            self.nodes[node_id]["meta"].update(meta)

    def add_edge(
        self, from_id: str, to_id: str, relation: str, source: str, date: str
    ) -> None:
        """Add a directed edge between two nodes (deduplicated).

        Args:
            from_id: Source node ID.
            to_id: Target node ID.
            relation: Relationship type string.
            source: Origin (e.g., filename, 'manual').
            date: ISO date string.
        """
        for e in self.edges:
            if (
                e["from"] == from_id
                and e["to"] == to_id
                and e["relation"] == relation
            ):
                return
        self.edges.append(
            {"from": from_id, "to": to_id, "relation": relation, "source": source, "date": date}
        )

    def get_related(self, node_id: str, depth: int = 1) -> dict:
        """Get all nodes related to given node within depth (BFS).

        Args:
            node_id: Starting node ID.
            depth: Maximum traversal depth.

        Returns:
            Dict with 'nodes' and 'edges' sub-dicts.
        """
        visited: set[str] = set()
        result_nodes: dict[str, dict] = {}
        result_edges: list[dict] = []
        queue: list[tuple[str, int]] = [(node_id, 0)]

        while queue:
            current, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)

            if current in self.nodes:
                result_nodes[current] = self.nodes[current]

            for edge in self.edges:
                if edge["from"] == current and edge["to"] not in visited:
                    result_edges.append(edge)
                    queue.append((edge["to"], d + 1))
                elif edge["to"] == current and edge["from"] not in visited:
                    result_edges.append(edge)
                    queue.append((edge["from"], d + 1))

        return {"nodes": result_nodes, "edges": result_edges}


def build_from_todos(graph: KnowledgeGraph, state_dir: Optional[Path] = None) -> None:
    """Build relationships from persistent_todos.json.

    Args:
        graph: KnowledgeGraph instance to populate.
        state_dir: State directory path (default: KCT_STATE_DIR).
    """
    sdir = state_dir or _STATE_DIR
    todos_file = sdir / "persistent_todos.json"
    if not todos_file.exists():
        return

    data = json.loads(todos_file.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "todos" in data:
        todos_list = data["todos"].values()
    elif isinstance(data, list):
        todos_list = data
    else:
        return

    for todo in todos_list:
        if isinstance(todo, str):
            title = todo
            source = ""
            counterparty = ""
        else:
            title = todo.get("title", "")
            source = todo.get("source", "")
            counterparty = todo.get("counterparty", "")

        if counterparty:
            cp_id = f"cp:{counterparty}"
            graph.add_node(cp_id, "counterparty", counterparty)
            todo_id = f"todo:{title[:30]}"
            graph.add_node(todo_id, "todo", title, status="active")
            graph.add_edge(cp_id, todo_id, REL_HAS_TODO, source, todo.get("called_at", ""))


def build_from_calendar(graph: KnowledgeGraph, state_dir: Optional[Path] = None) -> None:
    """Build relationships from calendar drafts.

    Args:
        graph: KnowledgeGraph instance to populate.
        state_dir: State directory path.
    """
    sdir = state_dir or _STATE_DIR
    cal_file = sdir / "calendar_drafts.json"
    if not cal_file.exists():
        return

    drafts = json.loads(cal_file.read_text(encoding="utf-8"))
    if isinstance(drafts, list):
        for draft in drafts:
            title = draft.get("title", "")
            date = draft.get("date", "")
            source = draft.get("source", "")
            event_id = f"event:{title[:30]}"
            graph.add_node(event_id, "event", title, date=date)
            counterparty = draft.get("counterparty", "")
            if counterparty:
                cp_id = f"cp:{counterparty}"
                graph.add_node(cp_id, "counterparty", counterparty)
                graph.add_edge(cp_id, event_id, REL_INVOLVED, source, date)


def build_from_completed(graph: KnowledgeGraph, state_dir: Optional[Path] = None) -> None:
    """Build relationships from completed_todos.json (last 50 entries).

    Args:
        graph: KnowledgeGraph instance to populate.
        state_dir: State directory path.
    """
    sdir = state_dir or _STATE_DIR
    comp_file = sdir / "completed_todos.json"
    if not comp_file.exists():
        return

    completed = json.loads(comp_file.read_text(encoding="utf-8"))
    if isinstance(completed, list):
        for item in completed[-50:]:
            title = item.get("title", "")
            source = item.get("source", "")
            todo_id = f"todo:{title[:30]}"
            graph.add_node(todo_id, "todo", title, status="completed")
            counterparty = source.split("_")[0] if "_" in source else ""
            if counterparty and len(counterparty) > 1:
                cp_id = f"cp:{counterparty}"
                graph.add_node(cp_id, "counterparty", counterparty)
                graph.add_edge(cp_id, todo_id, REL_HAS_TODO, source, "")


# ── CLI ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge graph builder")
    parser.add_argument("--build", action="store_true", help="Build graph from all sources")
    parser.add_argument("--query", type=str, help="Query related nodes")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    args = parser.parse_args()

    graph = KnowledgeGraph()

    if args.build:
        build_from_todos(graph)
        build_from_calendar(graph)
        build_from_completed(graph)
        graph.save()
        print(f"Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    if args.stats:
        print(f"Nodes: {len(graph.nodes)}")
        print(f"Edges: {len(graph.edges)}")
        types: dict[str, int] = {}
        for n in graph.nodes.values():
            t = n.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        for t, c in sorted(types.items(), key=lambda x: -x[1]):
            print(f"  {t}: {c}")

    if args.query:
        results = graph.get_related(args.query, depth=2)
        print(
            f"Related to '{args.query}': "
            f"{len(results['nodes'])} nodes, {len(results['edges'])} edges"
        )
        for nid, node in results["nodes"].items():
            print(f"  {nid}: {node['label']} ({node['type']})")

"""Schedule Extractor — Extract appointments, deadlines, and todos from transcriptions.

Uses ChromaDB RAG to find schedule-relevant transcripts, then LLM to extract structured data.

Usage:
    python extract_schedules.py --days 7
    python extract_schedules.py --dry-run
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import STATE_DIR, TRANSCRIPT_DIR
from src.extract.client import call_llm_json
from src.pipeline.utils import safe_save_json

log = logging.getLogger(__name__)
KST = timezone(timedelta(hours=9))

OUTPUT_FILE = STATE_DIR / "extracted_schedules.json"

SCHEDULE_PROMPT = """Extract appointments, deadlines, and todos from this call transcription.

Output format (strict JSON):
{{
  "appointments": [
    {{"date": "YYYY-MM-DD", "time": "HH:MM or null", "title": "Title", "description": "Description", "people": ["Person"], "location": "Location or null"}}
  ],
  "todos": [
    {{"title": "Task", "priority": "high/medium/low", "deadline": "YYYY-MM-DD or null", "context": "Context"}}
  ]
}}

Rules:
- Only include confirmed dates/times
- Relative dates (tomorrow, next week) → null
- Deduplicate same appointments from multiple calls
- Empty arrays if nothing to extract
- OUTPUT ONLY JSON

Transcription:
{content}"""


def get_recent_transcripts(days: int) -> list[dict]:
    """Get transcripts modified within the last N days."""
    cutoff = datetime.now(KST) - timedelta(days=days)
    results = []
    for txt_file in TRANSCRIPT_DIR.glob("*.txt"):
        mtime = datetime.fromtimestamp(txt_file.stat().st_mtime, tz=KST)
        if mtime < cutoff:
            continue
        content = txt_file.read_text(encoding="utf-8").strip()
        if len(content) < 50:
            continue
        results.append({"path": str(txt_file), "name": txt_file.stem, "content": content})
    return results


def get_schedule_relevant_transcripts(days: int) -> list[dict]:
    """Use ChromaDB to find schedule-relevant transcripts (if available)."""
    try:
        import chromadb
        chroma_path = TRANSCRIPT_DIR / "chroma_index"
        if not chroma_path.exists():
            return []
        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.get_collection("transcripts")
    except Exception as exc:
        log.debug("ChromaDB schedule lookup unavailable: %s", exc)
        return []

    queries = [
        "약속 날짜 시간 만남 회의",
        "내일 모레 다음주 언제 몇시",
        "마감 기한 데드라인 제출 기한",
        "할 일 해야할 것 정리 준비",
    ]

    all_results = []
    seen_ids = set()

    for query in queries:
        try:
            results = col.query(query_texts=[query], n_results=20)
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                doc_id = meta.get("source", "")
                if doc_id in seen_ids or dist > 1.5:
                    continue
                seen_ids.add(doc_id)
                doc_path = Path(doc_id)
                if doc_path.exists():
                    content = doc_path.read_text(encoding="utf-8").strip()
                    all_results.append({"path": doc_id, "name": doc_path.stem, "content": content})
        except Exception as exc:
            log.debug("Schedule RAG query failed for %r: %s", query, exc)
            continue

    return all_results


def call_llm(content: str) -> dict:
    """Call LLM for schedule extraction."""
    prompt = SCHEDULE_PROMPT.replace("{content}", content[:8000], 1)
    parsed, _usage = call_llm_json(prompt, max_tokens=2048, timeout=120, response_format=True)
    return parse_schedule_response(json.dumps(parsed, ensure_ascii=False)) if parsed is not None else {"appointments": [], "todos": []}


def parse_schedule_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.split("\n") if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
        return {"appointments": data.get("appointments", []), "todos": data.get("todos", [])}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end])
                return {"appointments": data.get("appointments", []), "todos": data.get("todos", [])}
            except Exception as exc:
                log.debug("Failed to parse nested schedule JSON: %s", exc)
        return {"appointments": [], "todos": []}


def parse_transcript_name(stem: str) -> dict:
    """Parse counterparty info from transcript filename stem."""
    parts = stem.split("_")
    result = {"counterparty": "Unknown", "phone": "", "called_at": None}

    if len(parts) >= 3:
        dt_part = parts[-1]
        if len(dt_part) == 14 and dt_part.isdigit():
            try:
                result["called_at"] = f"{dt_part[:4]}-{dt_part[4:6]}-{dt_part[6:8]} {dt_part[8:10]}:{dt_part[10:12]}"
            except Exception as exc:
                log.debug("Failed to parse transcript timestamp from %s: %s", stem, exc)
            phone_part = parts[-2]
            if phone_part.isdigit() and len(phone_part) >= 10:
                result["phone"] = phone_part
            name_parts = parts[:-2]
            if name_parts:
                result["counterparty"] = "_".join(name_parts)
    return result


def deduplicate(items: list, key_fn) -> list:
    seen = set()
    unique = []
    for item in items:
        key = key_fn(item)
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Schedule Extractor")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--method", choices=["rag", "recent", "both"], default="both")
    args = parser.parse_args()

    transcripts = []
    if args.method in ("rag", "both"):
        rag_results = get_schedule_relevant_transcripts(args.days)
        transcripts.extend(rag_results)
    if args.method in ("recent", "both"):
        recent_results = get_recent_transcripts(args.days)
        existing_paths = set(t["path"] for t in transcripts)
        for r in recent_results:
            if r["path"] not in existing_paths:
                transcripts.append(r)

    if not transcripts:
        print("No relevant transcripts found")
        return

    all_appointments = []
    all_todos = []

    for i, transcript in enumerate(transcripts):
        content = transcript["content"]
        if len(content) < 100:
            continue

        result = call_llm(content)
        meta = parse_transcript_name(transcript["name"])
        for apt in result.get("appointments", []):
            apt["source"] = transcript["name"]
            apt["counterparty"] = meta["counterparty"]
        for todo in result.get("todos", []):
            todo["source"] = transcript["name"]

        all_appointments.extend(result.get("appointments", []))
        all_todos.extend(result.get("todos", []))

        if i < len(transcripts) - 1:
            time.sleep(2)

    all_appointments = deduplicate(all_appointments, lambda a: f"{a.get('title','')}|{a.get('date','')}|{a.get('time','')}")
    all_todos = deduplicate(all_todos, lambda t: t.get("title", "").strip())

    print(f"\nExtracted: {len(all_appointments)} appointments, {len(all_todos)} todos")

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "extracted_at": datetime.now(KST).isoformat(),
        "appointments": all_appointments,
        "todos": all_todos,
    }
    safe_save_json(OUTPUT_FILE, output)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

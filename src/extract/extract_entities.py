"""Entity Extraction Pipeline — Standalone module.

Extracts entities and relations from transcription files using LLM.

Stability features:
- Checkpoint per batch: resume from last checkpoint on restart
- Rate limiting with exponential backoff
- Dry-run mode for validation
- JSON output per batch (never loses progress)

Usage:
    python extract_entities.py --dry-run
    python extract_entities.py
    python extract_entities.py --start-batch 10
"""

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.config import STATE_DIR, TRANSCRIPT_DIR
from src.extract.client import call_llm_json
from src.pipeline.utils import safe_save_json

log = logging.getLogger(__name__)
KST = __import__("datetime").timezone(__import__("datetime").timedelta(hours=9))

# ── Configuration ───────────────────────────────────────────────────────
DEFAULT_BASE_DIR = str(TRANSCRIPT_DIR)
DEFAULT_STATE_DIR = str(STATE_DIR / "entity_extraction")
BATCH_SIZE = 20
API_DELAY = 2.0

ENTITY_PROMPT = """Extract entities and relations from this call transcription.

Entity types: Person, Organization, Location, Date, Money, PhoneNumber, Product, Project, Event, Contract

Output format (strict JSON):
{{
  "entities": [
    {{"name": "name", "type": "type", "context": "context"}}
  ],
  "relations": [
    {{"from": "EntityA", "to": "EntityB", "type": "relation_type", "context": "context"}}
  ]
}}

If no entities or relations, use empty arrays.
OUTPUT ONLY JSON. No markdown or explanation.

Transcription:
{content}"""


def get_all_transcription_files(base_dir) -> list[Path]:
    base_dir = Path(base_dir)
    files = sorted(base_dir.glob("*.txt"))
    return [f for f in files if f.stat().st_size > 50]


def compute_file_hash(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.md5(content).hexdigest()


def call_llm(content: str) -> dict | None:
    """Call LLM for entity extraction."""
    prompt = ENTITY_PROMPT.replace("{content}", content[:8000], 1)
    parsed, _usage = call_llm_json(prompt, max_tokens=2048, timeout=180, response_format=True)
    return parse_entity_response(json.dumps(parsed, ensure_ascii=False)) if parsed is not None else None


def parse_entity_response(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
        return {"entities": data.get("entities", []), "relations": data.get("relations", [])}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end])
                return {"entities": data.get("entities", []), "relations": data.get("relations", [])}
            except json.JSONDecodeError as exc:
                log.debug("Failed to parse nested entity JSON: %s", exc)
        return {"entities": [], "relations": [], "parse_error": True, "raw": cleaned[:500]}


def run_extraction(args):
    """Main extraction pipeline."""
    base_dir = Path(args.base_dir)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    files = get_all_transcription_files(base_dir)
    total_files = len(files)
    print(f"Found {total_files} transcription files")

    if total_files == 0:
        return

    total_batches = (total_files + args.batch_size - 1) // args.batch_size

    checkpoint_path = state_dir / "checkpoint.json"
    start_batch = args.start_batch
    if checkpoint_path.exists() and not args.force_restart:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        start_batch = max(start_batch, checkpoint.get("last_completed_batch", -1) + 1)

    for batch_idx in range(start_batch, total_batches):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, total_files)
        batch_files = files[batch_start:batch_end]
        batch_id = f"batch_{batch_idx:04d}"

        batch_result_path = state_dir / f"{batch_id}.json"
        if batch_result_path.exists() and not args.force_restart:
            try:
                with open(batch_result_path) as f:
                    existing = json.load(f)
                if existing.get("status") == "done":
                    continue
            except (json.JSONDecodeError, KeyError) as exc:
                log.debug("Ignoring invalid existing entity batch %s: %s", batch_result_path, exc)

        batch_entities = []
        batch_relations = []
        batch_errors = []

        for file_path in batch_files:
            file_id = file_path.stem
            try:
                content = file_path.read_text(encoding="utf-8").strip()
                if not content:
                    batch_errors.append({"file": file_id, "error": "empty"})
                    continue

                result = call_llm(content)
                if result:
                    batch_entities.extend(result.get("entities", []))
                    batch_relations.extend(result.get("relations", []))
                else:
                    batch_errors.append({"file": file_id, "error": "api_failed"})

            except Exception as e:
                batch_errors.append({"file": file_id, "error": str(e)})

            time.sleep(args.api_delay)

        batch_output = {
            "batch_id": batch_id,
            "files_total": len(batch_files),
            "entities_found": len(batch_entities),
            "relations_found": len(batch_relations),
            "entities": batch_entities,
            "relations": batch_relations,
            "errors": batch_errors,
            "status": "done",
            "timestamp": datetime.now().isoformat(),
        }
        safe_save_json(batch_result_path, batch_output)

        print(f"  [{batch_id}] {len(batch_entities)} entities, {len(batch_relations)} relations, {len(batch_errors)} errors")

        checkpoint = {
            "last_completed_batch": batch_idx,
            "total_batches": total_batches,
            "last_updated": datetime.now().isoformat(),
        }
        safe_save_json(checkpoint_path, checkpoint)

        if batch_idx < total_batches - 1:
            time.sleep(args.api_delay)

    print(f"\nExtraction complete: {total_batches} batches processed")


def main():
    parser = argparse.ArgumentParser(description="Entity Extraction Pipeline")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--api-delay", type=float, default=API_DELAY)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN — Entity Extraction Pipeline")
        print(f"Base dir: {args.base_dir}")
        print(f"Files: {len(get_all_transcription_files(args.base_dir))}")
    else:
        run_extraction(args)


if __name__ == "__main__":
    main()

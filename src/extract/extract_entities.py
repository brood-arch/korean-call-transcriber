"""Entity Extraction Pipeline — Standalone module.

Extracts entities and relations from transcription files using LLM.

Uses :func:`src.extract.client.call_llm_extract` for all LLM HTTP calls,
keeping retry/backoff logic centralized.

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
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import EXIT_CONFIG, EXIT_OK, TRANSCRIPT_DIR
from src.extract.client import call_llm_extract, get_llm_config
from src.pipeline.utils import safe_save_json

log = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

# ── Configuration ───────────────────────────────────────────────────────
DEFAULT_BASE_DIR = str(TRANSCRIPT_DIR)
DEFAULT_STATE_DIR = "state/entity_extraction"
BATCH_SIZE = 20
API_DELAY = 2.0

ENTITY_PROMPT = """Extract entities and relations from this call transcription.

Entity types: Person, Organization, Location, Date, Money, PhoneNumber, Product, Project, Event, Contract

Output format (strict JSON):
{{
  "summary": {{"one_line": "", "details": [], "call_type": "unknown", "overall_confidence": 0.5}},
  "todos": [],
  "appointments": [],
  "entities": [
    {{"name": "name", "type": "type", "context": "context"}}
  ],
  "products": [],
  "money": [],
  "risks": [],
  "corrections": [],
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


def call_llm(content: str, api_key: str = "") -> dict | None:
    """Call LLM for entity extraction via the centralized client.

    Delegates HTTP + retry + backoff to :func:`src.extract.client.call_llm_extract`.
    """
    result = call_llm_extract(api_key=api_key or get_llm_config().api_key, content=content[:8000])
    if result is None:
        return None
    # Extract entity-specific fields from the unified response
    return {
        "entities": result.get("entities", []),
        "relations": result.get("relations", []),
    }


def parse_entity_response(text: str) -> dict:
    """Parse raw LLM text into entity dict (used for direct prompt results)."""
    import json as _json

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = _json.loads(cleaned)
        return {"entities": data.get("entities", []), "relations": data.get("relations", [])}
    except _json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = _json.loads(cleaned[start:end])
                return {"entities": data.get("entities", []), "relations": data.get("relations", [])}
            except _json.JSONDecodeError:
                log.debug("Failed to parse entity JSON after extraction: %s", cleaned[:200])
        return {"entities": [], "relations": [], "parse_error": True, "raw": cleaned[:500]}


def _process_entity_batch(batch_files, api_key, api_delay):
    """Process one batch of files for entity extraction.

    Returns (batch_entities, batch_relations, batch_errors).
    """
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

            result = call_llm(content, api_key=api_key)
            if result:
                batch_entities.extend(result.get("entities", []))
                batch_relations.extend(result.get("relations", []))
            else:
                batch_errors.append({"file": file_id, "error": "api_failed"})

        except Exception as e:
            log.warning("Entity extraction failed for %s: %s", file_id, e)
            batch_errors.append({"file": file_id, "error": str(e)})

        time.sleep(api_delay)

    return batch_entities, batch_relations, batch_errors


def _resolve_start_batch(state_dir, args_start_batch, force_restart):
    """Determine starting batch index from checkpoint or args."""
    checkpoint_path = state_dir / "checkpoint.json"
    start_batch = args_start_batch
    if checkpoint_path.exists() and not force_restart:
        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            start_batch = max(start_batch, checkpoint.get("last_completed_batch", -1) + 1)
        except (json.JSONDecodeError, OSError) as exc:
            log.debug("Failed to load checkpoint: %s", exc)
    return start_batch


def _is_batch_done(batch_result_path, force_restart):
    """Check if a batch result already exists and is done."""
    if not batch_result_path.exists() or force_restart:
        return False
    try:
        with open(batch_result_path) as f:
            existing = json.load(f)
        return existing.get("status") == "done"
    except (json.JSONDecodeError, KeyError) as exc:
        log.debug("Failed to inspect existing batch: %s", exc)
        return False


def run_extraction(args):
    """Main extraction pipeline."""
    base_dir = Path(args.base_dir)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    api_key = get_llm_config().api_key
    if not api_key:
        log.error("LLM_API_KEY not set")
        return EXIT_CONFIG

    files = get_all_transcription_files(base_dir)
    total_files = len(files)
    log.info(f"Found {total_files} transcription files")
    if total_files == 0:
        return EXIT_OK

    total_batches = (total_files + args.batch_size - 1) // args.batch_size
    start_batch = _resolve_start_batch(state_dir, args.start_batch, args.force_restart)

    for batch_idx in range(start_batch, total_batches):
        _run_entity_batch(batch_idx, total_batches, total_files, files, state_dir, api_key, args)

    log.info(f"\nExtraction complete: {total_batches} batches processed")
    return EXIT_OK


def _run_entity_batch(batch_idx, total_batches, total_files, files, state_dir, api_key, args):
    """Process and save a single batch of entity extractions."""
    batch_start = batch_idx * args.batch_size
    batch_end = min(batch_start + args.batch_size, total_files)
    batch_files = files[batch_start:batch_end]
    batch_id = f"batch_{batch_idx:04d}"

    batch_result_path = state_dir / f"{batch_id}.json"
    if _is_batch_done(batch_result_path, args.force_restart):
        return

    batch_entities, batch_relations, batch_errors = _process_entity_batch(
        batch_files, api_key, args.api_delay,
    )

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
    safe_save_json(batch_result_path, batch_output, origin="entity_extraction")

    log.error(
        f"  [{batch_id}] {len(batch_entities)} entities, "
        f"{len(batch_relations)} relations, {len(batch_errors)} errors"
    )

    checkpoint = {
        "last_completed_batch": batch_idx,
        "total_batches": total_batches,
        "last_updated": datetime.now().isoformat(),
    }
    safe_save_json(state_dir / "checkpoint.json", checkpoint, origin="entity_extraction")

    if batch_idx < total_batches - 1:
        time.sleep(args.api_delay)


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
        log.info("DRY RUN — Entity Extraction Pipeline")
        log.info(f"Base dir: {args.base_dir}")
        log.info(f"Files: {len(get_all_transcription_files(args.base_dir))}")
    else:
        run_extraction(args)


if __name__ == "__main__":
    main()

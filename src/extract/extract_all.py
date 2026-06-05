#!/usr/bin/env python3
"""Integrated LLM extraction from call transcription text.

Replaces the 3 separate extraction scripts (TODO, entity, schedule)
with a single unified prompt that extracts everything in one API call.

Usage:
    python extract_all.py                    # Full incremental run
    python extract_all.py --start-batch 10  # Resume from batch 10
    python extract_all.py --dry-run          # Validate without API calls


Config:
    Uses LLM_API_KEY (or legacy ZAI_API_KEY) for API authentication.
    State: memory/state/integrated_extraction/
"""

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import EXIT_CONFIG, EXIT_OK, EXIT_PARTIAL, TRANSCRIPT_DIR, WORKSPACE
from src.pipeline.utils import compress_transcript, fallback_summary

from .client import call_llm_extract, get_llm_config
from .prompt import setup_langfuse
from .state import (
    collect_new_appointments,
    compute_file_hash,
    load_checkpoint,
    load_notification_state,
    load_persistent_todos,
    load_processed_index,
    notify_new_items,
    print_todo_alert,
    save_batch_result,
    save_checkpoint,
    save_processed_index,
    sync_todos_to_persistent,
    track_notified,
)

log = logging.getLogger(__name__)

# Lazy import for fast_score_transcript (avoids circular / heavy init at module load)
_fast_score_fn = None
def _get_fast_score():
    global _fast_score_fn
    if _fast_score_fn is None:
        try:
            from src.knowledge.signal_detector import fast_score_transcript

            _fast_score_fn = fast_score_transcript
        except ImportError:
            def default_fast_score(text):
                return {
                    "score": 1.0, "band": "definite_keep",
                    "should_process": True, "signals": {}, "drop_reason": None,
                }
            _fast_score_fn = default_fast_score
    return _fast_score_fn

KST = timezone(timedelta(hours=9))

# Pipeline config
DEFAULT_BASE_DIR = str(TRANSCRIPT_DIR)
DEFAULT_STATE_DIR = WORKSPACE / "memory" / "state" / "integrated_extraction"
DEFAULT_BATCH_SIZE = 5            # files per API run (reduced for ZAI rate limit stability)
MAX_CONTENT_CHARS = 12000        # P1-4: GLM ctx 기준 여유 있음
DEFAULT_API_DELAY = 5.0          # seconds between API calls (ZAI rate limit safe)


# --- Pipeline orchestration ---

class IntegratedPipeline:
    """Unified extraction pipeline: TODO + Entity + Schedule in one LLM call."""

    def __init__(self, args):
        self.base_dir = Path(args.base_dir)
        self.state_dir = Path(args.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = args.batch_size
        self.api_delay = args.api_delay
        self.json_output = getattr(args, "json", False)
        self.today_only = getattr(args, 'today', False)
        self.start_batch_override = getattr(args, 'start_batch', 0)  # P2-C8
        self.run_id = f"run_{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        # P0-2: --today 모드와 전체 모드 체크포인트 분리
        if self.today_only:
            self.checkpoint_file = self.state_dir / "checkpoint_today.json"
        else:
            self.checkpoint_file = self.state_dir / "checkpoint.json"
        self.processed_index_file = self.state_dir / "processed_files.json"
        self.stats = {
            "summary": 0, "todos": 0, "appointments": 0,
            "entities": 0, "products": 0, "money": 0,
            "risks": 0, "corrections": 0, "batches_done": 0, "errors": 0,
        }
        self.notification_state_file = self.state_dir / "notification_state.json"
        self._last_new_todos = []
        self._last_notifications = []
        self._telegram_notified_new_todos = False

    def get_transcription_files(self) -> list:
        """Get files to process. If --today, only today's files. Skips already processed."""
        all_files = sorted(self.base_dir.glob("*.txt"))
        all_files = [f for f in all_files if f.stat().st_size > 50]

        # Filter by today if requested
        if self.today_only:
            today_str = datetime.now(KST).strftime("%Y%m%d")
            all_files = [f for f in all_files if today_str in f.stem]
            log.info("--today mode: filtering for date %s", today_str)

        # Skip already processed files (same hash = unchanged)
        processed = load_processed_index(self.processed_index_file)
        new_files = []
        for f in all_files:
            fhash = compute_file_hash(f)
            if processed.get(f.stem) != fhash:
                new_files.append(f)

        skipped = len(all_files) - len(new_files)
        if skipped > 0:
            log.info("Skipped %d already-processed (unchanged) files", skipped)

        return new_files

    def _process_single_file(self, file_path, api_key, processed):
        """Process one file through fast-score, compression, LLM, fallback."""
        stem = file_path.stem
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            self.stats["errors"] += 1
            return {"file": stem}, {"file": stem, "error": "empty"}, None, False

        # Phase 1: fast_score pre-filter
        fast_score = _get_fast_score()(content)
        if not fast_score.get("should_process", True):
            self.stats["fast_score_dropped"] = self.stats.get("fast_score_dropped", 0) + 1
            fhash = compute_file_hash(file_path)
            processed[stem] = fhash
            result = {
                "file": stem, "source": str(file_path),
                "file_hash": fhash, "status": "skipped_fast_score",
                "fast_score": fast_score,
            }
            return result, None, None, True

        # Phase 2: compress
        content = compress_transcript(content, budget=MAX_CONTENT_CHARS)
        result = call_llm_extract(api_key, content, run_id=self.run_id, lf_available=self._lf_available)

        # Phase 3: fallback if LLM failed
        if not result or result.get("parse_error"):
            fallback = fallback_summary(content, source=stem)
            fallback["fast_score"] = fast_score
            fhash = compute_file_hash(file_path)
            file_result = {
                "file": stem, "source": str(file_path),
                "file_hash": fhash, "status": "fallback", **fallback,
            }
            processed[stem] = fhash
            self.stats["fallbacks"] = self.stats.get("fallbacks", 0) + 1
            return file_result, None, None, True

        if not result.get("parse_error"):
            fhash = compute_file_hash(file_path)
            file_result = {
                "file": stem, "source": str(file_path),
                "file_hash": fhash, "status": "ok", **result,
            }
            processed[stem] = fhash
            self._update_stats_from_result(result)
            return file_result, None, None, True
        else:
            self.stats["errors"] += 1
            return {"file": stem}, {"file": stem, "error": "api_or_parse_failed"}, None, False

    def _update_stats_from_result(self, result):
        """Update stats counters from a successful extraction result."""
        if result.get("summary", {}).get("one_line"):
            self.stats["summary"] += 1
        self.stats["todos"] += len(result.get("todos", []))
        self.stats["appointments"] += len(result.get("appointments", []))
        self.stats["entities"] += len(result.get("entities", []))
        self.stats["products"] += len(result.get("products", []))
        self.stats["money"] += len(result.get("money", []))
        self.stats["risks"] += len(result.get("risks", []))
        self.stats["corrections"] += len(result.get("corrections", []))

    def _notify_batch(self, batch_results, batch_ok_stems):
        """Send notifications for batch results."""
        new_todo_count, new_todos = sync_todos_to_persistent(batch_results, self.run_id)
        self._last_new_todos = new_todos
        self.stats["new_todos"] = self.stats.get("new_todos", 0) + new_todo_count

        notification_state = load_notification_state(self.notification_state_file)
        persistent_data = load_persistent_todos()
        active_todos = [
            t for t in persistent_data.get("todos", {}).values()
            if isinstance(t, dict) and t.get("status", "active") == "active"
        ]
        new_appointments = collect_new_appointments(batch_results, notification_state)
        notifications = notify_new_items(
            new_todos, new_appointments=new_appointments, active_todos=active_todos,
        )
        if notifications or new_todos or new_appointments:
            self._last_notifications.extend(notifications)
            track_notified(self.notification_state_file, new_todos, new_appointments, notifications)
            if notifications and any(
                n.get("ok") for n in notifications if n.get("kind") == "new_items"
            ):
                if new_todos:
                    self._telegram_notified_new_todos = True

    def _finalize_batch(self, batch_idx, total_batches, batch_files, batch_results, batch_errors):
        """Save batch results and checkpoint."""
        if batch_errors:
            save_batch_result(
                self.state_dir, batch_idx, batch_files,
                batch_results, batch_errors, "partial", self.run_id,
            )
        else:
            save_batch_result(
                self.state_dir, batch_idx, batch_files,
                batch_results, batch_errors, "done", self.run_id,
            )
            self.stats["batches_done"] += 1
            save_checkpoint(
                self.checkpoint_file, batch_idx, total_batches, self.stats, self.run_id,
            )

        log.info(
            "    Done: %d todos, %d entities, %d products, %d money, %d risks, %d corrections | %d errors",
            self.stats['todos'], self.stats['entities'], self.stats['products'],
            self.stats['money'], self.stats['risks'], self.stats['corrections'], len(batch_errors),
        )

    def _print_final_summary(self, total_batches):
        """Print the final run summary."""
        log.info("\n" + "=" * 60)
        log.info("INTEGRATED EXTRACTION COMPLETE")
        log.info("=" * 60)
        log.info("Run ID:       %s", self.run_id)
        log.info("Batches:      %d/%d", self.stats['batches_done'], total_batches)
        log.info("Summaries:    %d", self.stats['summary'])
        log.info("TODOs:        %d", self.stats['todos'])
        log.info("Appointments: %d", self.stats['appointments'])
        log.info("Entities:     %d", self.stats['entities'])
        log.info("Products:     %d", self.stats['products'])
        log.info("Money:        %d", self.stats['money'])
        log.info("Risks:        %d", self.stats['risks'])
        log.info("Corrections:  %d", self.stats['corrections'])
        log.info("Errors:       %d", self.stats['errors'])
        log.info("Dropped:      %d (fast_score pre-filter)", self.stats.get('fast_score_dropped', 0))
        log.info("Fallbacks:    %d (LLM failed, heuristic)", self.stats.get('fallbacks', 0))
        log.info("New TODOs:    %d (synced to persistent_todos.json)", self.stats.get('new_todos', 0))
        if self._last_notifications:
            log.info("Notifications: %d attempted", len(self._last_notifications))
        log.info("~60-70%% fewer LLM calls vs. separate scripts")
        log.info("=" * 60)

    def _resolve_start_batch(self, total_batches):
        """Determine starting batch index."""
        if self.today_only:
            start_batch = self.start_batch_override
            log.info("--today mode: always starting from batch %d (checkpoint ignored for batch index)", start_batch)
        else:
            start_batch = max(
                load_checkpoint(self.checkpoint_file, today_only=self.today_only),
                self.start_batch_override,
            )
        log.info("Total batches: %d, starting from: %d", total_batches, start_batch)
        log.info("Run ID: %s", self.run_id)
        return start_batch

    def _should_skip_batch(self, batch_idx):
        """Check if a batch was already processed (non-today mode only)."""
        if self.today_only:
            return False
        batch_file = self.state_dir / f"batch_{batch_idx:04d}.json"
        if not batch_file.exists():
            return False
        try:
            existing = json.loads(batch_file.read_text(encoding="utf-8"))
            return existing.get("status") == "done"
        except Exception as exc:
            logging.debug("Failed to inspect existing batch %s: %s", batch_file, exc)
            return False

    def _process_batch(self, batch_idx, total_batches, batch_files, api_key):
        """Process a single batch of files through the pipeline."""
        log.info("  [%04d/%04d] Processing %d files...", batch_idx, total_batches - 1, len(batch_files))

        batch_results = []
        batch_errors = []
        processed = load_processed_index(self.processed_index_file)

        for file_path in batch_files:
            try:
                file_result, error, _, ok = self._process_single_file(
                    file_path, api_key, processed,
                )
                if error:
                    batch_errors.append(error)
                if file_result:
                    batch_results.append(file_result)
            except Exception as e:
                log.error("%s: %s", file_path.stem, e)
                batch_errors.append({"file": file_path.stem, "error": str(e)})
                self.stats["errors"] += 1
            time.sleep(self.api_delay)

        save_processed_index(self.processed_index_file, processed)
        self._notify_batch(batch_results, [])
        self._finalize_batch(batch_idx, total_batches, batch_files, batch_results, batch_errors)

    def run(self):
        api_key = get_llm_config()["api_key"]
        if not api_key:
            log.error("LLM_API_KEY environment variable not set.")
            sys.exit(EXIT_CONFIG)

        files = self.get_transcription_files()
        total_files = len(files)
        log.info("Found %d transcription files", total_files)
        if total_files == 0:
            log.info("No files. Exiting.")
            return

        total_batches = (total_files + self.batch_size - 1) // self.batch_size
        start_batch = self._resolve_start_batch(total_batches)

        try:
            self._run_batches(start_batch, total_batches, total_files, files, api_key)
        except KeyboardInterrupt:
            log.warning("\nInterrupted at batch %d. Run ID: %s", batch_idx, self.run_id)
            log.info("Resume with: python extract_all.py --start-batch %d", batch_idx)
            save_checkpoint(self.checkpoint_file, batch_idx, total_batches, self.stats, self.run_id)

        self._print_final_summary(total_batches)

        if self.stats.get("new_todos", 0) > 0 and not self._telegram_notified_new_todos:
            print_todo_alert()

        if self.json_output:
            log.info(json.dumps({"run_id": self.run_id, "stats": self.stats}, ensure_ascii=False, sort_keys=True))
        sys.exit(EXIT_PARTIAL if self.stats["errors"] > 0 else EXIT_OK)

    def _run_batches(self, start_batch, total_batches, total_files, files, api_key):
        """Execute all batches from start_batch to end."""
        batch_idx = start_batch
        for batch_idx in range(start_batch, total_batches):
            if self._should_skip_batch(batch_idx):
                log.info("  [%04d] SKIP (already done)", batch_idx)
                continue

            batch_start_idx = batch_idx * self.batch_size
            batch_end_idx = min(batch_start_idx + self.batch_size, total_files)
            batch_files = files[batch_start_idx:batch_end_idx]

            self._process_batch(batch_idx, total_batches, batch_files, api_key)

            if batch_idx < total_batches - 1:
                time.sleep(self.api_delay)


def dry_run(args):
    """Validate pipeline setup."""
    base_dir = Path(args.base_dir)
    state_dir = Path(args.state_dir)

    log.info("=" * 60)
    log.info("DRY RUN — Integrated Extraction Pipeline")
    log.info("=" * 60)

    if not base_dir.exists():
        log.error(f"Base directory not found: {base_dir}")
        return False
    log.info(f"OK: Base directory: {base_dir}")

    files = sorted(base_dir.glob("*.txt"))
    files = [f for f in files if f.stat().st_size > 50]
    log.info(f"OK: Found {len(files)} transcription files")

    total_batches = (len(files) + args.batch_size - 1) // args.batch_size
    log.info(f"OK: {total_batches} batches (size={args.batch_size})")

    state_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"OK: State directory: {state_dir}")

    # Check API key
    try:
        config = get_llm_config()
        key = config["api_key"]
        if not key:
            log.error("LLM_API_KEY NOT set")
        log.info(f"OK: LLM model: {config['model']}")
        log.info(f"OK: LLM base URL: {config['base_url']}")
    except Exception as e:
        log.warning(f"Could not check API key: {e}")

    completed = len(list(state_dir.glob("batch_*.json")))
    log.info(f"INFO: {completed} batches already completed")

    if files:
        sample = files[0]
        content = sample.read_text(encoding="utf-8")
        log.info(f"\nSample: {sample.name} ({len(content)} chars)")
        log.info(f"  Preview: {content[:80].strip()}...")

    log.info("\nDry run PASSED.")
    log.info("Command: python extract_all.py")
    return True


def main():
    parser = argparse.ArgumentParser(description="Integrated LLM Extraction Pipeline (TODO + Entity + Schedule)")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--api-delay", type=float, default=DEFAULT_API_DELAY)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--today", action="store_true", help="Only process today's files")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary JSON")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")

    if args.dry_run:
        dry_run(args)
    else:
        pipeline = IntegratedPipeline(args)
        pipeline._lf_available = setup_langfuse()
        pipeline.run()


if __name__ == "__main__":
    main()

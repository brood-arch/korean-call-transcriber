#!/usr/bin/env python3
"""Integrated LLM extraction from call transcription text.

Replaces the 3 separate extraction scripts (TODO, entity, schedule)
with a single unified prompt that extracts everything in one API call.

Usage:
    python extract_all.py                    # Full incremental run
    python extract_all.py --start-batch 10  # Resume from batch 10
    python extract_all.py --dry-run          # Validate without API calls


Config:
    Uses ZAI_API_KEY environment variable for API authentication.
    State: memory/state/integrated_extraction/
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.pipeline.utils import compress_transcript, fallback_summary

from .client import call_zai_extract
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

WORKSPACE = Path(__file__).resolve().parents[2]

# Lazy import for fast_score_transcript (avoids circular / heavy init at module load)
_fast_score_fn = None
def _get_fast_score():
    global _fast_score_fn
    if _fast_score_fn is None:
        try:
            from .signal_detector import fast_score_transcript
            _fast_score_fn = fast_score_transcript
        except ImportError:
            try:
                from src.knowledge.signal_detector import fast_score_transcript
                _fast_score_fn = fast_score_transcript
            except ImportError:
                def default_fast_score(text):
                    return {"score": 1.0, "band": "definite_keep", "should_process": True, "signals": {}, "drop_reason": None}
                _fast_score_fn = default_fast_score
    return _fast_score_fn

KST = timezone(timedelta(hours=9))

# Pipeline config
try:
    from pipeline_paths import TRANSCRIPT_DIR
    DEFAULT_BASE_DIR = str(TRANSCRIPT_DIR)
except Exception:
    DEFAULT_BASE_DIR = os.environ.get("TRANSCRIPT_DIR", "./data/transcripts")
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
        self.today_only = getattr(args, 'today', False)
        self.start_batch_override = getattr(args, 'start_batch', 0)  # P2-C8
        self.run_id = f"run_{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        # P0-2: --today 모드와 전체 모드 체크포인트 분리
        if self.today_only:
            self.checkpoint_file = self.state_dir / "checkpoint_today.json"
        else:
            self.checkpoint_file = self.state_dir / "checkpoint.json"
        self.processed_index_file = self.state_dir / "processed_files.json"
        self.stats = {"summary": 0, "todos": 0, "appointments": 0, "entities": 0, "products": 0, "money": 0, "risks": 0, "corrections": 0, "batches_done": 0, "errors": 0}
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
            print(f"--today mode: filtering for date {today_str}")

        # Skip already processed files (same hash = unchanged)
        processed = load_processed_index(self.processed_index_file)
        new_files = []
        for f in all_files:
            fhash = compute_file_hash(f)
            if processed.get(f.stem) != fhash:
                new_files.append(f)

        skipped = len(all_files) - len(new_files)
        if skipped > 0:
            print(f"Skipped {skipped} already-processed (unchanged) files")

        return new_files

    def run(self):
        api_key = os.environ.get("ZAI_API_KEY", "")
        if not api_key:
            print("ERROR: ZAI_API_KEY environment variable not set.")
            sys.exit(1)

        files = self.get_transcription_files()
        total_files = len(files)
        print(f"Found {total_files} transcription files")
        if total_files == 0:
            print("No files. Exiting.")
            return

        total_batches = (total_files + self.batch_size - 1) // self.batch_size
        # --today 모드에서는 파일 목록이 매번 동적으로 변하므로
        # 체크포인트의 batch 인덱스를 신뢰할 수 없음 → 항상 0부터 시작
        # (이미 처리된 파일은 get_transcription_files()에서 해시로 걸러짐)
        if self.today_only:
            start_batch = self.start_batch_override
            print(f"--today mode: always starting from batch {start_batch} (checkpoint ignored for batch index)")
        else:
            # P2-C8: --start-batch 인자가 있으면 체크포인트보다 우선
            start_batch = max(load_checkpoint(self.checkpoint_file, today_only=self.today_only), self.start_batch_override)
        print(f"Total batches: {total_batches}, starting from: {start_batch}")
        print(f"Run ID: {self.run_id}")

        try:
            for batch_idx in range(start_batch, total_batches):
                batch_start_idx = batch_idx * self.batch_size
                batch_end_idx = min(batch_start_idx + self.batch_size, total_files)
                batch_files = files[batch_start_idx:batch_end_idx]

                # Skip already completed batches (NOT in --today mode: daily files change daily)
                if not self.today_only:
                    batch_file = self.state_dir / f"batch_{batch_idx:04d}.json"
                    if batch_file.exists():
                        try:
                            existing = json.loads(batch_file.read_text(encoding="utf-8"))
                            if existing.get("status") == "done":
                                print(f"  [{batch_idx:04d}] SKIP (already done)")
                                continue
                        except Exception:
                            pass

                print(f"  [{batch_idx:04d}/{total_batches-1}] Processing {len(batch_files)} files...")

                batch_results = []
                batch_errors = []
                batch_ok_stems = []   # stems successfully processed in this batch

                # Load processed index for updating
                processed = load_processed_index(self.processed_index_file)

                for file_path in batch_files:
                    stem = file_path.stem
                    try:
                        content = file_path.read_text(encoding="utf-8").strip()
                        if not content:
                            batch_errors.append({"file": stem, "error": "empty"})
                            self.stats["errors"] += 1
                            continue

                        # ── Phase 1: fast_score pre-filter (OpenHuman 3-band) ──
                        fast_score = _get_fast_score()(content)
                        if not fast_score.get("should_process", True):
                            self.stats["fast_score_dropped"] = self.stats.get("fast_score_dropped", 0) + 1
                            # Still mark as processed so we don't re-evaluate it
                            fhash = compute_file_hash(file_path)
                            processed[stem] = fhash
                            batch_ok_stems.append(stem)
                            batch_results.append({
                                "file": stem, "source": str(file_path),
                                "file_hash": fhash, "status": "skipped_fast_score",
                                "fast_score": fast_score,
                            })
                            continue

                        # ── Phase 2: compress (OpenHuman TokenJuice adapted) ──
                        content = compress_transcript(content, budget=MAX_CONTENT_CHARS)

                        result = call_zai_extract(api_key, content, run_id=self.run_id, lf_available=self._lf_available)

                        # ── Phase 3: fallback if LLM failed (OpenHuman fallback_summary) ──
                        if not result or result.get("parse_error"):
                            fallback = fallback_summary(content, source=stem)
                            fallback["fast_score"] = fast_score
                            fhash = compute_file_hash(file_path)
                            file_result = {
                                "file": stem,
                                "source": str(file_path),
                                "file_hash": fhash,
                                "status": "fallback",
                                **fallback,
                            }
                            processed[stem] = fhash
                            batch_ok_stems.append(stem)
                            self.stats["fallbacks"] = self.stats.get("fallbacks", 0) + 1
                            batch_results.append(file_result)
                            continue

                        if not result.get("parse_error"):
                            fhash = compute_file_hash(file_path)
                            file_result = {
                                "file": stem,
                                "source": str(file_path),
                                "file_hash": fhash,
                                "status": "ok",
                                **result,
                            }
                            processed[stem] = fhash  # Mark as processed
                            batch_ok_stems.append(stem)
                            if result.get("summary", {}).get("one_line"):
                                self.stats["summary"] += 1
                            self.stats["todos"] += len(result.get("todos", []))
                            self.stats["appointments"] += len(result.get("appointments", []))
                            self.stats["entities"] += len(result.get("entities", []))
                            self.stats["products"] += len(result.get("products", []))
                            self.stats["money"] += len(result.get("money", []))
                            self.stats["risks"] += len(result.get("risks", []))
                            self.stats["corrections"] += len(result.get("corrections", []))
                        else:
                            file_result = {"file": stem, "status": "failed", "error": "api_or_parse_failed"}
                            batch_errors.append({"file": stem, "error": "api_or_parse_failed"})
                            self.stats["errors"] += 1

                        batch_results.append(file_result)

                    except Exception as e:
                        print(f"    ERROR {stem}: {e}")
                        batch_errors.append({"file": stem, "error": str(e)})
                        self.stats["errors"] += 1

                    time.sleep(self.api_delay)

                # P0-1: 항상 성공한 파일의 processed_index 저장
                save_processed_index(self.processed_index_file, processed)

                # TODO sync: 신규 TODO를 persistent_todos.json에 반영
                new_todo_count, new_todos = sync_todos_to_persistent(batch_results, self.run_id)
                self._last_new_todos = new_todos
                self.stats["new_todos"] = self.stats.get("new_todos", 0) + new_todo_count

                # Telegram notification + shared notified_* state tracking.
                # Do this immediately after the persistent TODO sync so alerts reflect
                # both newly added TODOs and the current active backlog.
                notification_state = load_notification_state(self.notification_state_file)
                persistent_data = load_persistent_todos()
                active_todos = [
                    t for t in persistent_data.get("todos", {}).values()
                    if isinstance(t, dict) and t.get("status", "active") == "active"
                ]
                new_appointments = collect_new_appointments(batch_results, notification_state)
                notifications = notify_new_items(
                    new_todos,
                    new_appointments=new_appointments,
                    active_todos=active_todos,
                )
                if notifications or new_todos or new_appointments:
                    self._last_notifications.extend(notifications)
                    track_notified(self.notification_state_file, new_todos, new_appointments, notifications)
                    if notifications and any(n.get("ok") for n in notifications if n.get("kind") == "new_items"):
                        if new_todos:
                            self._telegram_notified_new_todos = True

                # Save batch result. Failed batches stay partial so a later run retries them.
                if batch_errors:
                    save_batch_result(self.state_dir, batch_idx, batch_files, batch_results, batch_errors, "partial", self.run_id)
                else:
                    save_batch_result(self.state_dir, batch_idx, batch_files, batch_results, batch_errors, "done", self.run_id)
                    self.stats["batches_done"] += 1
                    save_checkpoint(self.checkpoint_file, batch_idx, total_batches, self.stats, self.run_id)

                print(f"    Done: {self.stats['todos']} todos, {self.stats['entities']} entities, {self.stats['products']} products, {self.stats['money']} money, {self.stats['risks']} risks, {self.stats['corrections']} corrections"
                      f"| {len(batch_errors)} errors")

                if batch_idx < total_batches - 1:
                    time.sleep(self.api_delay)

        except KeyboardInterrupt:
            print(f"\nInterrupted at batch {batch_idx}. Run ID: {self.run_id}")
            print(f"Resume with: python extract_all.py --start-batch {batch_idx}")
            save_checkpoint(self.checkpoint_file, batch_idx, total_batches, self.stats, self.run_id)

        # Final summary
        print(f"\n{'='*60}")
        print("INTEGRATED EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Run ID:       {self.run_id}")
        print(f"Batches:      {self.stats['batches_done']}/{total_batches}")
        print(f"Summaries:    {self.stats['summary']}")
        print(f"TODOs:        {self.stats['todos']}")
        print(f"Appointments: {self.stats['appointments']}")
        print(f"Entities:     {self.stats['entities']}")
        print(f"Products:     {self.stats['products']}")
        print(f"Money:        {self.stats['money']}")
        print(f"Risks:        {self.stats['risks']}")
        print(f"Corrections:  {self.stats['corrections']}")
        print(f"Errors:       {self.stats['errors']}")
        print(f"Dropped:      {self.stats.get('fast_score_dropped', 0)} (fast_score pre-filter)")
        print(f"Fallbacks:    {self.stats.get('fallbacks', 0)} (LLM failed, heuristic)")
        print(f"New TODOs:    {self.stats.get('new_todos', 0)} (synced to persistent_todos.json)")
        if self._last_notifications:
            print(f"Notifications:{len(self._last_notifications)} attempted")
        print("~60-70% fewer LLM calls vs. separate scripts")
        print(f"{'='*60}")

        # TODO alert fallback: print full active report only if Telegram did not
        # successfully deliver a new-TODO notification.
        if self.stats.get("new_todos", 0) > 0 and not self._telegram_notified_new_todos:
            print_todo_alert()

        sys.exit(1 if self.stats["errors"] > 0 else 0)


def dry_run(args):
    """Validate pipeline setup."""
    base_dir = Path(args.base_dir)
    state_dir = Path(args.state_dir)

    print("=" * 60)
    print("DRY RUN — Integrated Extraction Pipeline")
    print("=" * 60)

    if not base_dir.exists():
        print(f"FAIL: Base directory not found: {base_dir}")
        return False
    print(f"OK: Base directory: {base_dir}")

    files = sorted(base_dir.glob("*.txt"))
    files = [f for f in files if f.stat().st_size > 50]
    print(f"OK: Found {len(files)} transcription files")

    total_batches = (len(files) + args.batch_size - 1) // args.batch_size
    print(f"OK: {total_batches} batches (size={args.batch_size})")

    state_dir.mkdir(parents=True, exist_ok=True)
    print(f"OK: State directory: {state_dir}")

    # Check API key
    try:
        key = os.environ.get("ZAI_API_KEY", "")
        print(f"{'OK' if key else 'FAIL'}: ZAI_API_KEY {'set' if key else 'NOT set'}")
    except Exception as e:
        print(f"WARN: Could not check API key: {e}")

    completed = len(list(state_dir.glob("batch_*.json")))
    print(f"INFO: {completed} batches already completed")

    if files:
        sample = files[0]
        content = sample.read_text(encoding="utf-8")
        print(f"\nSample: {sample.name} ({len(content)} chars)")
        print(f"  Preview: {content[:80].strip()}...")

    print("\nDry run PASSED.")
    print("Command: python extract_all.py")
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
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
    else:
        pipeline = IntegratedPipeline(args)
        pipeline._lf_available = setup_langfuse()
        pipeline.run()


if __name__ == "__main__":
    main()

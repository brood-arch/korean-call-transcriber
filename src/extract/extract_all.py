"""Integrated LLM extraction from call transcription text.

Replaces separate extraction scripts (TODO, entity, schedule) with a single
unified prompt that extracts everything in one API call.

Extracts: Summary, TODOs, Appointments, Entities, Products, Money, Risks, Corrections.

Usage:
    python extract_all.py                    # Full incremental run
    python extract_all.py --start-batch 10  # Resume from batch 10
    python extract_all.py --dry-run          # Validate without API calls

Configuration via environment variables:
    TRANSCRIPT_DIR   — directory containing transcript .txt files
    LLM_API_KEY      — API key for the LLM provider
    LLM_BASE_URL     — Base URL for OpenAI-compatible API
    LLM_MODEL        — Model name (default: glm-5.1)
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib.request
import urllib.error

KST = timezone(timedelta(hours=9))

# ── Configuration ───────────────────────────────────────────────────────
TRANSCRIPT_DIR = Path(os.environ.get("TRANSCRIPT_DIR", "data/transcripts"))
STATE_DIR = Path("state/integrated_extraction")
BATCH_SIZE = 5
MAX_CONTENT_CHARS = 12000
API_DELAY = 5.0
MAX_RETRIES = 4
RETRY_BACKOFF = [5, 15, 45, 90]

LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.example.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "glm-5.1")

# ── Unified extraction prompt template ──────────────────────────────────
UNIFIED_EXTRACT_PROMPT = """Extract the following 8 categories from this call transcription:

1. Summary — one-line summary + detailed bullets
2. TODOs — tasks the user needs to handle
3. Appointments — confirmed dates/times
4. Entities — people, organizations, locations, phone numbers, products
5. Products — products/parts/specifications/quantities
6. Money — prices, deposits, balances, shipping costs
7. Risks — customer complaints, deadline delays, unpaid amounts
8. Corrections — transcription error corrections

Output format (strict JSON only, no markdown or explanation):
{{
  "summary": {{
    "one_line": "One-line summary",
    "details": ["detail 1", "detail 2"],
    "call_type": "order|delivery|as|quote|payment|schedule|internal|personal|unknown",
    "overall_confidence": 0.0~1.0
  }},
  "todos": [
    {{
      "title": "Task title",
      "owner": "user|counterpart|unknown",
      "priority": "high|medium|low",
      "status": "new|in_progress|waiting|done|cancelled",
      "due_date": "YYYY-MM-DD or null",
      "context": "Short context (1 sentence)",
      "confidence": 0.0~1.0
    }}
  ],
  "appointments": [
    {{
      "title": "Appointment title",
      "date": "YYYY-MM-DD",
      "time": "HH:MM or null",
      "timezone": "Asia/Seoul",
      "location": "Location or null",
      "participants": ["participant1"],
      "confidence": 0.0~1.0
    }}
  ],
  "entities": [
    {{
      "name": "Entity name",
      "type": "Person|Organization|Location|PhoneNumber|Product|Project|Event|Contract|Other",
      "role": "customer|supplier|employee|carrier|unknown",
      "confidence": 0.0~1.0
    }}
  ],
  "products": [
    {{
      "name": "Product name",
      "category": "category or unknown",
      "spec": "Specification or null",
      "quantity": {{"value": 0, "unit": "unit or null"}},
      "action": "quote|order|deliver|repair|check_stock|manufacture|unknown",
      "confidence": 0.0~1.0
    }}
  ],
  "money": [
    {{
      "amount": 0,
      "currency": "KRW",
      "kind": "price|deposit|balance|shipping|discount|tax|unknown",
      "payment_status": "paid|unpaid|partial|unknown",
      "confidence": 0.0~1.0
    }}
  ],
  "risks": [
    {{
      "severity": "high|medium|low",
      "type": "missed_deadline|payment_delay|customer_complaint|stock_shortage|quality_issue|other",
      "description": "Risk description",
      "recommended_action": "Recommended action or null",
      "confidence": 0.0~1.0
    }}
  ],
  "corrections": [
    {{
      "original": "Original text",
      "corrected": "Corrected text",
      "reason": "exact_rule|alias|contextual|spacing|number_normalization|other",
      "confidence": 0.0~1.0
    }}
  ]
}}

Rules:
- Only include actionable tasks, not information sharing or greetings
- Mark completed actions as status=done
- Entities should be unique (same person/org = one entry)
- Amounts in integer KRW (e.g., "120만원" → 1200000)
- If nothing to extract, use empty arrays
- OUTPUT ONLY JSON. No markdown, explanation, or comments.

Transcription:
{content}"""


def call_llm(api_key: str, content: str) -> dict | None:
    """Call LLM with unified extraction prompt. Returns parsed dict or None."""
    prompt = UNIFIED_EXTRACT_PROMPT.replace("{content}", content[:MAX_CONTENT_CHARS], 1)

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8192,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")

    api_url = LLM_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(api_url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return parse_unified_response(text)

        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (2 ** attempt)
                print(f"    429 rate limit (attempt {attempt+1}/{MAX_RETRIES}), waiting {wait}s...")
                time.sleep(wait)
            elif e.code == 401:
                print(f"    Auth error")
                break
            else:
                wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
                time.sleep(wait)
        except urllib.error.URLError as e:
            wait = RETRY_BACKOFF[attempt] if attempt < len(RETRY_BACKOFF) else 30
            print(f"    Network error: {e}")
            time.sleep(wait)
        except Exception as e:
            print(f"    Unexpected error: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)

    return None


def parse_unified_response(text: str) -> dict:
    """Parse unified JSON response, handling markdown code blocks."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                return {"parse_error": True, "raw": cleaned[:500]}
        else:
            return {"parse_error": True, "raw": cleaned[:500]}

    # Validate and normalize each category
    return {
        "summary": _validate_summary(data.get("summary", {})),
        "todos": _validate_todos(data.get("todos", [])),
        "appointments": _validate_appointments(data.get("appointments", [])),
        "entities": _validate_entities(data.get("entities", [])),
        "products": _validate_products(data.get("products", [])),
        "money": _validate_money(data.get("money", [])),
        "risks": _validate_risks(data.get("risks", [])),
        "corrections": _validate_corrections(data.get("corrections", [])),
        "parse_error": False,
    }


def _validate_summary(item):
    if not isinstance(item, dict):
        return {}
    return {
        "one_line": str(item.get("one_line", ""))[:300],
        "details": [str(d)[:200] for d in item.get("details", []) if d][:5],
        "call_type": str(item.get("call_type", "unknown")),
        "overall_confidence": float(item.get("overall_confidence", 0.0)),
    }


def _validate_todos(items):
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("title"):
            result.append({
                "title": str(item["title"]).strip(),
                "owner": item.get("owner", "unknown"),
                "priority": item.get("priority", "medium"),
                "status": item.get("status", "new"),
                "due_date": str(item.get("due_date")) if item.get("due_date") else None,
                "context": str(item.get("context", ""))[:200],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_appointments(items):
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("title"):
            result.append({
                "title": str(item["title"]).strip(),
                "date": str(item.get("date")) if item.get("date") else None,
                "time": str(item.get("time")) if item.get("time") else None,
                "timezone": str(item.get("timezone", "Asia/Seoul")),
                "location": str(item.get("location")) if item.get("location") else None,
                "participants": [str(p) for p in item.get("participants", []) if p][:10],
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_entities(items):
    valid_types = {"Person", "Organization", "Location", "PhoneNumber", "Product", "Project", "Event", "Contract", "Other"}
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            result.append({
                "name": str(item["name"]).strip(),
                "type": item.get("type", "Other") if item.get("type") in valid_types else "Other",
                "role": item.get("role", "unknown"),
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_products(items):
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("name"):
            qty = item.get("quantity", {}) or {}
            result.append({
                "name": str(item["name"]).strip(),
                "category": item.get("category", "unknown"),
                "spec": str(item.get("spec")) if item.get("spec") else None,
                "quantity": {"value": int(qty.get("value", 0)), "unit": str(qty.get("unit", ""))},
                "action": item.get("action", "unknown"),
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_money(items):
    result = []
    for item in items:
        if isinstance(item, dict):
            try:
                amount = int(item.get("amount", 0))
            except (ValueError, TypeError):
                amount = 0
            result.append({
                "amount": amount,
                "currency": str(item.get("currency", "KRW")),
                "kind": item.get("kind", "unknown"),
                "payment_status": item.get("payment_status", "unknown"),
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_risks(items):
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("description"):
            result.append({
                "severity": item.get("severity", "medium"),
                "type": item.get("type", "other"),
                "description": str(item["description"])[:300],
                "recommended_action": str(item.get("recommended_action")) if item.get("recommended_action") else None,
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _validate_corrections(items):
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("original") and item.get("corrected"):
            result.append({
                "original": str(item["original"])[:200],
                "corrected": str(item["corrected"])[:200],
                "reason": item.get("reason", "other"),
                "confidence": float(item.get("confidence", 0.5)),
            })
    return result


def _compute_file_hash(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class IntegratedPipeline:
    """Unified extraction pipeline: TODO + Entity + Schedule in one LLM call."""

    def __init__(self, args):
        self.base_dir = Path(args.base_dir)
        self.state_dir = Path(args.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = args.batch_size
        self.api_delay = args.api_delay
        self.today_only = getattr(args, "today", False)
        self.start_batch = getattr(args, "start_batch", 0)
        self.run_id = f"run_{datetime.now(KST).strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_file = self.state_dir / "checkpoint.json"
        self.processed_index_file = self.state_dir / "processed_files.json"
        self.stats = {
            "summary": 0, "todos": 0, "appointments": 0, "entities": 0,
            "products": 0, "money": 0, "risks": 0, "corrections": 0,
            "batches_done": 0, "errors": 0,
        }

    def load_processed_index(self) -> dict:
        if self.processed_index_file.exists():
            try:
                return json.loads(self.processed_index_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def save_processed_index(self, index: dict):
        tmp = self.processed_index_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.processed_index_file)

    def get_transcription_files(self) -> list[Path]:
        """Get files to process. Skips already-processed (same hash) files."""
        all_files = sorted(self.base_dir.glob("*.txt"))
        all_files = [f for f in all_files if f.stat().st_size > 50]

        if self.today_only:
            today_str = datetime.now(KST).strftime("%Y%m%d")
            all_files = [f for f in all_files if today_str in f.stem]

        processed = self.load_processed_index()
        new_files = []
        for f in all_files:
            fhash = _compute_file_hash(f)
            if processed.get(f.stem) != fhash:
                new_files.append(f)

        return new_files

    def load_checkpoint(self) -> int:
        if self.checkpoint_file.exists():
            try:
                cp = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
                return cp.get("last_completed_batch", -1) + 1
            except Exception:
                pass
        return 0

    def save_checkpoint(self, batch_idx: int, total: int, stats: dict):
        data = {
            "last_completed_batch": batch_idx,
            "total_batches": total,
            "last_updated": datetime.now(KST).isoformat(),
            "run_id": self.run_id,
            "stats": stats,
        }
        tmp = self.checkpoint_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.checkpoint_file)

    def run(self):
        api_key = LLM_API_KEY or os.environ.get("LLM_API_KEY")
        if not api_key:
            print("ERROR: LLM_API_KEY not set. Export it or set LLM_API_KEY env var.")
            sys.exit(1)

        files = self.get_transcription_files()
        total_files = len(files)
        print(f"Found {total_files} transcription files")
        if total_files == 0:
            print("No files. Exiting.")
            return

        total_batches = (total_files + self.batch_size - 1) // self.batch_size
        start_batch = max(self.load_checkpoint(), self.start_batch)
        print(f"Total batches: {total_batches}, starting from: {start_batch}")

        try:
            for batch_idx in range(start_batch, total_batches):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, total_files)
                batch_files = files[batch_start:batch_end]

                print(f"  [{batch_idx:04d}/{total_batches-1}] Processing {len(batch_files)} files...")

                batch_results = []
                batch_errors = []
                processed = self.load_processed_index()

                for file_path in batch_files:
                    stem = file_path.stem
                    try:
                        content = file_path.read_text(encoding="utf-8").strip()
                        if not content:
                            batch_errors.append({"file": stem, "error": "empty"})
                            self.stats["errors"] += 1
                            continue

                        result = call_llm(api_key, content)
                        if result and not result.get("parse_error"):
                            fhash = _compute_file_hash(file_path)
                            file_result = {
                                "file": stem, "file_hash": fhash, "status": "ok", **result,
                            }
                            processed[stem] = fhash
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
                            file_result = {"file": stem, "status": "failed"}
                            batch_errors.append({"file": stem, "error": "api_or_parse_failed"})
                            self.stats["errors"] += 1

                        batch_results.append(file_result)

                    except Exception as e:
                        print(f"    ERROR {stem}: {e}")
                        batch_errors.append({"file": stem, "error": str(e)})
                        self.stats["errors"] += 1

                    time.sleep(self.api_delay)

                self.save_processed_index(processed)

                if not batch_errors:
                    self.stats["batches_done"] += 1
                    self.save_checkpoint(batch_idx, total_batches, self.stats)

                print(f"    Done: {self.stats['todos']} todos, {self.stats['entities']} entities | {len(batch_errors)} errors")

                if batch_idx < total_batches - 1:
                    time.sleep(self.api_delay)

        except KeyboardInterrupt:
            print(f"\nInterrupted at batch {batch_idx}. Resume with --start-batch {batch_idx}")
            self.save_checkpoint(batch_idx, total_batches, self.stats)

        # Final summary
        print(f"\n{'='*60}")
        print(f"INTEGRATED EXTRACTION COMPLETE")
        print(f"{'='*60}")
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
        print(f"{'='*60}")


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

    api_key = LLM_API_KEY or os.environ.get("LLM_API_KEY")
    print(f"{'OK' if api_key else 'FAIL'}: API key {'found' if api_key else 'NOT found'}")

    print(f"\nDry run PASSED.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Integrated LLM Extraction Pipeline")
    parser.add_argument("--base-dir", default=str(TRANSCRIPT_DIR))
    parser.add_argument("--state-dir", default=str(STATE_DIR))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--api-delay", type=float, default=API_DELAY)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--today", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        dry_run(args)
    else:
        pipeline = IntegratedPipeline(args)
        pipeline.run()


if __name__ == "__main__":
    main()

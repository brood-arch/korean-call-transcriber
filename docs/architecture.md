# Architecture

This repository contains a Korean call transcription pipeline with five main stages:

```text
audio files
  -> Whisper transcription
  -> optional alignment and speaker diarization
  -> transcript correction
  -> TODO, schedule, and entity extraction
  -> optional Obsidian/wiki export and recovery queue analysis

extended sources (v0.3.0):
  SMS messages  -> normalize -> extract TODOs
  Gmail inbox   -> classify  -> extract TODOs
  Naver Mail    -> IMAP archive -> extract TODOs
  Calendar      -> check events -> reminders
  all sources   -> persistent TODO store (Jaccard dedup)
                -> knowledge graph + signal detector
```

## Package Layout

- `src/transcribe/`: batch transcription, worker process, and diarization orchestration.
- `src/correct/`: transcript correction rules and Korean/English normalization.
- `src/extract/`: LLM-assisted TODO, schedule, and entity extraction.
- `src/sync/`: transcript export to an Obsidian-compatible vault.
- `src/pipeline/`: shared paths, JSON helpers, and health checks.
- `src/queue/`: deterministic gap analysis and retry queue generation.
- `tests/`: focused regression tests for health checks, gap analysis, and retry queue behavior.

## Configuration

Runtime paths and credentials are configured with environment variables. Copy `.env.example` to `.env` for local development and set only the values needed for your environment.

The code defaults to local relative directories:

- `data/audio`
- `output/transcripts`
- `state`
- `logs`

Generated data, models, and local credentials are intentionally ignored by Git.

## Extended Pipeline (v0.3.0)

- `src/integrations/`: external service integrations.
  - `gmail_classifier.py`: auto-classify Gmail inbox (ads → trash, important → highlight).
  - `email_todo_extract.py`: extract action items from incoming emails via LLM.
  - `calendar.py`: Google Calendar event checking via OAuth2.
  - `sms_handler.py`: placeholder for SMS-to-transcription pipeline integration.
  - `naver_mail.py`: IMAP-based Naver Mail archiver with structured JSON output.
- `src/todo/`: persistent TODO management.
  - `persistent_store.py`: Jaccard fuzzy dedup (≥ 0.55), same-source merge, completed tracking.
- `src/knowledge/`: knowledge graph and signal detection.
  - `graph.py`: entity relationship extraction and traversal.
  - `signal_detector.py`: 3-band fast scoring + idea/entity extraction from any text.
- `src/pipeline/minions_queue.py`: Postgres-backed durable job queue with fan-out, DAG, and crash recovery.
- `src/pipeline/validate_state.py`: automated state file existence, staleness, and integrity checks.

## Recovery Flow

`src.queue.gap_analyzer` classifies pipeline gaps into deterministic categories such as missing transcript, diarization failure, extraction pending, index pending, and sync pending.

`src.queue.retry_queue` converts those gaps into a JSONL queue that can be reviewed or processed by a conservative worker.

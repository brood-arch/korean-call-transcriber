# Architecture (v0.9.1)

This repository contains a Korean call transcription pipeline with multiple stages:

```text
audio files
  -> Whisper transcription
  -> optional alignment and speaker diarization
  -> transcript correction
  -> TODO, schedule, and entity extraction
  -> optional Obsidian/wiki export and recovery queue analysis

extended sources (v0.9.1):
  SMS messages  -> normalize -> extract TODOs
  Gmail inbox   -> classify  -> extract TODOs
  Naver Mail    -> IMAP archive -> extract TODOs
  Calendar      -> check events -> reminders
  all sources   -> persistent TODO store (Jaccard dedup, 14-day retention)
                -> knowledge graph + signal detector (3-band fast scoring)

pipeline health:
  gap_analyzer  -> classify pipeline gaps (missing transcript, diarization failure, etc.)
  retry_queue   -> JSONL-based retry queue for conservative worker processing
  validate_state -> state file existence, staleness, and integrity checks
  minions_queue -> Postgres-backed durable job queue with DAG and crash recovery
```

## Package Layout

- `src/transcribe/`: batch transcription, worker process, and diarization orchestration.
- `src/correct/`: transcript correction rules and Korean/English normalization.
- `src/extract/`: LLM-assisted TODO, schedule, and entity extraction.
- `src/sync/`: transcript export to an Obsidian-compatible vault.
- `src/pipeline/`: shared paths, JSON helpers, health checks, state validation, and durable job queue.
- `src/queue/`: deterministic gap analysis and retry queue generation.
- `src/knowledge/`: knowledge graph, entity extraction, and signal detection (3-band fast scoring).
- `src/todo/`: persistent TODO management with Jaccard fuzzy dedup and 14-day retention.
- `src/integrations/`: external service integrations (Gmail, Naver Mail, Calendar, SMS).
- `tests/`: focused regression tests for health checks, gap analysis, retry queue, and correction rules.

## Configuration

Runtime paths and credentials are configured with environment variables. Copy `.env.example` to `.env` for local development and set only the values needed for your environment.

The code defaults to vendor-neutral values:

- `LLM_BASE_URL`: `https://api.openai.com/v1`
- `LLM_MODEL`: `gpt-4o-mini`
- `KCT_AUDIO_DIR`: `data/audio`
- `KCT_TRANSCRIPT_DIR`: `output/transcripts`
- `KCT_STATE_DIR`: `state`
- `KCT_LOG_DIR`: `logs`

Generated data, models, and local credentials are intentionally ignored by Git.

## Extended Pipeline (v0.9.1)

- `src/integrations/`: external service integrations.
  - `gmail_classifier.py`: auto-classify Gmail inbox (ads → trash, important → highlight).
  - `email_todo_extract.py`: extract action items from incoming emails via LLM.
  - `calendar.py`: Google Calendar event checking via OAuth2.
  - `sms_handler.py`: placeholder for SMS-to-transcription pipeline integration.
  - `naver_mail.py`: IMAP-based Naver Mail archiver with structured JSON output.
- `src/todo/`: persistent TODO management.
  - `persistent_store.py`: Jaccard fuzzy dedup (≥ 0.55), same-source merge, completed tracking, 14-day retention.
- `src/knowledge/`: knowledge graph and signal detection.
  - `graph.py`: entity relationship extraction and traversal.
  - `signal_detector.py`: 3-band fast scoring + idea/entity extraction from any text.
- `src/pipeline/minions_queue.py`: Postgres-backed durable job queue with fan-out, DAG, and crash recovery.
- `src/pipeline/validate_state.py`: automated state file existence, staleness, and integrity checks.
- `src/extract/domain_corrections.json`: externalized domain-specific transcription correction rules (5 rules).
- `src/extract/extract_all.py`: unified extraction — single LLM call per batch for TODO, schedule, entity, product, money, risk, and correction extraction.

## Recovery Flow

`src.queue.gap_analyzer` classifies pipeline gaps into deterministic categories such as missing transcript, diarization failure, extraction pending, index pending, and sync pending.

`src.queue.retry_queue` converts those gaps into a JSONL queue that can be reviewed or processed by a conservative worker.

## Module Details

### Knowledge Module (`src/knowledge/`)

- `signal_detector.py`: 3-band fast scoring engine that automatically captures ideas, entities, and actionable signals from any text. Designed for real-time ingestion across all pipeline sources.
- `graph.py`: entity relationship extraction and traversal for building a local knowledge graph.

### TODO Module (`src/todo/`)

- `persistent_store.py`: Jaccard fuzzy dedup (≥ 0.55 threshold), same-source merge to collapse related action items, completed-item tracking, and 14-day retention policy.

### Queue Module (`src/queue/`)

- `gap_analyzer`: deterministic gap classification for pipeline health auditing.
- `retry_queue`: JSONL-based retry queue for conservative, reviewable worker processing.

### Pipeline Module (`src/pipeline/`)

- `health_check`: end-to-end pipeline health verification.
- `validate_state`: automated state file existence, staleness, and integrity checks with optional auto-fix.
- `minions_queue.py`: Postgres-backed durable job queue supporting fan-out, parent-child DAG dependencies, crash recovery, and priority scheduling.

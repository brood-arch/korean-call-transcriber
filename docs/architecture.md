# Architecture

This repository contains a Korean call transcription pipeline with five main stages:

```text
audio files
  -> Whisper transcription
  -> optional alignment and speaker diarization
  -> transcript correction
  -> TODO, schedule, and entity extraction
  -> optional Obsidian/wiki export and recovery queue analysis
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

## Recovery Flow

`src.queue.gap_analyzer` classifies pipeline gaps into deterministic categories such as missing transcript, diarization failure, extraction pending, index pending, and sync pending.

`src.queue.retry_queue` converts those gaps into a JSONL queue that can be reviewed or processed by a conservative worker.

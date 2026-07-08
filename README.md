# korean-call-transcriber

[![PyPI version](https://img.shields.io/pypi/v/korean-call-transcriber.svg)](https://pypi.org/project/korean-call-transcriber/) [![Python](https://img.shields.io/pypi/pyversions/korean-call-transcriber.svg)](https://pypi.org/project/korean-call-transcriber/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/brood-arch/korean-call-transcriber/actions/workflows/ci.yml/badge.svg)](https://github.com/brood-arch/korean-call-transcriber/actions/workflows/ci.yml)

Korean phone call transcription pipeline with Whisper + speaker diarization, auto TODO/schedule/entity extraction, and Obsidian sync.

```bash
pip install korean-call-transcriber
```

```bash
# GPU support (transcription)
pip install korean-call-transcriber[gpu]
```

## Features

- **🎙️ WhisperX Transcription** — faster-whisper (CTranslate2) for fast GPU-accelerated Korean speech-to-text
- **👥 Speaker Diarization** — pyannote-based 2-speaker identification with Korean honorific heuristics
- **📝 Unified LLM Extraction** — Single API call extracts: Summary, TODOs, Appointments, Entities, Products, Money, Risks, and Corrections
- **🔧 STT Correction Layer** — Persistent exact replacements and alias normalization with hot-reload
- **📊 Gap Analyzer** — Deterministic pipeline health check with cause taxonomy
- **🔄 Retry Queue** — JSONL-based atomic retry queue with exponential backoff
- **📓 Obsidian Sync** — Automatic transcript → markdown conversion with counterparty indexing
- **📧 Gmail Classifier** — Auto-classify inbox emails (ads → trash, important → highlight)
- **📋 Email TODO Extract** — Extract action items from incoming emails with LLM
- **📅 Calendar Integration** — Google Calendar event checking via OAuth2
- **💬 SMS Pipeline** — Placeholder module for SMS-to-transcription integration
- **📮 Naver Mail Archiver** — IMAP-based Naver Mail archiver with structured JSON output
- **✅ Persistent TODO Store** — Jaccard fuzzy-dedup, same-source merge, completed tracking
- **🧠 Knowledge Graph** — Entity relationship extraction and traversal (counterparty ↔ TODO ↔ event)
- **📡 Signal Detector** — 3-band fast scoring + idea/entity extraction from any text
- **⚙️ Minions Queue** — Postgres-backed durable job queue with fan-out, DAG, and crash recovery
- **🔍 State Validator** — Automated state file existence, staleness, and integrity checks

## ⚠️ Privacy Notice

This tool processes sensitive communications (calls, emails, SMS). **Transcript text is sent to the configured LLM API for extraction** — all other processing runs locally. See [PRIVACY.md](PRIVACY.md) for details and recommendations.

## Module Status

| Module | Status | Notes |
|--------|--------|-------|
| Transcription | Beta | WhisperX + diarization; GPU setup required |
| LLM Extraction | Beta | 8-category unified extraction; OpenAI-compatible API |
| STT Correction | Stable | Hot-reload rules |
| Obsidian Sync | Beta | Counterparty indexing; local vault paths required |
| Gmail Classifier | Experimental | Keyword-based classification; credentials required |
| Email TODO Extract | Experimental | LLM + rule-based; credentials required |
| Naver Mail | Experimental | IMAP archiving; Naver IMAP setup required |
| Calendar | Experimental | Google Calendar OAuth2 setup required |
| TODO Store | Stable | Jaccard fuzzy dedup |
| Knowledge Graph | Beta | Entity relationships |
| Signal Detector | Stable | 3-band fast scoring |
| Minions Queue | Beta | Postgres-backed; optional |
| State Validator | Stable | Integrity checks |
| SMS Handler | Placeholder | Integration pattern only |
| Retry Queue | Stable | JSONL + backoff |

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Audio Files     │────▶│  batch_transcribe │────▶│  Transcript .txt │
│  (*.m4a)         │     │  (WhisperX)       │     │                  │
└─────────────────┘     │  ├─ transcribe     │     └────────┬─────────┘
                        │  ├─ align          │              │
                        │  └─ diarize        │              │
                        └──────────────────┘              │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Correction     │◀───▶│  extract_all      │────▶│  Structured Data │
│  Layer          │     │  (LLM)            │     │  (JSON)          │
└─────────────────┘     │  ├─ summary       │     └──────────────────┘
                        │  ├─ todos         │
                        │  ├─ entities      │     ┌──────────────────┐
                        │  ├─ products      │────▶│  Obsidian Vault  │
                        │  └─ risks         │     │  (sync)          │
                        └──────────────────┘     └──────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Extended Pipeline (v0.5)                                           │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Gmail        │  │ Email TODO   │  │ Calendar     │              │
│  │ Classifier   │  │ Extract      │  │ Integration  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                  │                      │
│         ▼                 ▼                  ▼                      │
│  ┌─────────────────────────────────────────────────────┐           │
│  │  Persistent TODO Store (Jaccard dedup)              │           │
│  └──────────────────────────┬──────────────────────────┘           │
│                             │                                       │
│         ┌───────────────────┼───────────────────┐                  │
│         ▼                   ▼                   ▼                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Knowledge    │  │ Signal       │  │ State        │              │
│  │ Graph        │  │ Detector     │  │ Validator    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐                                │
│  │ Minions      │  │ SMS Handler  │                                │
│  │ Job Queue    │  │ (placeholder)│                                │
│  │ (Postgres)   │  │              │                                │
│  └──────────────┘  └──────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for detailed documentation.

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (tested on RTX 3090)
- ffmpeg in PATH
- HuggingFace token with pyannote access (for diarization)
- PostgreSQL 16+ (for Minions job queue — optional)

### Installation

```bash
# Clone and install (editable mode)
git clone https://github.com/brood-arch/korean-call-transcriber.git
cd korean-call-transcriber

# Basic development setup (no GPU deps)
pip install -e ".[dev]"

# With GPU transcription support (requires CUDA)
pip install -e ".[gpu,dev]"

# With Minions job queue (requires PostgreSQL + psycopg2)
pip install -e ".[queue,dev]"

# Everything
pip install -e ".[all,dev]"
```

See `pyproject.toml` `[project.optional-dependencies]` for available extras.

### Usage

#### 1. Transcribe audio files

```bash
# Transcribe all pending files
python -m kct.transcribe.batch_transcribe

# Transcribe a single file
python -m kct.transcribe.batch_transcribe --file path/to/audio.m4a

# Process newest files first
python -m kct.transcribe.batch_transcribe --recent-first --limit 10

# Equivalent installed CLI
kct-transcribe --recent-first --limit 10
```

#### 2. Extract structured data

```bash
# Full extraction run (summary + todos + entities + ...)
python -m kct.extract.extract_all

# Dry run to validate setup
python -m kct.extract.extract_all --dry-run

# Process only today's files
python -m kct.extract.extract_all --today

# Equivalent installed CLI
kct-extract --today
```

#### 3. Analyze pipeline health

```bash
# Check for gaps in the pipeline
python -m kct.queue.gap_analyzer

# Generate detailed report
python -m kct.queue.gap_analyzer --output-json report.json --output-md report.md

# Pipeline health shortcut
kct-health
```

#### 4. Sync to Obsidian

```bash
# Sync new transcripts
python -m kct.sync.sync_obsidian

# Dry run
python -m kct.sync.sync_obsidian --dry-run

# Re-sync all files
python -m kct.sync.sync_obsidian --all

# Equivalent installed CLI
kct-sync-obsidian --all
```

#### 5. Gmail auto-classification

```bash
# Set GMAIL_ADDRESS and GMAIL_APP_PASSWORD in .env first
python -m kct.integrations.gmail_classifier
```

#### 6. Email TODO extraction

```bash
# Manage sender exclusion list
python -m kct.integrations.email_todo_extract exclude "newsletter@example.com"
python -m kct.integrations.email_todo_extract list-exclusions
python -m kct.integrations.email_todo_extract status
```

#### 7. Calendar integration

```bash
# Set GCAL_TOKEN_PATH in .env to your OAuth2 token file
python -m kct.integrations.calendar
```

#### 7.5. Naver Mail archiving

```bash
# Set NAVER_MAIL_ADDRESS and NAVER_MAIL_PASSWORD in .env first
# Enable IMAP in Naver Mail web settings
python -m kct.integrations.naver_mail

# Dry run (don't save state)
python -m kct.integrations.naver_mail --dry-run

# Single folder
python -m kct.integrations.naver_mail --folder INBOX --limit 50
```

#### 8. Persistent TODO management

```bash
# Check status
python -m kct.todo.persistent_store status

# Sync completed TODOs
python -m kct.todo.persistent_store sync
```

#### 9. Knowledge graph

```bash
# Build graph from all state sources
python -m kct.knowledge.graph --build

# Query related nodes
python -m kct.knowledge.graph --query "cp:CompanyName"

# Show statistics
python -m kct.knowledge.graph --stats
```

#### 10. Signal detection

```bash
# Fast-score a text (no LLM needed)
python -m kct.knowledge.signal_detector --score "주문 500개 확인해주세요"

# Full signal detection
python -m kct.knowledge.signal_detector "Meeting with Acme about 500 units"
```

#### 11. Minions job queue (requires PostgreSQL)

```bash
# Set MINIONS_DB_PASS in .env
python -m kct.pipeline.minions_queue submit sync_transcripts '{"cmd": "python -m kct.extract.extract_all"}'
python -m kct.pipeline.minions_queue list
python -m kct.pipeline.minions_queue stats
python -m kct.pipeline.minions_queue work
```

#### 12. State validation

```bash
# Check all state files
python -m kct.pipeline.validate_state

# JSON output
python -m kct.pipeline.validate_state --json

# Quiet mode (only show issues)
python -m kct.pipeline.validate_state --quiet
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | LLM API key (`ZAI_API_KEY` is also supported for backward compatibility) | (required) |
| `LLM_BASE_URL` | OpenAI-compatible API base URL (`ZAI_BASE_URL` is also supported) | `https://api.openai.com/v1` |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `LLM_DISABLE_THINKING` | Disable GLM thinking traces (`auto`, `true`, `false`) | `auto` |
| `KCT_AUDIO_DIR` | Audio source directory (`AUDIO_DIR` is also supported for backward compatibility) | `data/audio` |
| `KCT_TRANSCRIPT_DIR` | Transcript output directory (`TRANSCRIPT_DIR` is also supported for backward compatibility) | `output/transcripts` |
| `WHISPER_MODEL` | faster-whisper model | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |
| `HF_TOKEN_FILE` | HuggingFace token file path | (empty) |
| `MY_NAME` | Speaker name for caller ID | `Me` |
| `GMAIL_ADDRESS` | Gmail address for IMAP login | (empty) |
| `GMAIL_APP_PASSWORD` | Gmail app-specific password | (empty) |
| `GCAL_TOKEN_PATH` | Path to Google Calendar OAuth2 token | `state/gcal_token.json` |
| `EMAIL_TODO_STATE` | Path to email TODO state file | `state/email_todo_state.json` |
| `EMAIL_TODO_EXCLUSIONS` | Path to sender exclusion list | `state/email_todo_exclusions.json` |
| `KCT_STATE_DIR` | Base state directory | `state` |
| `KCT_LOG_DIR` | Log directory | `logs` |
| `KCT_ENABLE_SHELL_JOBS` | Enable trusted local shell command payloads in Minions queue | `0` |
| `MINIONS_DB_HOST` | Minions Postgres host | `localhost` |
| `MINIONS_DB_PORT` | Minions Postgres port | `5432` |
| `MINIONS_DB_NAME` | Minions database name | `minions` |
| `MINIONS_DB_USER` | Minions database user | `minions` |
| `MINIONS_DB_PASS` | Minions database password | (required for queue) |
| `SMS_GATEWAY_URL` | SMS gateway API endpoint | (empty) |
| `SMS_API_KEY` | SMS gateway API key | (empty) |
| `NAVER_MAIL_ADDRESS` | Naver email address | (empty) |
| `NAVER_MAIL_PASSWORD` | Naver mail password or app password | (empty) |
| `NAVER_MAIL_HOST` | Naver IMAP host | `imap.naver.com` |
| `NAVER_MAIL_FOLDERS` | Comma-separated IMAP folders | `INBOX,"Sent Messages"` |
| `NAVER_MAIL_LIMIT` | Max messages per folder per run | `100` |
| `NAVER_MAIL_STATE_DIR` | State directory for processed UIDs | `state/naver_mail` |

See [.env.example](.env.example) for the full list.

Additional operational notes:
- [Security model](docs/security-model.md)
- [Known limitations](docs/known-limitations.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Operations and dashboard status](docs/operations-dashboard.md)

## Project Structure

```
kct/
├── transcribe/          # WhisperX transcription engine
│   ├── batch_transcribe.py   # Main batch transcription script
│   ├── worker.py             # Isolated subprocess worker
│   └── align_worker.py       # Alignment + diarization worker
├── extract/             # LLM-based extraction
│   ├── extract_all.py        # Unified extraction (8 categories)
│   ├── extract_entities.py   # Standalone entity extraction
│   └── extract_schedules.py  # Schedule/appointment extraction
├── correct/             # STT correction layer
│   └── corrections.py        # Exact + alias corrections
├── sync/                # Output synchronization
│   └── sync_obsidian.py      # Transcript → Obsidian sync
├── pipeline/            # Shared utilities
│   ├── paths.py              # Central path configuration
│   ├── utils.py              # Common utilities
│   ├── health_check.py       # Pipeline health checks
│   ├── minions_queue.py      # Postgres-backed durable job queue
│   └── validate_state.py     # State file validation
├── integrations/        # External service integrations
│   ├── gmail_classifier.py   # Gmail inbox auto-classifier
│   ├── email_todo_extract.py # Email → TODO extraction
│   ├── calendar.py           # Google Calendar integration
│   ├── sms_handler.py        # SMS pipeline placeholder
│   └── naver_mail.py          # Naver Mail IMAP archiver
├── todo/                # TODO management
│   └── persistent_store.py   # Persistent store with Jaccard dedup
├── knowledge/           # Knowledge graph & signal detection
│   ├── graph.py              # Entity relationship graph
│   └── signal_detector.py    # 3-band fast scoring + signal detection
└── queue/               # Pipeline health & retry
    ├── gap_analyzer.py       # Pipeline gap analysis
    └── retry_queue.py        # Atomic retry queue
```

## Key Design Decisions

### Process Isolation for DLL Safety

faster-whisper (CTranslate2) and whisperx (pyannote) have conflicting DLL requirements on Windows. The pipeline uses **subprocess isolation**: transcription runs in the main process, alignment and diarization run in a child process.

### BatchedInferencePipeline

Uses `BatchedInferencePipeline` from faster-whisper for ~3x throughput improvement on NVIDIA GPUs, with automatic fallback to sequential mode.

### Long Audio Chunking

Audio files longer than 5 minutes are automatically split into chunks to prevent CTranslate2 hard-kills on Windows. Timestamps are preserved across chunks.

### Speaker Identification Heuristics

For 2-speaker Korean business calls, the pipeline uses multiple signals:
- Speech duration ratio
- Korean honorific detection (습니다, 입니다, 드리, etc.)
- First-speaker greeting analysis

### Jaccard Fuzzy Dedup for TODOs

The persistent TODO store uses bigram Jaccard similarity (threshold ≥ 0.55) to prevent duplicate TODOs from being added across multiple extraction runs. Same-source dedup merges shorter titles into longer, more descriptive ones.

### 3-Band Signal Scoring

The signal detector uses a weighted multi-signal scoring system to pre-filter text before any LLM API call, saving tokens and latency on trivial messages.

### Minions Job Queue

The Postgres-backed job queue provides crash recovery, idempotent submission, fan-out parallel execution with aggregators, and job steering via messages. Shell command payloads are disabled by default; set `KCT_ENABLE_SHELL_JOBS=1` only for trusted local automation.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Changelog

### v1.1.0
- Added Hermes active-runner orchestration for the full transcription pipeline.
- Added WSL/Windows workspace launcher scripts and operations dashboard documentation.
- Added extraction quality verification/re-extraction to the active-run sequence.
- Hardened retry queue locking, stale-lock reclaim, JSONL validation, and corrupt-state reporting.
- Updated active-run failure semantics so upstream failures block downstream stages by default and `partial_success` exits non-zero.

### v0.8.2
- Refactored `extract_entities.py` and `extract_schedules.py` to use centralized LLM client (`kct.extract.client`)
- Added logging to all silent `except` blocks across codebase
- Added `ffmpeg-python` to core dependencies in pyproject.toml
- Added `chromadb` and `obsidian` optional dependency groups
- Created `docs/` with security model, known limitations, and troubleshooting guides
- Cleaned up `requirements.txt` to match pyproject.toml

### v0.3.4
- Centralized LLM configuration and reused the common LLM client across extraction modules
- Added sensitive-output redaction before retry/minions logs store stdout and stderr tails
- Disabled Minions shell command payloads by default unless `KCT_ENABLE_SHELL_JOBS=1`
- Replaced duplicated signal detector implementation with a compatibility shim
- Reused centralized WSL detection and atomic write helpers across more runtime modules
- Preferred `KCT_*` path environment variables while preserving legacy names

### v0.3.3
- Added console entry points: `kct-transcribe`, `kct-extract`, `kct-health`, `kct-sync-obsidian`
- Generalized unified extraction client to prefer `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL`
- Made retry queue command tests robust across native Windows and WSL command shapes
- Switched CI linting to raw `ruff check .`
- Replaced over-broad module status labels with Stable/Beta/Experimental/Placeholder

### v0.3.0
- Added `kct/integrations/` — Gmail classifier, email TODO extraction, calendar integration, SMS handler, Naver Mail archiver
- Added `kct/todo/` — Persistent TODO store with Jaccard fuzzy dedup
- Added `kct/knowledge/` — Knowledge graph builder and 3-band signal detector
- Added `kct/pipeline/minions_queue.py` — Postgres-backed durable job queue
- Added `kct/pipeline/validate_state.py` — State file validation
- All personal data, passwords, and internal paths removed

### v0.2.0
- STT correction layer with hot-reload
- Gap analyzer with cause taxonomy
- Retry queue with exponential backoff

### v0.1.0
- Initial release: WhisperX transcription, LLM extraction, Obsidian sync

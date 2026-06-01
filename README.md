# korean-call-transcriber

Korean phone call transcription pipeline with Whisper + speaker diarization, auto TODO/schedule/entity extraction, and Obsidian sync.

## Features

- **🎙️ WhisperX Transcription** — faster-whisper (CTranslate2) for fast GPU-accelerated Korean speech-to-text
- **👥 Speaker Diarization** — pyannote-based 2-speaker identification with Korean honorific heuristics
- **📝 Unified LLM Extraction** — Single API call extracts: Summary, TODOs, Appointments, Entities, Products, Money, Risks, and Corrections
- **🔧 STT Correction Layer** — Persistent exact replacements and alias normalization with hot-reload
- **📊 Gap Analyzer** — Deterministic pipeline health check with cause taxonomy
- **🔄 Retry Queue** — JSONL-based atomic retry queue with exponential backoff
- **📓 Obsidian Sync** — Automatic transcript → markdown conversion with counterparty indexing

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
```

See [docs/architecture.md](docs/architecture.md) for detailed documentation.

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (tested on RTX 3090)
- ffmpeg in PATH
- HuggingFace token with pyannote access (for diarization)

### Installation

```bash
git clone https://github.com/brood-arch/korean-call-transcriber.git
cd korean-call-transcriber
pip install -r requirements.txt

# Copy and edit environment config
cp .env.example .env
# Edit .env with your API keys and paths
```

### Usage

#### 1. Transcribe audio files

```bash
# Transcribe all pending files
python -m src.transcribe.batch_transcribe

# Transcribe a single file
python -m src.transcribe.batch_transcribe --file path/to/audio.m4a

# Process newest files first
python -m src.transcribe.batch_transcribe --recent-first --limit 10
```

#### 2. Extract structured data

```bash
# Full extraction run (summary + todos + entities + ...)
python -m src.extract.extract_all

# Dry run to validate setup
python -m src.extract.extract_all --dry-run

# Process only today's files
python -m src.extract.extract_all --today
```

#### 3. Analyze pipeline health

```bash
# Check for gaps in the pipeline
python -m src.queue.gap_analyzer

# Generate detailed report
python -m src.queue.gap_analyzer --output-json report.json --output-md report.md
```

#### 4. Sync to Obsidian

```bash
# Sync new transcripts
python -m src.sync.sync_obsidian

# Dry run
python -m src.sync.sync_obsidian --dry-run

# Re-sync all files
python -m src.sync.sync_obsidian --all
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | LLM API key | (required) |
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.example.com/v1` |
| `LLM_MODEL` | Model name | `example-model` |
| `AUDIO_DIR` | Audio source directory | `data/audio` |
| `TRANSCRIPT_DIR` | Transcript output directory | `output/transcripts` |
| `WHISPER_MODEL` | faster-whisper model | `mobiuslabsgmbh/faster-whisper-large-v3-turbo` |
| `HF_TOKEN_FILE` | HuggingFace token file path | (empty) |
| `MY_NAME` | Speaker name for caller ID | `Me` |

See [.env.example](.env.example) for the full list.

## Project Structure

```
src/
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
│   └── utils.py              # Common utilities
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

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

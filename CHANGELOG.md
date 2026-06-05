# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.1] - 2025-06-05

### Changed
- **DEFAULT_LLM_BASE_URL** changed from ZAI to `https://api.openai.com/v1` (vendor-neutral default)
- **Domain correction rules** moved from hardcoded prompt to `domain_corrections.json` (user-customizable)
- **batch_transcribe.py**: removed `log = logging.info` override, unified all logging to `log.info()`
- **except Exception** narrowed to specific exceptions across ~55 sites (OSError, JSONDecodeError, ConnectionError, etc.)
- **requirements-ci.txt** marked deprecated in favor of `pip install -e ".[dev]"`
- **docs/architecture.md** updated with knowledge/todo/queue/pipeline module descriptions

### Added
- Public function docstrings for all 204 public functions (100% coverage)
- Type hints for 47 parameters and returns across extract/pipeline/queue/todo/sync/knowledge modules
- `__all__` explicit exports in `transcription_corrections.py`
- `PRIVACY.md` with data flow diagram and sensitive info handling
- `README.ko.md` Korean README

### Fixed
- **CI ruff errors** (28 total): F821 undefined `batch_idx`, F841 unused variable, F811 redefined `log`, I001 import sorting

## [0.9.0] - 2025-06-05

### Changed
- **print → logger**: 121 `print()` calls replaced with `logging.info/error/warning` across 17 files
- **E501 line length**: 55 violations → 0 (120 char limit)
- **C901 complexity**: 11 violations → 0 (helper extraction + early return pattern)
- **CLI print()** in `main()` blocks redirected to `sys.stderr`

### Added
- Config test suite expanded: 9 → 30 tests
- `_attempt_llm_call` with 4-stage retry (HTTPError, URLError, JSONDecodeError, Exception)
- `fallback_summary` for graceful LLM failure handling

### Fixed
- `worker.py` format string bug
- `extract_all.py` batch checkpoint resume logic
- Subagent-introduced syntax errors in `align_worker.py`, `batch_transcribe.py`

## [0.8.2] - 2025-06-05

### Changed
- Silent-except blocks now log warnings instead of silently swallowing errors
- Dependency cleanup: removed unused imports
- New extraction modules for unified LLM extraction pipeline

## [0.8.0] - 2025-06-01

### Added
- OpenHuman pattern: 3-band fast scoring, token compression, fallback summary
- `pipeline_health_check.py` with new status types (skipped_fast_score, fallback)
- Korean-English correction module with bilingual docstrings

## [0.3.0] - 2025-05-20

### Added
- Extended pipeline: Gmail classifier, email TODO extraction, calendar integration
- Persistent TODO store with Jaccard fuzzy dedup
- Knowledge graph entity extraction
- Minions queue (Postgres-backed durable job queue)

## [0.1.0] - 2025-06-01

### Added
- Initial release: WhisperX transcription, speaker diarization, LLM extraction
- Obsidian sync, retry queue, state validator
- MIT license

[0.9.1]: https://github.com/brood-arch/korean-call-transcriber/releases/tag/v0.9.1
[0.9.0]: https://github.com/brood-arch/korean-call-transcriber/releases/tag/v0.9.0
[0.8.2]: https://github.com/brood-arch/korean-call-transcriber/releases/tag/v0.8.2
[0.8.0]: https://github.com/brood-arch/korean-call-transcriber/releases/tag/v0.8.0
[0.3.0]: https://github.com/brood-arch/korean-call-transcriber/releases/tag/v0.3.0
[0.1.0]: https://github.com/brood-arch/korean-call-transcriber/releases/tag/v0.1.0

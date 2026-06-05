# Troubleshooting

## "CUDA out of memory"
- Use `--device cpu` for transcription
- Reduce `--batch-size` (default: 8)
- Use smaller Whisper model: `--model base`

## "Module not found: kct.pipeline_utils"
- Ensure you run from repo root
- Install: `pip install -e ".[dev]"`

## "pyannote authentication failed"
- Accept conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
- Set `HF_TOKEN` env var

## "LLM API 429 rate limit"
- Increase `--retry-delay` (default: 60s)
- Use different LLM provider via `LLM_BASE_URL`

## "Windows path issues"
- Use forward slashes or raw strings
- Set `KCT_AUDIO_DIR` explicitly

## "Shell jobs disabled"
- Set `KCT_ENABLE_SHELL_JOBS=1` for trusted local automation
- See `docs/security-model.md` for details

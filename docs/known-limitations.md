# Known Limitations

- **GPU Required**: WhisperX + pyannote diarization needs CUDA GPU (tested on RTX 3090)
- **Korean-only**: STT correction rules and LLM extraction prompts are Korean-specific
- **WSL/Windows**: Primary development environment; Linux/macOS should work but less tested
- **LLM Dependency**: Extraction quality depends on the configured LLM. Defaults to ZAI GLM-5.1.
- **No streaming**: Batch processing only; real-time transcription not supported
- **Pyannote token**: Requires HuggingFace accept of pyannote/speaker-diarization-3.1 conditions
- **SMS module**: Placeholder only; requires external SMS gateway integration

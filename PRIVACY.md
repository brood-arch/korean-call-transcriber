# Privacy & Data Handling

## ⚠️ Important: Sensitive Data Warning

This project processes **phone call recordings, email messages, SMS texts, and calendar events** — all of which may contain highly sensitive personal and business information.

### Data Flows

| Data Type | Source | Processing | Destination |
|-----------|--------|-----------|-------------|
| Audio recordings | Local files | Whisper STT (local GPU) | Text transcripts (local) |
| Transcripts | Local files | LLM API extraction | Structured JSON (local) |
| Email | IMAP (Gmail/Naver) | Local classification + LLM | Local state files |
| Calendar | Google Calendar API | Local processing | Local state files |
| SMS | Gateway/API | Local processing | Local state files |

### External Data Transmission

**Audio transcription**: Runs locally on your GPU via faster-whisper. **No audio data leaves your machine.**

**LLM extraction**: Transcript TEXT is sent to the configured LLM API endpoint. This is the only point where conversational data leaves your environment. 
- Configure `LLM_BASE_URL` to use a local/private LLM if data sensitivity is a concern.
- The prompt instructs the LLM not to store data, but API providers may log requests per their own policies.

**Email/Calendar**: Credentials are read from environment variables only. IMAP connections use TLS.

### What Gets Stored Locally

- `output/transcripts/` — Full text transcripts
- `state/` — Extracted TODOs, entities, schedules, processed-file tracking
- `logs/` — Pipeline execution logs (may contain filenames and summaries)

### Recommendations

1. **Never commit** `output/`, `state/`, or `logs/` to version control
2. Use `.env` for credentials (already in `.gitignore`)
3. Review `LLM_BASE_URL` — use a self-hosted or private endpoint for sensitive calls
4. Regularly audit `state/` and `logs/` for accumulated sensitive data
5. Consider encrypting the `state/` directory if on a shared machine

### Disclaimer

This project is provided as-is with no warranty regarding data security. Users are responsible for ensuring compliance with applicable privacy regulations (GDPR, PIPA, etc.) when processing personal communications.

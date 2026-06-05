# Security Model

## Shell Job Execution

Shell jobs in the Minions queue allow arbitrary command execution.
This is **disabled by default** and requires `KCT_ENABLE_SHELL_JOBS=1`.

**Warning**: Only enable in trusted local automation environments.
Never accept shell job payloads from external/untrusted input.

## Environment Variables

All credentials are read from environment variables. No secrets are hardcoded.

- `LLM_API_KEY` — LLM API authentication (preferred over legacy `ZAI_API_KEY`)
- `GMAIL_APP_PASSWORD` — Gmail app-specific password
- `NAVER_MAIL_PASSWORD` — Naver Mail password
- `MINIONS_DB_PASS` — PostgreSQL password

## Logging and Redaction

All log output passes through `redact_sensitive_text()` which masks:
- Phone numbers → `[REDACTED_PHONE]`
- Email addresses → `l***@domain`
- API keys/tokens → `[REDACTED_TOKEN]`
- Windows user paths → `[REDACTED_PATH]`

## Data Isolation

- Transcription state files are stored locally in `state/`
- No data is sent to external services except the configured LLM endpoint
- Telegram notifications require explicit `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`

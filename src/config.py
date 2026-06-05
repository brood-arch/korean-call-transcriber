"""Central runtime configuration for korean-call-transcriber."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

WORKSPACE = Path(os.environ.get("KCT_WORKSPACE", Path.cwd())).resolve()


DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FAILURE = 2
EXIT_CONFIG = 3


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    disable_thinking: str


def get_env(*names: str, default: str = "", deprecated: tuple[str, ...] = ()) -> str:
    """Return the first non-empty environment variable from names.

    Names listed in ``deprecated`` still work, but emit a deprecation warning
    through logging so older deployments can migrate to KCT_/LLM_ names.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            if name in deprecated:
                preferred = next((n for n in names if n not in deprecated), names[0])
                log.warning("Environment variable %s is deprecated; use %s instead", name, preferred)
            return value
    return default


def get_llm_config(api_key: str = "") -> LLMConfig:
    """Resolve OpenAI-compatible LLM settings.

    Public names are preferred:
    - LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

    ZAI_* is retained as a compatibility fallback.
    """
    return LLMConfig(
        api_key=api_key or get_env("LLM_API_KEY", "ZAI_API_KEY", deprecated=("ZAI_API_KEY",)),
        base_url=get_env(
            "LLM_BASE_URL", "ZAI_BASE_URL",
            default=DEFAULT_LLM_BASE_URL,
            deprecated=("ZAI_BASE_URL",),
        ).rstrip("/"),
        model=get_env("LLM_MODEL", default=DEFAULT_LLM_MODEL),
        disable_thinking=get_env("LLM_DISABLE_THINKING", default="auto").lower(),
    )


def path_from_env(*names: str, default: str | Path, deprecated: tuple[str, ...] | None = None) -> Path:
    """Return a Path from the first available environment variable."""
    dep = deprecated or ()
    return Path(get_env(*names, default=str(default), deprecated=dep))


TRANSCRIPT_DIR = path_from_env("KCT_TRANSCRIPT_DIR", "TRANSCRIPT_DIR", default=WORKSPACE / "output" / "transcripts")
AUDIO_DIR = path_from_env("KCT_AUDIO_DIR", "AUDIO_DIR", default=WORKSPACE / "data" / "audio")
OUTPUT_DIR = path_from_env("KCT_OUTPUT_DIR", "OUTPUT_DIR", default=WORKSPACE / "output")
STATE_DIR = path_from_env("KCT_STATE_DIR", "STATE_DIR", default=WORKSPACE / "state")
LOG_DIR = path_from_env("KCT_LOG_DIR", "LOG_DIR", default=WORKSPACE / "logs")
MODELS_DIR = path_from_env("KCT_MODELS_DIR", "MODELS_DIR", default=WORKSPACE / "models")
OBSIDIAN_VAULT = path_from_env("KCT_OBSIDIAN_VAULT", "OBSIDIAN_VAULT", default=OUTPUT_DIR / "obsidian")

LLM_API_KEY = get_llm_config().api_key
LLM_BASE_URL = get_llm_config().base_url
LLM_MODEL = get_llm_config().model

GMAIL_ADDRESS = get_env("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = get_env("GMAIL_APP_PASSWORD")
NAVER_MAIL_ADDRESS = get_env("NAVER_MAIL_ADDRESS")
NAVER_MAIL_PASSWORD = get_env("NAVER_MAIL_PASSWORD")
GCAL_TOKEN_PATH = path_from_env("GCAL_TOKEN_PATH", default=STATE_DIR / "gcal_token.json")
MINIONS_DB_URL = get_env("MINIONS_DB_URL")
KCT_ENABLE_SHELL_JOBS = get_env("KCT_ENABLE_SHELL_JOBS", default="0")

# Transcription
MY_NAME = get_env("MY_NAME", default="Me")
WHISPER_MODEL = get_env("WHISPER_MODEL", default="large-v3-turbo")
WHISPER_COMPUTE_TYPE = get_env("WHISPER_COMPUTE_TYPE", default="float16")
TRANSCRIBE_LOG = path_from_env("TRANSCRIBE_LOG", default=LOG_DIR / "transcribe_whisperx.log")

# Minions DB
MINIONS_DB_HOST = get_env("MINIONS_DB_HOST", default="localhost")
MINIONS_DB_PORT = get_env("MINIONS_DB_PORT", default="5432")
MINIONS_DB_NAME = get_env("MINIONS_DB_NAME", default="minions")
MINIONS_DB_USER = get_env("MINIONS_DB_USER", default="minions")
MINIONS_DB_PASS = get_env("MINIONS_DB_PASS")

# Naver Mail
NAVER_MAIL_HOST = get_env("NAVER_MAIL_HOST", default="imap.naver.com")
NAVER_MAIL_PORT = get_env("NAVER_MAIL_PORT", default="993")
NAVER_MAIL_LIMIT = get_env("NAVER_MAIL_LIMIT", default="100")
NAVER_MAIL_STATE_DIR = path_from_env("NAVER_MAIL_STATE_DIR", default=STATE_DIR / "naver_mail")
NAVER_MAIL_FOLDERS = get_env("NAVER_MAIL_FOLDERS", default="INBOX,Sent Messages")

# Corrections
CORRECTIONS_RULES_PATH = path_from_env("KCT_CORRECTIONS_RULES", default=STATE_DIR / "correction_rules.json")
CORRECTIONS_LOG_PATH = path_from_env("KCT_CORRECTIONS_LOG", default=STATE_DIR / "corrections.jsonl")

# Langfuse
LANGFUSE_SECRET_KEY = get_env("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = get_env("LANGFUSE_PUBLIC_KEY")

# Telegram notifications
TELEGRAM_BOT_TOKEN = get_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_env("TELEGRAM_CHAT_ID")

# Windows/WSL bridge
WINDOWS_PYTHON = get_env("KCT_WINDOWS_PYTHON", "WINDOWS_PYTHON", default="python", deprecated=("WINDOWS_PYTHON",))
WHISPERX_PYTHON = get_env(
    "KCT_WHISPERX_PYTHON", "WHISPERX_PYTHON",
    default=r".\tools\whisperx-venv\Scripts\python.exe",
    deprecated=("WHISPERX_PYTHON",),
)
HF_TOKEN_FILE = get_env("KCT_HF_TOKEN_FILE", "HF_TOKEN_FILE", default="", deprecated=("HF_TOKEN_FILE",))
KCT_ALIGN_WORKER = get_env("KCT_ALIGN_WORKER", default=r"src\transcribe\align_worker.py")
CHROMA_INDEX_DIR = path_from_env(
    "KCT_CHROMA_INDEX_DIR", "CHROMA_INDEX_DIR",
    default=TRANSCRIPT_DIR / "chroma_index",
    deprecated=("CHROMA_INDEX_DIR",),
)

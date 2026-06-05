"""Central runtime configuration for korean-call-transcriber.

Environment variables are resolved lazily on first access so that
``import kct.config`` never triggers side-effects.  Each module-level
name can still be read as a plain value (``config.TRANSCRIPT_DIR``)
but the underlying ``os.environ`` lookup is deferred until the
attribute is actually requested.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (no env involvement – safe at import time)
# ---------------------------------------------------------------------------

WORKSPACE = Path(os.environ.get("KCT_WORKSPACE", Path.cwd())).resolve()

DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FAILURE = 2
EXIT_CONFIG = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    disable_thinking: str


def get_env(*names: str, default: str = "", deprecated: tuple[str, ...] = ()) -> str:
    """Return the first non-empty environment variable from *names*.

    Names listed in *deprecated* still work but emit a deprecation
    warning so older deployments can migrate to ``KCT_/LLM_`` names.
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
    """Resolve OpenAI-compatible LLM settings."""
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


# ---------------------------------------------------------------------------
# Lazy descriptor – deferred env resolution
# ---------------------------------------------------------------------------

class _LazyEnv:
    """Descriptor that resolves an env-driven value on first access."""

    def __init__(self, factory):
        self._factory = factory
        self._attr_name: str | None = None

    def __set_name__(self, owner, name):
        self._attr_name = f"_lazy_{name}"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self  # type: ignore[return-value]
        val = getattr(obj, self._attr_name, _SENTINEL)
        if val is _SENTINEL:
            val = self._factory()
            object.__setattr__(obj, self._attr_name, val)
        return val

    def __set__(self, obj, value):
        object.__setattr__(obj, self._attr_name, value)


_SENTINEL = object()


class _Config:
    """Module-level configuration proxy with lazy env resolution.

    All attributes are resolved from ``os.environ`` on first access.
    Assigning to an attribute caches the new value immediately.
    """

    # Paths
    TRANSCRIPT_DIR: Path = _LazyEnv(
        lambda: path_from_env("KCT_TRANSCRIPT_DIR", "TRANSCRIPT_DIR", default=WORKSPACE / "output" / "transcripts")
    )
    AUDIO_DIR: Path = _LazyEnv(
        lambda: path_from_env("KCT_AUDIO_DIR", "AUDIO_DIR", default=WORKSPACE / "data" / "audio")
    )
    OUTPUT_DIR: Path = _LazyEnv(
        lambda: path_from_env("KCT_OUTPUT_DIR", "OUTPUT_DIR", default=WORKSPACE / "output")
    )
    STATE_DIR: Path = _LazyEnv(
        lambda: path_from_env("KCT_STATE_DIR", "STATE_DIR", default=WORKSPACE / "state")
    )
    LOG_DIR: Path = _LazyEnv(
        lambda: path_from_env("KCT_LOG_DIR", "LOG_DIR", default=WORKSPACE / "logs")
    )
    MODELS_DIR: Path = _LazyEnv(
        lambda: path_from_env("KCT_MODELS_DIR", "MODELS_DIR", default=WORKSPACE / "models")
    )
    OBSIDIAN_VAULT: Path = _LazyEnv(
        lambda: path_from_env("KCT_OBSIDIAN_VAULT", "OBSIDIAN_VAULT", default=_cfg.OUTPUT_DIR / "obsidian")
    )

    # LLM (lazy so get_llm_config() runs only on access)
    LLM_API_KEY: str = _LazyEnv(lambda: get_llm_config().api_key)
    LLM_BASE_URL: str = _LazyEnv(lambda: get_llm_config().base_url)
    LLM_MODEL: str = _LazyEnv(lambda: get_llm_config().model)

    # Integrations
    GMAIL_ADDRESS: str = _LazyEnv(lambda: get_env("GMAIL_ADDRESS"))
    GMAIL_APP_PASSWORD: str = _LazyEnv(lambda: get_env("GMAIL_APP_PASSWORD"))
    NAVER_MAIL_ADDRESS: str = _LazyEnv(lambda: get_env("NAVER_MAIL_ADDRESS"))
    NAVER_MAIL_PASSWORD: str = _LazyEnv(lambda: get_env("NAVER_MAIL_PASSWORD"))
    GCAL_TOKEN_PATH: Path = _LazyEnv(
        lambda: path_from_env("GCAL_TOKEN_PATH", default=_cfg.STATE_DIR / "gcal_token.json")
    )
    MINIONS_DB_URL: str = _LazyEnv(lambda: get_env("MINIONS_DB_URL"))
    KCT_ENABLE_SHELL_JOBS: str = _LazyEnv(lambda: get_env("KCT_ENABLE_SHELL_JOBS", default="0"))

    # Transcription
    MY_NAME: str = _LazyEnv(lambda: get_env("MY_NAME", default="Me"))
    WHISPER_MODEL: str = _LazyEnv(lambda: get_env("WHISPER_MODEL", default="large-v3-turbo"))
    WHISPER_COMPUTE_TYPE: str = _LazyEnv(lambda: get_env("WHISPER_COMPUTE_TYPE", default="float16"))
    TRANSCRIBE_LOG: Path = _LazyEnv(
        lambda: path_from_env("TRANSCRIBE_LOG", default=_cfg.LOG_DIR / "transcribe_whisperx.log")
    )

    # Minions DB
    MINIONS_DB_HOST: str = _LazyEnv(lambda: get_env("MINIONS_DB_HOST", default="localhost"))
    MINIONS_DB_PORT: str = _LazyEnv(lambda: get_env("MINIONS_DB_PORT", default="5432"))
    MINIONS_DB_NAME: str = _LazyEnv(lambda: get_env("MINIONS_DB_NAME", default="minions"))
    MINIONS_DB_USER: str = _LazyEnv(lambda: get_env("MINIONS_DB_USER", default="minions"))
    MINIONS_DB_PASS: str = _LazyEnv(lambda: get_env("MINIONS_DB_PASS"))

    # Naver Mail
    NAVER_MAIL_HOST: str = _LazyEnv(lambda: get_env("NAVER_MAIL_HOST", default="imap.naver.com"))
    NAVER_MAIL_PORT: str = _LazyEnv(lambda: get_env("NAVER_MAIL_PORT", default="993"))
    NAVER_MAIL_LIMIT: str = _LazyEnv(lambda: get_env("NAVER_MAIL_LIMIT", default="100"))
    NAVER_MAIL_STATE_DIR: Path = _LazyEnv(
        lambda: path_from_env("NAVER_MAIL_STATE_DIR", default=_cfg.STATE_DIR / "naver_mail")
    )
    NAVER_MAIL_FOLDERS: str = _LazyEnv(lambda: get_env("NAVER_MAIL_FOLDERS", default="INBOX,Sent Messages"))

    # Corrections
    CORRECTIONS_RULES_PATH: Path = _LazyEnv(
        lambda: path_from_env("KCT_CORRECTIONS_RULES", default=_cfg.STATE_DIR / "correction_rules.json")
    )
    CORRECTIONS_LOG_PATH: Path = _LazyEnv(
        lambda: path_from_env("KCT_CORRECTIONS_LOG", default=_cfg.STATE_DIR / "corrections.jsonl")
    )

    # Langfuse
    LANGFUSE_SECRET_KEY: str = _LazyEnv(lambda: get_env("LANGFUSE_SECRET_KEY"))
    LANGFUSE_PUBLIC_KEY: str = _LazyEnv(lambda: get_env("LANGFUSE_PUBLIC_KEY"))

    # Telegram
    TELEGRAM_BOT_TOKEN: str = _LazyEnv(lambda: get_env("TELEGRAM_BOT_TOKEN"))
    TELEGRAM_CHAT_ID: str = _LazyEnv(lambda: get_env("TELEGRAM_CHAT_ID"))

    # Windows/WSL bridge
    WINDOWS_PYTHON: str = _LazyEnv(
        lambda: get_env("KCT_WINDOWS_PYTHON", "WINDOWS_PYTHON", default="python", deprecated=("WINDOWS_PYTHON",))
    )
    WHISPERX_PYTHON: str = _LazyEnv(
        lambda: get_env(
            "KCT_WHISPERX_PYTHON", "WHISPERX_PYTHON",
            default=r".\tools\whisperx-venv\Scripts\python.exe",
            deprecated=("WHISPERX_PYTHON",),
        )
    )
    HF_TOKEN_FILE: str = _LazyEnv(
        lambda: get_env("KCT_HF_TOKEN_FILE", "HF_TOKEN_FILE", default="", deprecated=("HF_TOKEN_FILE",))
    )
    KCT_ALIGN_WORKER: str = _LazyEnv(lambda: get_env("KCT_ALIGN_WORKER", default=r"kct\transcribe\align_worker.py"))
    CHROMA_INDEX_DIR: Path = _LazyEnv(
        lambda: path_from_env(
            "KCT_CHROMA_INDEX_DIR", "CHROMA_INDEX_DIR",
            default=_cfg.TRANSCRIPT_DIR / "chroma_index",
            deprecated=("CHROMA_INDEX_DIR",),
        )
    )


_cfg = _Config()

# Public module-level names — lazy proxies
TRANSCRIPT_DIR = _cfg.TRANSCRIPT_DIR  # type: ignore[assignment]
AUDIO_DIR = _cfg.AUDIO_DIR  # type: ignore[assignment]
OUTPUT_DIR = _cfg.OUTPUT_DIR  # type: ignore[assignment]
STATE_DIR = _cfg.STATE_DIR  # type: ignore[assignment]
LOG_DIR = _cfg.LOG_DIR  # type: ignore[assignment]
MODELS_DIR = _cfg.MODELS_DIR  # type: ignore[assignment]
OBSIDIAN_VAULT = _cfg.OBSIDIAN_VAULT  # type: ignore[assignment]

LLM_API_KEY = _cfg.LLM_API_KEY  # type: ignore[assignment]
LLM_BASE_URL = _cfg.LLM_BASE_URL  # type: ignore[assignment]
LLM_MODEL = _cfg.LLM_MODEL  # type: ignore[assignment]

GMAIL_ADDRESS = _cfg.GMAIL_ADDRESS  # type: ignore[assignment]
GMAIL_APP_PASSWORD = _cfg.GMAIL_APP_PASSWORD  # type: ignore[assignment]
NAVER_MAIL_ADDRESS = _cfg.NAVER_MAIL_ADDRESS  # type: ignore[assignment]
NAVER_MAIL_PASSWORD = _cfg.NAVER_MAIL_PASSWORD  # type: ignore[assignment]
GCAL_TOKEN_PATH = _cfg.GCAL_TOKEN_PATH  # type: ignore[assignment]
MINIONS_DB_URL = _cfg.MINIONS_DB_URL  # type: ignore[assignment]
KCT_ENABLE_SHELL_JOBS = _cfg.KCT_ENABLE_SHELL_JOBS  # type: ignore[assignment]

MY_NAME = _cfg.MY_NAME  # type: ignore[assignment]
WHISPER_MODEL = _cfg.WHISPER_MODEL  # type: ignore[assignment]
WHISPER_COMPUTE_TYPE = _cfg.WHISPER_COMPUTE_TYPE  # type: ignore[assignment]
TRANSCRIBE_LOG = _cfg.TRANSCRIBE_LOG  # type: ignore[assignment]

MINIONS_DB_HOST = _cfg.MINIONS_DB_HOST  # type: ignore[assignment]
MINIONS_DB_PORT = _cfg.MINIONS_DB_PORT  # type: ignore[assignment]
MINIONS_DB_NAME = _cfg.MINIONS_DB_NAME  # type: ignore[assignment]
MINIONS_DB_USER = _cfg.MINIONS_DB_USER  # type: ignore[assignment]
MINIONS_DB_PASS = _cfg.MINIONS_DB_PASS  # type: ignore[assignment]

NAVER_MAIL_HOST = _cfg.NAVER_MAIL_HOST  # type: ignore[assignment]
NAVER_MAIL_PORT = _cfg.NAVER_MAIL_PORT  # type: ignore[assignment]
NAVER_MAIL_LIMIT = _cfg.NAVER_MAIL_LIMIT  # type: ignore[assignment]
NAVER_MAIL_STATE_DIR = _cfg.NAVER_MAIL_STATE_DIR  # type: ignore[assignment]
NAVER_MAIL_FOLDERS = _cfg.NAVER_MAIL_FOLDERS  # type: ignore[assignment]

CORRECTIONS_RULES_PATH = _cfg.CORRECTIONS_RULES_PATH  # type: ignore[assignment]
CORRECTIONS_LOG_PATH = _cfg.CORRECTIONS_LOG_PATH  # type: ignore[assignment]

LANGFUSE_SECRET_KEY = _cfg.LANGFUSE_SECRET_KEY  # type: ignore[assignment]
LANGFUSE_PUBLIC_KEY = _cfg.LANGFUSE_PUBLIC_KEY  # type: ignore[assignment]

TELEGRAM_BOT_TOKEN = _cfg.TELEGRAM_BOT_TOKEN  # type: ignore[assignment]
TELEGRAM_CHAT_ID = _cfg.TELEGRAM_CHAT_ID  # type: ignore[assignment]

WINDOWS_PYTHON = _cfg.WINDOWS_PYTHON  # type: ignore[assignment]
WHISPERX_PYTHON = _cfg.WHISPERX_PYTHON  # type: ignore[assignment]
HF_TOKEN_FILE = _cfg.HF_TOKEN_FILE  # type: ignore[assignment]
KCT_ALIGN_WORKER = _cfg.KCT_ALIGN_WORKER  # type: ignore[assignment]
CHROMA_INDEX_DIR = _cfg.CHROMA_INDEX_DIR  # type: ignore[assignment]

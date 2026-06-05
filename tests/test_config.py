"""Tests for kct.config — env resolution, defaults, deprecation warnings."""

import pytest


class TestWorkspaceResolution:
    def test_default_workspace_is_cwd(self, monkeypatch):
        monkeypatch.delenv("KCT_WORKSPACE", raising=False)
        # Re-import to get fresh resolution
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.WORKSPACE.is_absolute()

    def test_env_workspace_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.WORKSPACE == tmp_path.resolve()


class TestGetEnv:
    def test_returns_first_non_empty(self, monkeypatch):
        monkeypatch.delenv("KCT_TEST_A", raising=False)
        monkeypatch.setenv("KCT_TEST_B", "hello")
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        result = cfg.get_env("KCT_TEST_A", "KCT_TEST_B", default="")
        assert result == "hello"

    def test_returns_default_when_none_set(self, monkeypatch):
        monkeypatch.delenv("KCT_NONEXISTENT_XYZ", raising=False)
        monkeypatch.delenv("KCT_NONEXISTENT_XYZ2", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        result = cfg.get_env("KCT_NONEXISTENT_XYZ", "KCT_NONEXISTENT_XYZ2", default="fallback")
        assert result == "fallback"


class TestLLMConfig:
    def test_missing_api_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.api_key == ""

    def test_llm_api_key_preferred_over_zai(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "llm-key")
        monkeypatch.setenv("ZAI_API_KEY", "zai-key")
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.api_key == "llm-key"

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.model == "gpt-4o-mini"


class TestPathConstants:
    def test_transcript_dir_is_absolute(self, monkeypatch):
        monkeypatch.delenv("KCT_WORKSPACE", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.TRANSCRIPT_DIR.is_absolute()

    def test_state_dir_under_workspace(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.STATE_DIR == tmp_path.resolve() / "state"


class TestPathFromEnv:
    def test_path_from_env_returns_default(self, monkeypatch):
        monkeypatch.delenv("KCT_AUDIO_DIR", raising=False)
        monkeypatch.delenv("AUDIO_DIR", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.AUDIO_DIR.is_absolute()

    def test_path_from_env_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_AUDIO_DIR", str(tmp_path / "custom_audio"))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.AUDIO_DIR == (tmp_path / "custom_audio").resolve()


class TestConstants:
    def test_exit_codes_are_int(self):
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.EXIT_OK == 0
        assert cfg.EXIT_PARTIAL == 1
        assert cfg.EXIT_FAILURE == 2
        assert cfg.EXIT_CONFIG == 3

    def test_default_constants(self, monkeypatch):
        monkeypatch.delenv("MY_NAME", raising=False)
        monkeypatch.delenv("WHISPER_MODEL", raising=False)
        monkeypatch.delenv("KCT_ENABLE_SHELL_JOBS", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.MY_NAME == "Me"
        assert cfg.WHISPER_MODEL == "large-v3-turbo"
        assert cfg.KCT_ENABLE_SHELL_JOBS == "0"

    def test_deprecated_env_emits_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("ZAI_API_KEY", "test-deprecated-key")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        import logging
        with caplog.at_level(logging.WARNING, logger="kct.config"):
            config = cfg.get_llm_config()
        assert config.api_key == "test-deprecated-key"
        assert any("deprecated" in r.message for r in caplog.records)


class TestLLMConfigExplicitKey:
    def test_explicit_api_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config(api_key="explicit-key")
        assert config.api_key == "explicit-key"

    def test_base_url_default(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("ZAI_BASE_URL", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.base_url == "https://api.openai.com/v1"

    def test_base_url_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://custom.api/v1/")
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.base_url == "https://custom.api/v1"

    def test_disable_thinking_default(self, monkeypatch):
        monkeypatch.delenv("LLM_DISABLE_THINKING", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.disable_thinking == "auto"

    def test_disable_thinking_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_DISABLE_THINKING", "TRUE")
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.disable_thinking == "true"

    def test_llmconfig_is_frozen(self, monkeypatch):
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        import dataclasses
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.api_key = "mutated"


class TestPathFromEnvDeprecated:
    def test_path_from_env_with_deprecated(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CHROMA_INDEX_DIR", str(tmp_path / "old_chroma"))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        # Should resolve the deprecated CHROMA_INDEX_DIR
        assert cfg.CHROMA_INDEX_DIR == (tmp_path / "old_chroma").resolve()


class TestMorePathConstants:
    def test_output_dir_under_workspace(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.OUTPUT_DIR == tmp_path.resolve() / "output"

    def test_log_dir_under_workspace(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.LOG_DIR == tmp_path.resolve() / "logs"

    def test_models_dir_under_workspace(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.MODELS_DIR == tmp_path.resolve() / "models"

    def test_obsidian_vault_under_output(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.OBSIDIAN_VAULT == tmp_path.resolve() / "output" / "obsidian"


class TestTranscriptionConstants:
    def test_whisper_compute_type_default(self, monkeypatch):
        monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.WHISPER_COMPUTE_TYPE == "float16"

    def test_transcribe_log_under_log_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.TRANSCRIBE_LOG == tmp_path.resolve() / "logs" / "transcribe_whisperx.log"


class TestMinionsConstants:
    def test_minions_db_defaults(self, monkeypatch):
        monkeypatch.delenv("MINIONS_DB_HOST", raising=False)
        monkeypatch.delenv("MINIONS_DB_PORT", raising=False)
        monkeypatch.delenv("MINIONS_DB_NAME", raising=False)
        monkeypatch.delenv("MINIONS_DB_USER", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.MINIONS_DB_HOST == "localhost"
        assert cfg.MINIONS_DB_PORT == "5432"
        assert cfg.MINIONS_DB_NAME == "minions"
        assert cfg.MINIONS_DB_USER == "minions"


class TestNaverMailConstants:
    def test_naver_mail_defaults(self, monkeypatch):
        monkeypatch.delenv("NAVER_MAIL_HOST", raising=False)
        monkeypatch.delenv("NAVER_MAIL_PORT", raising=False)
        monkeypatch.delenv("NAVER_MAIL_LIMIT", raising=False)
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.NAVER_MAIL_HOST == "imap.naver.com"
        assert cfg.NAVER_MAIL_PORT == "993"
        assert cfg.NAVER_MAIL_LIMIT == "100"


class TestCorrectionsConstants:
    def test_corrections_paths_under_state(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import kct.config as cfg
        importlib.reload(cfg)
        assert cfg.CORRECTIONS_RULES_PATH == tmp_path.resolve() / "state" / "correction_rules.json"
        assert cfg.CORRECTIONS_LOG_PATH == tmp_path.resolve() / "state" / "corrections.jsonl"

"""Tests for src.config — env resolution, defaults, deprecation warnings."""




class TestWorkspaceResolution:
    def test_default_workspace_is_cwd(self, monkeypatch):
        monkeypatch.delenv("KCT_WORKSPACE", raising=False)
        # Re-import to get fresh resolution
        import importlib

        import src.config as cfg
        importlib.reload(cfg)
        assert cfg.WORKSPACE.is_absolute()

    def test_env_workspace_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import src.config as cfg
        importlib.reload(cfg)
        assert cfg.WORKSPACE == tmp_path.resolve()


class TestGetEnv:
    def test_returns_first_non_empty(self, monkeypatch):
        monkeypatch.delenv("KCT_TEST_A", raising=False)
        monkeypatch.setenv("KCT_TEST_B", "hello")
        import importlib

        import src.config as cfg
        importlib.reload(cfg)
        result = cfg.get_env("KCT_TEST_A", "KCT_TEST_B", default="")
        assert result == "hello"

    def test_returns_default_when_none_set(self, monkeypatch):
        monkeypatch.delenv("KCT_NONEXISTENT_XYZ", raising=False)
        monkeypatch.delenv("KCT_NONEXISTENT_XYZ2", raising=False)
        import importlib

        import src.config as cfg
        importlib.reload(cfg)
        result = cfg.get_env("KCT_NONEXISTENT_XYZ", "KCT_NONEXISTENT_XYZ2", default="fallback")
        assert result == "fallback"


class TestLLMConfig:
    def test_missing_api_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        import importlib

        import src.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.api_key == ""

    def test_llm_api_key_preferred_over_zai(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "llm-key")
        monkeypatch.setenv("ZAI_API_KEY", "zai-key")
        import importlib

        import src.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.api_key == "llm-key"

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("LLM_MODEL", raising=False)
        import importlib

        import src.config as cfg
        importlib.reload(cfg)
        config = cfg.get_llm_config()
        assert config.model == "glm-5.1"


class TestPathConstants:
    def test_transcript_dir_is_absolute(self, monkeypatch):
        monkeypatch.delenv("KCT_WORKSPACE", raising=False)
        import importlib

        import src.config as cfg
        importlib.reload(cfg)
        assert cfg.TRANSCRIPT_DIR.is_absolute()

    def test_state_dir_under_workspace(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KCT_WORKSPACE", str(tmp_path))
        import importlib

        import src.config as cfg
        importlib.reload(cfg)
        assert cfg.STATE_DIR == tmp_path.resolve() / "state"

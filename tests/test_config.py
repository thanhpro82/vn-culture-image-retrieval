from retrieval.config import Settings, get_settings


def _fresh(monkeypatch, **env):
    for k in ("API_KEY", "GEMINI_TIMEOUT_MS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    return get_settings()


def test_settings_reads_env(monkeypatch):
    s = _fresh(monkeypatch, API_KEY="abc", GEMINI_TIMEOUT_MS="900")
    assert s.api_key == "abc"
    assert s.gemini_timeout_ms == 900


def test_timeout_default(monkeypatch):
    s = Settings(_env_file=None)
    assert s.gemini_timeout_ms == 1200

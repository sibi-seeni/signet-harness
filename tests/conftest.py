import os
from pathlib import Path

import pytest

from signet_harness import settings

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def repo_settings(tmp_path, monkeypatch):
    """Every test sees the repo's signet.toml, no stray SIGNET_*/TTS_* variables,
    and a throwaway cache."""
    for name in list(os.environ):
        if name.startswith(("SIGNET_", "TTS_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("SIGNET_CONFIG", str(REPO / "signet.toml"))
    settings.reset()
    s = settings.configure(settings.load(), cache_dir=tmp_path / "cache")
    yield s
    settings.reset()

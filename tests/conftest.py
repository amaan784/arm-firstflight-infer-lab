"""Test isolation: keep the suite hermetic regardless of the developer's machine state.

`firstflight setup-engine` drops real llama.cpp binaries into ./engine, which `find_binary`
picks up automatically — great for users, but the "skips cleanly without a binary" tests must
not accidentally find it. Point the engine dir at a nonexistent path for every test; tests
that WANT a binary set LLAMA_CPP_BIN / env_value explicitly.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_engine_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("FIRSTFLIGHT_ENGINE_DIR", str(tmp_path / "no-engine"))

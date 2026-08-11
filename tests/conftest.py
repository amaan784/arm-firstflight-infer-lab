"""Keep the suite hermetic regardless of the developer's machine state.

`firstflight setup-engine` drops real llama.cpp binaries into ./engine and `find_binary` picks
them up, which breaks the "skips without a binary" tests. Point the engine dir at a nonexistent
path for every test; tests that WANT a binary set LLAMA_CPP_BIN / env_value themselves.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_engine_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("FIRSTFLIGHT_ENGINE_DIR", str(tmp_path / "no-engine"))

from pathlib import Path

from firstflight.engines import llama_cpp


def test_find_binary_none(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    assert llama_cpp.find_binary("cli", env_value="") is None


def test_find_binary_in_dir(tmp_path):
    (tmp_path / "llama-cli").write_text("#!/bin/sh\n")
    found = llama_cpp.find_binary("cli", env_value=str(tmp_path))
    assert found is not None
    assert found.name == "llama-cli"


def test_find_binary_recursive(tmp_path):
    nested = tmp_path / "build" / "bin"
    nested.mkdir(parents=True)
    (nested / "llama-bench").write_text("#!/bin/sh\n")
    found = llama_cpp.find_binary("bench", env_value=str(tmp_path))
    assert found is not None
    assert found.name == "llama-bench"


def test_build_run_cmd():
    cmd = llama_cpp.build_run_cmd(
        Path("llama-cli"), Path("m.gguf"), "hello world", n_predict=8, threads=4, seed=1
    )
    assert "-m" in cmd and "-p" in cmd and "-n" in cmd
    assert "hello world" in cmd
    assert "8" in cmd  # n_predict

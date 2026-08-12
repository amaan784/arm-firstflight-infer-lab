import subprocess
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


def test_find_binary_prefers_completion_over_cli(tmp_path):
    """A modern llama-cli is a chat REPL that never exits at EOF: it must never be picked
    over llama-completion just because both are in the release archive."""
    (tmp_path / "llama-cli").write_text("#!/bin/sh\n")
    (tmp_path / "llama-completion").write_text("#!/bin/sh\n")
    found = llama_cpp.find_binary("cli", env_value=str(tmp_path))
    assert found is not None
    assert found.name == "llama-completion"


def test_build_run_cmd():
    cmd = llama_cpp.build_run_cmd(
        Path("llama-completion"), Path("m.gguf"), "hello world", n_predict=8, threads=4, seed=1
    )
    assert "-m" in cmd and "-p" in cmd and "-n" in cmd
    assert "hello world" in cmd
    assert "8" in cmd  # n_predict
    assert "-v" not in cmd  # verbose is opt-in


def test_build_run_cmd_verbose():
    cmd = llama_cpp.build_run_cmd(Path("llama-completion"), Path("m.gguf"), "hi", verbose=True)
    assert "-v" in cmd


def test_run_once_timeout_keeps_engine_output(monkeypatch, tmp_path):
    """A timeout must carry what the engine printed first, or a rejected flag looks like a
    slow machine (the whole diagnosis for arm-bench runs 1-14)."""

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            cmd=["llama-completion"],
            timeout=180,
            output=b"--no-conversation is not supported by llama-cli\n" + b"> " * 5000,
            stderr=b"load: some warning",
        )

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    res = llama_cpp.run_once(tmp_path / "llama-completion", Path("m.gguf"), timeout=180)

    assert not res.ok
    assert "timed out after 180s" in res.stderr
    assert "not supported by llama-cli" in res.stderr  # the head, where the reason lives
    assert "load: some warning" in res.stderr
    assert len(res.stderr) < 1200  # bounded: the runaway "> " spew is not replayed


def test_run_once_timeout_without_output(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["llama-completion"], timeout=5)

    monkeypatch.setattr(llama_cpp.subprocess, "run", fake_run)
    res = llama_cpp.run_once(tmp_path / "llama-completion", Path("m.gguf"), timeout=5)
    assert res.stderr.strip() == "llama-completion timed out after 5s"

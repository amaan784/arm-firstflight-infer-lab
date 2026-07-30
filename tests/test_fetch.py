import pytest

from firstflight.engines import fetch, llama_cpp


def test_asset_for_platforms():
    tag = "b9999"
    assert fetch.asset_for("windows", "amd64", tag) == "llama-b9999-bin-win-cpu-x64.zip"
    assert fetch.asset_for("windows", "x86_64", tag) == "llama-b9999-bin-win-cpu-x64.zip"
    assert fetch.asset_for("linux", "aarch64", tag) == "llama-b9999-bin-ubuntu-arm64.tar.gz"
    assert fetch.asset_for("linux", "arm64", tag) == "llama-b9999-bin-ubuntu-arm64.tar.gz"
    assert fetch.asset_for("darwin", "arm64", tag) == "llama-b9999-bin-macos-arm64.tar.gz"


def test_asset_for_unsupported():
    with pytest.raises(fetch.EngineFetchError):
        fetch.asset_for("plan9", "mips", "b1")


def test_download_url():
    url = fetch.download_url("b9999", "linux", "aarch64")
    assert url.startswith("https://github.com/ggml-org/llama.cpp/releases/download/b9999/")
    assert url.endswith("llama-b9999-bin-ubuntu-arm64.tar.gz")


def test_find_binary_prefers_engine_dir(monkeypatch, tmp_path):
    # env unset -> local engine dir found before PATH
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    eng = tmp_path / "engine"
    (eng / "build" / "bin").mkdir(parents=True)
    (eng / "build" / "bin" / "llama-cli").write_text("#!/bin/sh\n")
    monkeypatch.setenv("FIRSTFLIGHT_ENGINE_DIR", str(eng))
    found = llama_cpp.find_binary("cli")
    assert found is not None and found.name == "llama-cli"


def test_explicit_env_override_wins(monkeypatch, tmp_path):
    # LLAMA_CPP_BIN dir beats the engine dir
    override = tmp_path / "override"
    override.mkdir()
    (override / "llama-cli").write_text("#!/bin/sh\n")
    eng = tmp_path / "engine"
    eng.mkdir()
    (eng / "llama-cli").write_text("#!/bin/sh\n")
    monkeypatch.setenv("FIRSTFLIGHT_ENGINE_DIR", str(eng))
    monkeypatch.setenv("LLAMA_CPP_BIN", str(override))
    found = llama_cpp.find_binary("cli")
    assert found is not None and str(override) in str(found)


def test_detect_kleidiai(monkeypatch, tmp_path):
    from pathlib import Path

    def fake_run(stderr_text, ok=True):
        def fn(cli_bin, model_path, *, prompt, n_predict, threads=None, seed=42, timeout=300.0):
            return llama_cpp.RunResult(
                returncode=0 if ok else 1, stdout="hello", stderr=stderr_text, wall_s=0.1, cmd=[]
            )

        return fn

    monkeypatch.setattr(
        llama_cpp, "run_once", fake_run("load_tensors: CPU_KLEIDIAI model buffer size = 100 MiB")
    )
    assert llama_cpp.detect_kleidiai(Path("x"), Path("m")) is True

    monkeypatch.setattr(llama_cpp, "run_once", fake_run("load_tensors: CPU model buffer size"))
    assert llama_cpp.detect_kleidiai(Path("x"), Path("m")) is False

    monkeypatch.setattr(llama_cpp, "run_once", fake_run("boom", ok=False))
    assert llama_cpp.detect_kleidiai(Path("x"), Path("m")) is None

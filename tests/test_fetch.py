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


# --- model download: resume + retry -------------------------------------------


class _FakeResp:
    """Minimal requests.Response stand-in for the streaming download path."""

    def __init__(self, body: bytes, status: int = 200, total: int | None = None):
        self._body = body
        self.status_code = status
        self.headers = {"content-length": str(total if total is not None else len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests as _rq

            raise _rq.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size=None):
        yield self._body


def test_download_resumes_from_partial(tmp_path, monkeypatch):
    """A retry must continue from the .part file, not restart the transfer."""
    from firstflight import download as dl

    tmp = tmp_path / "m.gguf.part"
    tmp.write_bytes(b"AAAA")  # 4 bytes already fetched
    seen = {}

    def fake_get(url, **kw):
        seen["range"] = kw.get("headers", {}).get("Range")
        return _FakeResp(b"BBBB", status=206, total=4)

    monkeypatch.setattr(dl.requests, "get", fake_get)
    dl._fetch_with_resume("http://x/m.gguf", tmp, "m.gguf")
    assert seen["range"] == "bytes=4-"
    assert tmp.read_bytes() == b"AAAABBBB"  # appended, not overwritten


def test_download_retries_then_succeeds(tmp_path, monkeypatch):
    import requests as _rq

    from firstflight import download as dl

    calls = {"n": 0}

    def flaky_get(url, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _rq.ConnectionError("throttled")
        return _FakeResp(b"DATA")

    monkeypatch.setattr(dl.requests, "get", flaky_get)
    monkeypatch.setattr(dl.time, "sleep", lambda *_: None)
    out = tmp_path / "m.gguf.part"
    dl._fetch_with_resume("http://x/m.gguf", out, "m.gguf")
    assert calls["n"] == 3
    assert out.read_bytes() == b"DATA"


def test_download_gives_up_with_clear_error(tmp_path, monkeypatch):
    import requests as _rq

    from firstflight import download as dl

    def always_fail(url, **kw):
        raise _rq.ConnectionError("nope")

    monkeypatch.setattr(dl.requests, "get", always_fail)
    monkeypatch.setattr(dl.time, "sleep", lambda *_: None)
    import pytest

    with pytest.raises(dl.DownloadError, match="giving up"):
        dl._fetch_with_resume("http://x/m.gguf", tmp_path / "m.gguf.part", "m.gguf")


# --- cached-model validation --------------------------------------------------


def test_check_gguf_accepts_valid_file(tmp_path):
    from firstflight.download import check_gguf

    p = tmp_path / "ok.gguf"
    p.write_bytes(b"GGUF" + b"\0" * 4096)
    assert check_gguf(p) == ""
    assert check_gguf(p, min_bytes=1000) == ""


def test_check_gguf_rejects_truncated_and_wrong_magic(tmp_path):
    """A cache hit must not be trusted on existence alone.

    A truncated GGUF passes exists(), then hangs or crashes the engine much later where the
    cause is unrecognizable. Size and magic are checked instead.
    """
    from firstflight.download import check_gguf

    truncated = tmp_path / "short.gguf"
    truncated.write_bytes(b"GGUF" + b"\0" * 4096)
    assert "truncated" in check_gguf(truncated, min_bytes=10_000_000)

    html = tmp_path / "err.gguf"
    html.write_bytes(b"<!DOCTYPE html>" + b" " * 4096)
    assert "not a GGUF" in check_gguf(html)

    tiny = tmp_path / "tiny.gguf"
    tiny.write_bytes(b"GG")
    assert "too small" in check_gguf(tiny)

    assert "unreadable" in check_gguf(tmp_path / "missing.gguf")


def test_ensure_model_redownloads_a_corrupt_cache(tmp_path, monkeypatch):
    from firstflight import download as dl
    from firstflight.config import ModelSpec, ModelVariant

    target = tmp_path / "m.gguf"
    target.write_bytes(b"GGUF" + b"\0" * 5000)  # present but far below min_bytes
    variant = ModelVariant(name="q4_0", file="m.gguf", url="http://x/m.gguf", min_bytes=10_000_000)
    spec = ModelSpec(id="m", hf_repo="r", description="d", variants={"q4_0": variant})

    def fake_fetch(url, tmp, label):
        tmp.write_bytes(b"GGUF" + b"\0" * 20_000_000)  # a good file this time

    monkeypatch.setattr(dl, "_fetch_with_resume", fake_fetch)
    out = dl.ensure_model(spec, variant, dest_dir=tmp_path)
    assert out.stat().st_size > 10_000_000  # the bad cache was replaced

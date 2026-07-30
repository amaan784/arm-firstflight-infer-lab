from click.testing import CliRunner

from firstflight.cli import main


def test_help():
    res = CliRunner().invoke(main, ["--help"])
    assert res.exit_code == 0
    assert "firstflight" in res.output.lower()


def test_info_runs():
    res = CliRunner().invoke(main, ["info"])
    assert res.exit_code == 0
    assert "version" in res.output.lower()


def test_smoke_skips_without_binary(monkeypatch):
    # No binary on PATH and no LLAMA_CPP_BIN -> clean skip (exit 0), no download attempted.
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    res = CliRunner().invoke(main, ["smoke"])
    assert res.exit_code == 0
    assert "skip" in res.output.lower()

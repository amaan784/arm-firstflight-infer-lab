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


def test_bench_dry_run_prints_command():
    res = CliRunner().invoke(main, ["bench", "--dry-run"])
    assert res.exit_code == 0
    assert "llama-bench" in res.output
    assert "-p" in res.output


def test_bench_skips_without_binary(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    res = CliRunner().invoke(main, ["bench", "--no-download"])
    assert res.exit_code == 0
    assert "skip" in res.output.lower()


def test_ttft_skips_without_server_binary(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    res = CliRunner().invoke(main, ["ttft", "--no-download"])
    assert res.exit_code == 0
    assert "skip" in res.output.lower()
    assert "llama-server" in res.output


def test_throughput_skips_without_binary(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    res = CliRunner().invoke(main, ["throughput", "--no-download"])
    assert res.exit_code == 0
    assert "skip" in res.output.lower()
    assert "llama-batched-bench" in res.output


def test_autotune_is_opt_in():
    res = CliRunner().invoke(main, ["autotune"])
    assert res.exit_code == 0
    assert "opt-in" in res.output.lower()


def test_autotune_enable_skips_without_binary(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    res = CliRunner().invoke(main, ["autotune", "--enable", "--no-download", "--no-profile"])
    assert res.exit_code == 0
    assert "skip" in res.output.lower()


def test_experiment_skips_without_binary(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    res = CliRunner().invoke(main, ["experiment", "--no-download", "--no-report"])
    assert res.exit_code == 0
    assert "skip" in res.output.lower() or "no configs" in res.output.lower()


def test_profile_noop_off_arm():
    # On a non-Arm-Linux machine, profile no-ops cleanly (exit 0) with a clear reason.
    res = CliRunner().invoke(main, ["profile"])
    assert res.exit_code == 0
    assert "unavailable" in res.output.lower()


def test_report_no_results_is_graceful(tmp_path):
    # empty results dir -> friendly message, exit 0 (no matplotlib needed for this path)
    res = CliRunner().invoke(main, ["report", "--results-dir", str(tmp_path)])
    assert res.exit_code == 0
    assert "no results" in res.output.lower()


def test_smoke_skips_without_binary(monkeypatch):
    # No binary on PATH and no LLAMA_CPP_BIN -> clean skip (exit 0), no download attempted.
    monkeypatch.delenv("LLAMA_CPP_BIN", raising=False)
    monkeypatch.setenv("PATH", "")
    res = CliRunner().invoke(main, ["smoke"])
    assert res.exit_code == 0
    assert "skip" in res.output.lower()

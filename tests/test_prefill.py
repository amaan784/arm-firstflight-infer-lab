import math
from pathlib import Path

import pytest

from firstflight.bench import prefill


def test_build_llama_bench_command():
    cmd = prefill.build_llama_bench_command(
        Path("llama-bench"), Path("m.gguf"), [128, 512], [32], threads=8, repetitions=3
    )
    assert "-p" in cmd and "128,512" in cmd
    assert "-n" in cmd and "32" in cmd
    assert cmd[cmd.index("-o") + 1] == "json"
    assert "-t" in cmd and "8" in cmd
    assert "-r" in cmd and "3" in cmd


def test_build_requires_prompts():
    with pytest.raises(ValueError):
        prefill.build_llama_bench_command(Path("b"), Path("m"), [], [32])


def test_build_cmd_cpu_mask():
    cmd = prefill.build_llama_bench_command(
        Path("b"), Path("m"), [128], [0], cpu_mask="0x0f", cpu_strict=True
    )
    assert "-C" in cmd and "0x0f" in cmd
    assert "--cpu-strict" in cmd


def test_build_cmd_kv_cache_types():
    # flags verified against the real b9873 llama-bench --help (2026-07-07)
    cmd = prefill.build_llama_bench_command(
        Path("b"), Path("m"), [128], [0], cache_type_k="q8_0", cache_type_v="q8_0"
    )
    assert cmd[cmd.index("-ctk") + 1] == "q8_0"
    assert cmd[cmd.index("-ctv") + 1] == "q8_0"
    # default: no cache flags
    plain = prefill.build_llama_bench_command(Path("b"), Path("m"), [128], [0])
    assert "-ctk" not in plain and "-ctv" not in plain


def test_build_cmd_ubatch_and_flash_attn():
    # flags verified against the real b9873 llama-bench --help (2026-07-07)
    cmd = prefill.build_llama_bench_command(
        Path("b"), Path("m"), [128], [0], ubatch_size=1024, flash_attn="on"
    )
    assert cmd[cmd.index("-ub") + 1] == "1024"
    assert cmd[cmd.index("-fa") + 1] == "on"
    plain = prefill.build_llama_bench_command(Path("b"), Path("m"), [128], [0])
    assert "-ub" not in plain and "-fa" not in plain


def test_ttft_derivation():
    assert math.isclose(prefill.ttft_seconds(100.0, 200), 2.0)
    assert math.isinf(prefill.ttft_seconds(0.0, 200))


def test_build_cmd_steadiness_flags():
    # --prio / -mmp / --delay verified against the real b9873 llama-bench --help (2026-08-09)
    cmd = prefill.build_llama_bench_command(
        Path("b"), Path("m"), [128], [0], prio=2, mmap=False, delay=2
    )
    assert cmd[cmd.index("--prio") + 1] == "2"
    assert cmd[cmd.index("-mmp") + 1] == "0"
    assert cmd[cmd.index("--delay") + 1] == "2"
    # defaults (None) leave the command unchanged for other call sites
    plain = prefill.build_llama_bench_command(Path("b"), Path("m"), [128], [0])
    for flag in ("--prio", "-mmp", "--delay"):
        assert flag not in plain


def test_sweep_timeout_is_bencherror_not_crash(monkeypatch):
    import subprocess

    def boom(cmd, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(prefill, "run_with_peak_rss", boom)
    with pytest.raises(prefill.BenchError, match="timed out"):
        prefill.run_sweep(
            bench_bin=Path("b"),
            model_path=Path("m"),
            model_id="x",
            variant="q4_0",
            workload_name="w",
            prompt_lengths=[128],
            n_gen=32,
            threads=None,
            repetitions=1,
        )


def test_sweep_does_not_default_to_prio(monkeypatch):
    """run_sweep must not pass --prio by default.

    Raising scheduler priority needs privileges a container usually lacks, and llama-bench
    treats the refusal as fatal ("failed to set process priority ... Inappropriate ioctl
    for device"), so every config dies before measuring anything.
    """
    seen = {}

    def capture(cmd, timeout):
        seen["cmd"] = cmd
        raise prefill.BenchError("stop here; we only want the command")

    monkeypatch.setattr(prefill, "run_with_peak_rss", capture)
    with pytest.raises(prefill.BenchError):
        prefill.run_sweep(
            bench_bin=Path("b"),
            model_path=Path("m"),
            model_id="x",
            variant="q4_0",
            workload_name="w",
            prompt_lengths=[128],
            n_gen=32,
            threads=None,
            repetitions=1,
        )
    assert "--prio" not in seen["cmd"]
    # the other steadiness flags stay on: they need no privileges
    assert "-mmp" in seen["cmd"] and "--delay" in seen["cmd"]

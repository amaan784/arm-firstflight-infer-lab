from pathlib import Path

import pytest

from firstflight.bench import ttft
from firstflight.engines import llama_cpp


def test_build_prefix_scales():
    short = ttft.build_prefix(256)
    long = ttft.build_prefix(4096)
    assert len(long) > len(short) * 8
    # deterministic: same input -> same prefix
    assert ttft.build_prefix(1024) == ttft.build_prefix(1024)


def test_parse_timings():
    t = ttft.parse_timings(
        {
            "content": "x",
            "timings": {
                "prompt_n": 2048,
                "prompt_ms": 1234.5,
                "predicted_n": 16,
                "predicted_ms": 300.0,
            },
        }
    )
    assert t.prompt_n == 2048
    assert t.prompt_ms == 1234.5
    assert abs(t.prompt_s - 1.2345) < 1e-9


def test_parse_timings_bad_shape():
    with pytest.raises(ttft.TtftError):
        ttft.parse_timings({"content": "x"})


def test_ttft_result_reduction_and_roundtrip(tmp_path):
    cold = ttft.Timings(prompt_n=2048, prompt_ms=4000.0, predicted_n=16, predicted_ms=300.0)
    warm = ttft.Timings(prompt_n=24, prompt_ms=80.0, predicted_n=16, predicted_ms=300.0)
    res = ttft.TtftResult(
        timestamp="t",
        model="m",
        variant="q4_0",
        prefix_target_tokens=2048,
        cache_reuse=256,
        threads=4,
        cold=cold,
        warm=warm,
    )
    assert 97.0 < res.reduction_pct < 99.0  # 4000 -> 80 ms = 98%
    p = res.save_json(tmp_path / "ttft_x.json")
    import json

    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["cold"]["prompt_n"] == 2048 and d["warm"]["prompt_ms"] == 80.0
    assert d["kind"] == "measured-ttft-prompt-cache"


def test_build_server_cmd():
    cmd = ttft.build_server_cmd(
        Path("llama-server"), Path("m.gguf"), port=9000, threads=4, cache_reuse=128
    )
    assert cmd[cmd.index("--port") + 1] == "9000"
    assert cmd[cmd.index("--cache-reuse") + 1] == "128"
    assert cmd[cmd.index("-t") + 1] == "4"


def test_find_binary_server_kind(tmp_path):
    (tmp_path / "llama-server").write_text("#!/bin/sh\n")
    found = llama_cpp.find_binary("server", env_value=str(tmp_path))
    assert found is not None and found.name == "llama-server"

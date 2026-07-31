from pathlib import Path

from firstflight.bench import throughput as tp
from firstflight.engines import llama_cpp


def test_build_batched_bench_command():
    cmd = tp.build_batched_bench_command(
        Path("llama-batched-bench"),
        Path("m.gguf"),
        npp=1024,
        ntg=32,
        levels=[1, 2, 4, 8],
        threads=4,
    )
    assert cmd[cmd.index("-npp") + 1] == "1024"
    assert cmd[cmd.index("-ntg") + 1] == "32"
    assert cmd[cmd.index("-npl") + 1] == "1,2,4,8"
    assert cmd[cmd.index("--output-format") + 1] == "jsonl"
    # context sized to hold every parallel sequence: 8 * (1024+32) + margin
    assert int(cmd[cmd.index("-c") + 1]) >= 8 * (1024 + 32)


def test_parse_jsonl_tolerant():
    text = (
        "main: n_kv_max = 8704\n"
        '{"pp": 1024, "tg": 32, "pl": 1, "speed_pp": 300.5, "speed_tg": 45.0, "speed": 280.0}\n'
        '{"n_pp": 1024, "n_tg": 32, "npl": 4, "s_pp": 900.0, "s_tg": 120.0, "s": 800.0}\n'
        "not json\n"
    )
    pts = tp.parse_jsonl(text)
    assert len(pts) == 2
    assert pts[0].parallel == 1 and pts[0].speed_pp == 300.5
    assert pts[1].parallel == 4 and pts[1].speed_pp == 900.0  # alias keys accepted
    assert pts[0].raw["pp"] == 1024  # raw rows always preserved


def test_throughput_result_roundtrip(tmp_path):
    import json

    res = tp.ThroughputResult(
        timestamp="t",
        model="m",
        variant="q4_0",
        npp=1024,
        ntg=32,
        threads=4,
        points=[tp.ThroughputPoint(1, 1024, 32, 300.0, 45.0, 280.0, raw={"pp": 1024})],
    )
    p = res.save_json(tmp_path / "throughput_x.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["kind"] == "batched-throughput"
    assert d["points"][0]["parallel"] == 1 and d["points"][0]["raw"]["pp"] == 1024


def test_find_binary_batched_kind(tmp_path):
    (tmp_path / "llama-batched-bench").write_text("#!/bin/sh\n")
    found = llama_cpp.find_binary("batched", env_value=str(tmp_path))
    assert found is not None and found.name == "llama-batched-bench"

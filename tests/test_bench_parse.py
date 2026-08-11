import json
import math

import pytest

from firstflight.bench import prefill
from firstflight.bench.result import EngineInfo, HostInfo, ModelInfo, SweepResult

# `llama-bench -o json` output, schema confirmed 2026-06-26: top-level array of rows,
# pp rows (n_gen==0) + a tg row (n_prompt==0), with avg_ts/stddev_ts.
_COMMON = {
    "build_commit": "abc1234",
    "build_number": 9821,
    "cpu_info": "Neoverse-V2",
    "backends": "CPU",
    "model_filename": "qwen.gguf",
    "model_size": 491400032,
    "model_n_params": 494000000,
    "n_threads": 8,
}
SAMPLE = [
    {
        **_COMMON,
        "n_prompt": 128,
        "n_gen": 0,
        "avg_ts": 256.0,
        "stddev_ts": 2.0,
        "samples_ts": [255.0, 257.0],
    },
    {
        **_COMMON,
        "n_prompt": 512,
        "n_gen": 0,
        "avg_ts": 240.0,
        "stddev_ts": 3.0,
        "samples_ts": [238.0, 242.0],
    },
    {
        **_COMMON,
        "n_prompt": 0,
        "n_gen": 32,
        "avg_ts": 48.0,
        "stddev_ts": 0.5,
        "samples_ts": [47.5, 48.5],
    },
]


def test_parse_rows_str_and_list():
    assert len(prefill.parse_rows(SAMPLE)) == 3
    assert len(prefill.parse_rows(json.dumps(SAMPLE))) == 3


def test_row_kind():
    assert prefill.row_kind(SAMPLE[0]) == "pp"
    assert prefill.row_kind(SAMPLE[2]) == "tg"
    assert prefill.row_kind({"n_prompt": 0, "n_gen": 0}) == "mixed"


def test_rows_to_points_and_ttft():
    points = prefill.rows_to_points(SAMPLE)
    pp = [p for p in points if p.kind == "pp"]
    tg = [p for p in points if p.kind == "tg"]
    assert len(pp) == 2 and len(tg) == 1
    p128 = next(p for p in pp if p.n_prompt == 128)
    assert math.isclose(p128.throughput_tok_s, 256.0)
    assert math.isclose(p128.throughput_stddev, 2.0)
    assert math.isclose(p128.ttft_s, 128 / 256.0)  # 0.5s
    assert tg[0].ttft_s is None
    assert math.isclose(tg[0].throughput_tok_s, 48.0)


def test_parse_rows_rejects_non_array():
    with pytest.raises(prefill.BenchError):
        prefill.parse_rows('{"not": "an array"}')


def test_sweep_result_roundtrip(tmp_path):
    points = prefill.rows_to_points(SAMPLE)
    result = SweepResult(
        label="baseline",
        timestamp="2026-06-26T00:00:00+00:00",
        workload="prefill-scaling",
        host=HostInfo(
            platform="Linux-aarch64",
            arch="aarch64",
            cpu_count=8,
            is_arm_linux=True,
            cpu_info="Neoverse-V2",
        ),
        engine=EngineInfo(
            build_number=9821, build_commit="abc1234", backends="CPU", binary="/x/llama-bench"
        ),
        model=ModelInfo(
            id="qwen2.5-0.5b-instruct",
            variant="q4_k_m",
            filename="qwen.gguf",
            size_bytes=491400032,
            n_params=494000000,
        ),
        threads=8,
        repetitions=5,
        n_gen=32,
        peak_rss_bytes=1234567,
        points=points,
    )
    result.experiment = "quant-sweep"
    path = result.save_json(tmp_path / "r.json")
    loaded = SweepResult.load_json(path)
    assert loaded.label == "baseline"
    assert len(loaded.points) == 3
    assert loaded.prefill_points and loaded.gen_points
    assert loaded.engine.build_number == 9821
    assert loaded.experiment == "quant-sweep"
    assert loaded.schema_version == 1

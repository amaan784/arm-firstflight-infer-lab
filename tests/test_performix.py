from firstflight.profile import performix
from firstflight.profile.performix import Hotspot, PerformixProfiler, ProfileResult


def test_parse_hotspots_csv():
    text = "function,module,self%\nggml_mul_mat,libggml.so,42.1\nmemcpy,libc.so,3.2\n"
    spots = performix.parse_hotspots(text)
    assert spots[0].function == "ggml_mul_mat"
    assert spots[0].percent == 42.1
    assert spots[0].module == "libggml.so"
    assert spots[1].percent == 3.2  # sorted descending


def test_parse_hotspots_json():
    text = (
        '[{"function":"kai_matmul","percent":40.0,"module":"libggml.so"},'
        '{"symbol":"memcpy","self":2.5}]'
    )
    spots = performix.parse_hotspots(text)
    assert spots[0].function == "kai_matmul"
    assert spots[0].percent == 40.0
    assert spots[1].function == "memcpy"  # symbol -> function, self -> percent


def test_parse_hotspots_box_table():
    # the ┃-delimited table apx render emits (columns from Arm's code_hotspots query)
    text = (
        "┃ function_name ┃ node_type ┃ periodic_samples_self ┃ periodic_samples_self_percent ┃\n"
        "┃ __complex_abs ┃ function ┃ 65213 ┃ 65.2 ┃\n"
        "┃ __hypot ┃ function ┃ 18044 ┃ 18.0 ┃\n"
    )
    spots = performix.parse_hotspots(text)
    assert spots[0].function == "__complex_abs"
    assert spots[0].percent == 65.2
    assert spots[1].function == "__hypot"


def test_parse_hotspots_empty():
    assert performix.parse_hotspots("") == []
    assert performix.parse_hotspots("no,useful,columns\n1,2,3") == []


def test_extract_id_anchored():
    # JSON branch: exact key match
    assert performix._extract_id('{"run_id":"R1"}', ("run_id", "id")) == "R1"
    # regex fallback (non-JSON): 'id' must NOT match inside 'session_id'/'run_id'
    text = "session_id = ABC ; run_id = XYZ"
    assert performix._extract_id(text, ("run_id",)) == "XYZ"
    assert performix._extract_id(text, ("id",)) is None


def test_build_recipe_run_cmd():
    from pathlib import Path

    cmd = performix.build_recipe_run_cmd(Path("apx"), ["llama-bench", "-p", "2048"])
    assert cmd[:4] == ["apx", "recipe", "run", "code_hotspots"]
    assert "--workload=llama-bench -p 2048" in cmd
    assert "--json" in cmd and "--target=localhost" in cmd


def test_profiler_off_arm_is_skip():
    # off Arm Linux, Performix is unavailable and profile() skips instead of raising
    prof = PerformixProfiler()
    assert prof.available() is False
    res = prof.profile(["echo", "hi"])
    assert res.skipped is True
    assert res.reason


def test_profile_result_roundtrip(tmp_path):
    pr = ProfileResult(
        skipped=False,
        timestamp="2026-06-26T00:00:00+00:00",
        target="llama-bench -p 2048",
        hotspots=[Hotspot("ggml_mul_mat", 42.1, "libggml.so")],
    )
    path = pr.save_json(tmp_path / "profile_x.json")
    loaded = ProfileResult.load_json(path)
    assert loaded.hotspots[0].function == "ggml_mul_mat"
    assert loaded.skipped is False

"""Report data-logic tests — no matplotlib/jinja2 needed (chart/HTML tests live elsewhere)."""

from firstflight.config import InstanceSpec
from firstflight.report import render


def test_synthetic_and_model():
    results, instance = render.synthetic_results()
    assert len(results) == 2
    model = render.build_report_model(results, instance, demo=True)
    assert "faster prefill" in model.headline_main
    assert model.demo is True
    assert model.priced is True
    assert all(r.cost.priced for r in model.result_rows)
    assert 32768 in model.contexts
    assert model.metric_cards  # at-a-glance cards populated


def test_markdown_contains_headline_and_cost():
    results, instance = render.synthetic_results()
    model = render.build_report_model(results, instance, demo=True)
    md = render.build_markdown(model, [], "report-x")
    assert model.headline_main in md
    assert "$/M prompt tok" in md  # prompt-token cost matches the prefill headline
    assert "$/M gen tok" in md
    assert "32,768" in md  # context appears in the prefill table
    assert "DEMO" in md
    assert "derived" in md  # TTFT approximation caveat is stated in the report itself


def test_unpriced_instance_shows_set_price():
    results, _ = render.synthetic_results()
    inst = InstanceSpec(name="x", arch="arm64", cpu="cpu", vcpus=8, usd_per_hour=0.0)
    model = render.build_report_model(results, inst, demo=False)
    assert model.priced is False
    md = render.build_markdown(model, [], "s")
    assert "set price" in md or "n/a" in md


def test_single_result_headline():
    results, instance = render.synthetic_results()
    model = render.build_report_model([results[0]], instance)
    assert "tok/s prefill" in model.headline_main


def test_report_quality_column_and_headline():
    results, instance = render.synthetic_results()
    model = render.build_report_model(results, instance, demo=True)
    md = render.build_markdown(model, [], "s")
    assert "quality" in md
    assert "32/40" in md  # n/N display: honest about the probe's granularity
    assert any("quality 32/40" in s for s in model.headline_subs)


def test_prompt_cost_matches_prefill_story():
    # The headline is prefill/TTFT, so the primary cost card must be PROMPT-token cost
    # (computed from prefill throughput), not generation cost.
    results, instance = render.synthetic_results()
    model = render.build_report_model(results, instance, demo=True)
    assert any("$/M prompt tok" in label for _, label in model.metric_cards)
    assert any("prompt cost" in s for s in model.headline_subs)
    # per-row prompt cost: optimized pp@32k = 800 tok/s @ $0.319/hr
    opt = next(r for r in model.result_rows if r.label == "kleidiai-q4_0")
    expected = 0.319 / (800.0 * 3600.0) * 1e6
    assert abs(opt.prompt_cost.usd_per_million_tokens - expected) < 1e-6


def test_stddev_rendered_in_prefill_table():
    results, instance = render.synthetic_results()
    model = render.build_report_model(results, instance, demo=True)
    cell = model.prefill_table[0]["cells"]["baseline"]["tput"]
    assert "±" in cell  # spread across repetitions is shown, not hidden


def test_report_includes_hotspots():
    results, instance = render.synthetic_results()
    profile = render.synthetic_profile()
    model = render.build_report_model(results, instance, demo=True, profiles=[profile])
    assert model.hotspots
    md = render.build_markdown(model, [], "s")
    assert "Top hotspots" in md
    assert "kai_matmul" in md or "ggml" in md


def test_skipped_profile_is_ignored():
    from firstflight.profile.performix import ProfileResult

    results, instance = render.synthetic_results()
    skipped = ProfileResult(skipped=True, reason="off arm")
    model = render.build_report_model(results, instance, profiles=[skipped])
    assert model.hotspots == []


def test_committed_example_profiles_excluded(tmp_path):
    # Regression: profile_example.json (synthetic hotspots) must not load into real reports.
    from firstflight.profile.performix import ProfileResult

    ProfileResult(skipped=False, timestamp="t", target="x").save_json(
        tmp_path / "profile_example.json"
    )
    ProfileResult(skipped=False, timestamp="t", target="y").save_json(
        tmp_path / "profile_real.json"
    )
    profs = render.load_profiles(tmp_path)
    assert len(profs) == 1 and profs[0].target == "y"


def test_measured_ttft_and_throughput_render_in_report():
    # The newest evidence must appear in the WOW artifact, not just console output.
    results, instance = render.synthetic_results()
    model = render.build_report_model(
        results,
        instance,
        demo=True,
        ttft_results=[render.synthetic_ttft()],
        throughput_results=[render.synthetic_throughput()],
    )
    md = render.build_markdown(model, [], "s")
    assert "Measured TTFT" in md
    assert "97%" in md or "prefill saved" in md  # cold 1620ms -> warm 42ms
    assert "Throughput vs parallel requests" in md
    assert any("measured prompt-cache TTFT" in s for s in model.headline_subs)


def test_ttft_throughput_loaders(tmp_path):
    render.synthetic_ttft().save_json(tmp_path / "ttft_20260101.json")
    render.synthetic_throughput().save_json(tmp_path / "throughput_20260101.json")
    # sweep loader must NOT try to parse them; dedicated loaders must find them
    assert render.load_results(tmp_path) == []
    assert len(render.load_ttft_results(tmp_path)) == 1
    assert len(render.load_throughput_results(tmp_path)) == 1
    assert render.load_ttft_results(tmp_path)[0].cold.prompt_n == 2071


def test_duplicate_labels_deduped_keep_newest():
    results, instance = render.synthetic_results()
    old = results[0]  # label "baseline", timestamp 2026-06-26
    import copy

    newer = copy.deepcopy(old)
    newer.timestamp = "2026-07-01T00:00:00+00:00"
    newer.points[0].throughput_tok_s = 9999.0
    model = render.build_report_model([old, newer, results[1]], instance, demo=True)
    assert model.labels.count("baseline") == 1  # no duplicate columns
    assert len(model.duplicates_dropped) == 1
    base_cell = model.prefill_table[0]["cells"]["baseline"]["tput"]
    assert base_cell.startswith("9999")  # the NEWEST run won


def test_load_results_roundtrip(tmp_path):
    results, _ = render.synthetic_results()
    results[0].save_json(tmp_path / "a.json")
    results[1].save_json(tmp_path / "b.json")
    loaded = render.load_results(tmp_path)
    assert len(loaded) == 2


def test_committed_examples_never_contaminate_real_reports(tmp_path):
    # Regression: the repo ships synthetic example_*.json; a real run's report must not load
    # them (they'd force the DEMO banner and become the headline baseline).
    results, _ = render.synthetic_results()
    results[0].save_json(tmp_path / "example_baseline_q4_k_m.json")
    results[1].save_json(tmp_path / "real_run.json")
    loaded = render.load_results(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].label == results[1].label

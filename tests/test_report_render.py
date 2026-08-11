"""Full render tests. Need the [report] extra (matplotlib + jinja2)."""

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("jinja2")

from click.testing import CliRunner  # noqa: E402

from firstflight.cli import main  # noqa: E402
from firstflight.report import render  # noqa: E402


def test_render_demo_writes_files(tmp_path):
    paths = render.render_report(demo=True, out_dir=tmp_path)
    assert paths is not None
    assert paths.html.exists() and paths.markdown.exists()
    assert len(paths.charts) >= 2 and all(p.exists() for p in paths.charts)

    html = paths.html.read_text(encoding="utf-8")
    assert "data:image/png;base64," in html  # charts inlined -> standalone
    assert "DEMO" in html
    assert "faster prefill" in html
    assert "$/M prompt tok" in html

    md = paths.markdown.read_text(encoding="utf-8")
    assert ".png)" in md  # references sibling chart images


def test_render_no_results_returns_none(tmp_path):
    assert render.render_report(results_source=tmp_path, out_dir=tmp_path) is None


def test_cost_chart_uses_prompt_cost_and_skips_nonfinite():
    # cost chart is PROMPT-token cost (prefill throughput). pp-only result -> finite cost, so
    # the chart is included. A result with NO prefill points costs +inf and must be skipped,
    # never plotted as an infinite bar.
    from firstflight.bench.result import BenchPoint, EngineInfo, HostInfo, ModelInfo, SweepResult
    from firstflight.config import InstanceSpec

    def mk(label, points):
        return SweepResult(
            label=label,
            timestamp="t",
            workload="w",
            host=HostInfo("Linux-aarch64", "aarch64", 8, True, "cpu"),
            engine=EngineInfo(),
            model=ModelInfo(id="m", variant="q4_0"),
            threads=8,
            repetitions=1,
            n_gen=0,
            peak_rss_bytes=None,
            points=points,
        )

    pp_only = mk("pp-only", [BenchPoint("pp", 2048, 0, 1000.0, 5.0, 2.048, 8)])
    tg_only = mk("tg-only", [BenchPoint("tg", 0, 32, 50.0, 1.0, None, 8)])
    inst = InstanceSpec(name="x", arch="arm64", cpu="cpu", vcpus=8, usd_per_hour=0.3)
    names = [c.name for c in render.build_charts([pp_only, tg_only], inst)]
    assert "cost" in names  # finite prompt cost from the pp-only result
    assert "prefill-ttft" in names


def test_cli_report_demo(tmp_path):
    res = CliRunner().invoke(main, ["report", "--demo", "--out-dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "HTML" in res.output
    assert list(tmp_path.glob("*.html"))

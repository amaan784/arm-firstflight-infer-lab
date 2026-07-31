"""Render the before/after report — the WOW artifact.

Reads `bench/results/*.json` and emits a one-page report in two forms:
  - standalone HTML (`bench/reports/*.html`) — self-contained, charts inlined as base64 PNG
  - markdown (`bench/reports/*.md`) — diff-friendly, references sibling PNG charts

The report LEADS with the headline result (best before/after prefill-TTFT delta, or the
long-context prefill number for a single run), then metric cards, charts, and tables for
prefill scaling, throughput, peak memory, and **$/M tokens** (via `cost.py` + `instances.yaml`).

matplotlib + jinja2 come from the `[report]` extra and are imported lazily, so loading this
module (and `firstflight report` with no results) works without them.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..bench.result import BenchPoint, SweepResult
from ..config import InstanceSpec, load_instances
from ..cost import CostResult
from ..cost import compute as compute_cost
from ..util import bytes_human, reports_dir, results_dir

PALETTE = ["#0B7285", "#E8590C", "#5F3DC4", "#2B8A3E", "#A61E4D", "#1864AB"]


class ReportError(RuntimeError):
    """Something prevented rendering (e.g. the [report] extra isn't installed)."""


# --- loading ------------------------------------------------------------------


def _is_committed_example(path: Path) -> bool:
    """The repo ships synthetic example_*.json / profile_example.json for illustration.

    They must NEVER mix into a real report (they would force the DEMO banner and become the
    headline baseline), so the loaders skip them whenever any real result is present-or-not —
    the demo path injects synthetic data explicitly instead of reading these files.
    """
    return path.name.startswith("example_") or path.name == "profile_example.json"


def load_results(source: Path | None = None) -> list[SweepResult]:
    """Load real SweepResults from a directory (default bench/results), sorted by timestamp.

    Committed synthetic examples (example_*.json) are excluded — see _is_committed_example.
    """
    source = source or results_dir()
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    out: list[SweepResult] = []
    for f in files:
        if _is_committed_example(f):
            continue
        try:
            out.append(SweepResult.load_json(f))
        except Exception:  # noqa: BLE001 - skip unparseable, keep going
            continue
    out.sort(key=lambda r: r.timestamp)
    return out


# --- computed view ------------------------------------------------------------


@dataclass
class ResultRow:
    label: str
    model: str
    threads: int
    build: str
    gen_tput: float
    peak_rss: int | None
    cost: CostResult  # $/M GENERATED tokens (from generation throughput)
    prompt_cost: CostResult | None = None  # $/M PROMPT tokens (prefill tput @ row's max ctx)
    quality_acc: float | None = None
    quality_counts: tuple[int, int] | None = None  # (n_correct, n_total)
    kleidiai: bool | None = None  # detected-active (proof), None = unknown
    experiment: str = ""  # which experiment produced this row ("" = ad-hoc)


@dataclass
class ReportModel:
    title: str
    generated_at: str
    demo: bool
    instance_name: str
    instance_cpu: str
    priced: bool
    cpu_info: str
    tool_version: str
    headline_main: str
    headline_subs: list[str] = field(default_factory=list)
    metric_cards: list[tuple[str, str]] = field(default_factory=list)
    result_rows: list[ResultRow] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    contexts: list[int] = field(default_factory=list)
    prefill_table: list[dict] = field(default_factory=list)
    quality_note: str = ""
    hotspots: list = field(default_factory=list)
    profile_target: str = ""
    thp: str = ""  # transparent-hugepage mode captured with the results (run evidence)


def _prefill_map(r: SweepResult) -> dict[int, BenchPoint]:
    return {p.n_prompt: p for p in r.prefill_points}


def _gen_tput(r: SweepResult) -> float:
    return r.gen_points[0].throughput_tok_s if r.gen_points else 0.0


def _ctx_label(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


def _fmt_ttft(t: float | None) -> str:
    if t is None or t != t or t in (float("inf"),):
        return "-"
    return f"{t:.2f}"


def _quality_acc(q) -> float | None:
    if isinstance(q, dict) and q.get("accuracy") is not None:
        return float(q["accuracy"])
    return None


def _quality_counts(q) -> tuple[int, int] | None:
    if isinstance(q, dict) and q.get("n_total"):
        return int(q.get("n_correct", 0)), int(q["n_total"])
    return None


def _fmt_quality(counts: tuple[int, int] | None, acc: float | None) -> str:
    """Prefer '32/40' (honest granularity) over a percentage; '-' when absent."""
    if counts is not None:
        return f"{counts[0]}/{counts[1]}"
    return f"{acc:.0%}" if acc is not None else "-"


def _prompt_tput_max_ctx(r: SweepResult) -> tuple[float, int]:
    """Prefill throughput at this result's LARGEST context (the headline regime)."""
    pts = r.prefill_points
    if not pts:
        return 0.0, 0
    top = max(pts, key=lambda p: p.n_prompt)
    return top.throughput_tok_s, top.n_prompt


def _fmt_kleidiai(k: bool | None) -> str:
    return "-" if k is None else ("yes" if k else "no")


def build_report_model(
    results: list[SweepResult],
    instance: InstanceSpec,
    *,
    demo: bool = False,
    profiles: list | None = None,
    title: str = "Arm FirstFlight - Inference Optimization Report",
) -> ReportModel:
    if not results:
        raise ReportError("no results to render")

    # Honesty: if any result is synthetic, force the DEMO banner even outside --demo mode
    # (e.g. when rendering the committed example results).
    demo = demo or any("SYNTHETIC" in (r.host.cpu_info or "") for r in results)

    price = instance.usd_per_hour
    priced = price > 0

    # Per-result summary rows. Two cost views: $/M generated tokens (gen throughput) and
    # $/M PROMPT tokens (prefill throughput at the row's largest context — the metric that
    # matches the prefill/TTFT headline).
    rows: list[ResultRow] = []
    for r in results:
        p_tput, _p_ctx = _prompt_tput_max_ctx(r)
        rows.append(
            ResultRow(
                label=r.label,
                model=f"{r.model.id}:{r.model.variant}",
                threads=r.threads,
                build=f"b{r.engine.build_number}" if r.engine.build_number else "",
                gen_tput=_gen_tput(r),
                peak_rss=r.peak_rss_bytes,
                cost=compute_cost(_gen_tput(r), price),
                prompt_cost=compute_cost(p_tput, price) if p_tput else None,
                quality_acc=_quality_acc(r.quality),
                quality_counts=_quality_counts(r.quality),
                kleidiai=r.host.kleidiai,
                experiment=r.experiment,
            )
        )

    # Prefill-scaling table: union of contexts across results.
    contexts = sorted({p.n_prompt for r in results for p in r.prefill_points})
    labels = [r.label for r in results]
    maps = {r.label: _prefill_map(r) for r in results}
    prefill_table = []
    for ctx in contexts:
        cells = {}
        for r in results:
            pt = maps[r.label].get(ctx)
            cells[r.label] = {
                "ttft": _fmt_ttft(pt.ttft_s) if pt else "-",
                # tokens/sec with the spread across repetitions (honesty about noise)
                "tput": f"{pt.throughput_tok_s:.0f} ±{pt.throughput_stddev:.0f}" if pt else "-",
            }
        prefill_table.append({"ctx": ctx, "ctx_label": _ctx_label(ctx), "cells": cells})

    # Headline: comparison if a baseline + another result exist, else single-run.
    baseline = next((r for r in results if r.label.lower() == "baseline"), results[0])
    others = [r for r in results if r is not baseline]
    headline_main, subs, cards = _headline(baseline, others, rows, instance, priced)

    has_quality = any(_quality_acc(r.quality) is not None for r in results)
    quality = (
        "Quality = a small exact-match probe via llama-cli (a regression guardrail, not a "
        "leaderboard) showing the speedup did not tank accuracy."
        if has_quality
        else "Run `firstflight experiment` to add the quality-delta column (proves the "
        "speedup did not degrade accuracy)."
    )

    # Newest non-skipped Performix profile, if any.
    hotspots: list = []
    profile_target = ""
    for p in reversed(profiles or []):
        if not p.skipped and p.hotspots:
            hotspots = p.hotspots
            profile_target = p.target
            break

    return ReportModel(
        title=title,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        demo=demo,
        instance_name=instance.name,
        instance_cpu=instance.cpu,
        priced=priced,
        cpu_info=results[0].host.cpu_info or instance.cpu,
        tool_version=results[0].tool_version,
        headline_main=headline_main,
        headline_subs=subs,
        metric_cards=cards,
        result_rows=rows,
        labels=labels,
        contexts=contexts,
        prefill_table=prefill_table,
        quality_note=quality,
        hotspots=hotspots,
        profile_target=profile_target,
        thp=getattr(results[0].host, "thp", "") or "",
    )


def _cost_for(rows: list[ResultRow], label: str) -> CostResult | None:
    return next((r.cost for r in rows if r.label == label), None)


def _quality_for(rows: list[ResultRow], label: str):
    """(accuracy, (n_correct, n_total)|None) for a labelled row, or (None, None)."""
    for r in rows:
        if r.label == label:
            return r.quality_acc, r.quality_counts
    return None, None


def _headline(baseline, others, rows, instance, priced):
    """Return (headline_main, subs, metric_cards)."""
    base_map = _prefill_map(baseline)

    # Choose the strongest "after": highest prefill throughput at the largest shared context.
    best = None
    best_ctx = None
    for cand in others:
        cand_map = _prefill_map(cand)
        shared = sorted(set(base_map) & set(cand_map))
        if not shared:
            continue
        ctx = shared[-1]
        if (
            best is None
            or cand_map[ctx].throughput_tok_s > _prefill_map(best)[ctx].throughput_tok_s
        ):
            best, best_ctx = cand, ctx

    if best is not None and best_ctx is not None:
        b = base_map[best_ctx]
        o = _prefill_map(best)[best_ctx]
        speed = (o.throughput_tok_s / b.throughput_tok_s) if b.throughput_tok_s else 0.0
        subs = [
            f"TTFT {_fmt_ttft(b.ttft_s)}s -> {_fmt_ttft(o.ttft_s)}s at {best_ctx:,} tokens "
            f"({best.label} vs {baseline.label})",
            f"prefill {b.throughput_tok_s:.0f} -> {o.throughput_tok_s:.0f} tok/s",
            f"generation {_gen_tput(baseline):.0f} -> {_gen_tput(best):.0f} tok/s",
        ]
        cards = [
            (f"{speed:.2f}x", f"prefill speedup @ {_ctx_label(best_ctx)}"),
            (f"{_fmt_ttft(o.ttft_s)}s", f"TTFT @ {_ctx_label(best_ctx)} ({best.label})"),
            (f"{o.throughput_tok_s:.0f}", "prefill tok/s"),
        ]
        if priced:
            # Prompt-token cost from the SAME prefill points as the headline (context-matched
            # to the prefill story); generation cost is secondary context.
            bp = compute_cost(b.throughput_tok_s, instance.usd_per_hour)
            op = compute_cost(o.throughput_tok_s, instance.usd_per_hour)
            subs.append(
                f"prompt cost {bp.format_usd_per_mtok()} -> {op.format_usd_per_mtok()} "
                f"per M tokens at {best_ctx:,}-token context"
            )
            bc, oc = _cost_for(rows, baseline.label), _cost_for(rows, best.label)
            if bc and oc:
                subs.append(
                    f"generation cost {bc.format_usd_per_mtok()} -> "
                    f"{oc.format_usd_per_mtok()} per M tokens"
                )
            cards.append((op.format_usd_per_mtok(), f"$/M prompt tok @ {_ctx_label(best_ctx)}"))
        else:
            cards.append(("set price", "$/M prompt tok"))
        bq, bqc = _quality_for(rows, baseline.label)
        oq, oqc = _quality_for(rows, best.label)
        if bq is not None and oq is not None:
            if bqc and oqc:
                # one-probe-item tolerance: honest about the instrument's granularity
                held = "held" if oqc[0] >= bqc[0] - 1 else "down"
                subs.append(f"quality {_fmt_quality(bqc, bq)} -> {_fmt_quality(oqc, oq)} ({held})")
                cards.append((_fmt_quality(oqc, oq), "quality (probe)"))
            else:
                held = "held" if oq >= bq - 0.02 else "down"
                subs.append(f"quality {bq:.0%} -> {oq:.0%} ({held})")
                cards.append((f"{oq:.0%}", "quality (probe)"))
        return f"{speed:.2f}x faster prefill at {best_ctx:,}-token context", subs, cards

    # Single-run headline.
    if not base_map:
        return "Benchmark complete", ["No prefill points found in the result."], []
    ctx = max(base_map)
    p = base_map[ctx]
    subs = [
        f"TTFT {_fmt_ttft(p.ttft_s)}s at {ctx:,} tokens",
        f"generation {_gen_tput(baseline):.0f} tok/s",
        f"CPU: {baseline.host.cpu_info or instance.cpu}",
    ]
    cards = [
        (f"{p.throughput_tok_s:.0f}", f"prefill tok/s @ {_ctx_label(ctx)}"),
        (f"{_fmt_ttft(p.ttft_s)}s", f"TTFT @ {_ctx_label(ctx)}"),
        (f"{_gen_tput(baseline):.0f}", "gen tok/s"),
    ]
    if priced:
        pc = compute_cost(p.throughput_tok_s, instance.usd_per_hour)
        cards.append((pc.format_usd_per_mtok(), f"$/M prompt tok @ {_ctx_label(ctx)}"))
    else:
        cards.append(("set price", "$/M prompt tok"))
    return f"{p.throughput_tok_s:.0f} tok/s prefill at {ctx:,}-token context", subs, cards


# --- charts (matplotlib, lazy) -----------------------------------------------


@dataclass
class Chart:
    name: str
    title: str
    png: bytes

    @property
    def data_uri(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png).decode("ascii")


def _require_report_deps():
    try:
        import jinja2  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ReportError(
            'report rendering needs the [report] extra: pip install -e ".[report]"'
        ) from exc


def _fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight")
    return buf.getvalue()


def build_charts(results: list[SweepResult], instance: InstanceSpec) -> list[Chart]:
    _require_report_deps()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#adb5bd",
            "axes.grid": True,
            "grid.color": "#e9ecef",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )

    charts: list[Chart] = []
    all_ctx = sorted({p.n_prompt for r in results for p in r.prefill_points})

    def line_chart(name, title, ylabel, value_fn):
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for i, r in enumerate(results):
            pts = sorted(r.prefill_points, key=lambda p: p.n_prompt)
            # Skip missing/non-finite values entirely — plotting them as 0.0 would draw a
            # misleading dip to zero.
            pairs = []
            for p in pts:
                v = value_fn(p)
                if v is not None and math.isfinite(v):
                    pairs.append((p.n_prompt, v))
            if not pairs:
                continue
            xs, ys = zip(*pairs, strict=True)
            ax.plot(
                xs, ys, marker="o", linewidth=2.2, color=PALETTE[i % len(PALETTE)], label=r.label
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("context length (prompt tokens)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if all_ctx:
            ax.set_xticks(all_ctx)
            ax.set_xticklabels([_ctx_label(x) for x in all_ctx])
            ax.minorticks_off()
        if len(results) > 1:
            ax.legend()
        fig.tight_layout()
        png = _fig_png(fig)
        plt.close(fig)
        charts.append(Chart(name, title, png))

    line_chart(
        "prefill-ttft",
        "Prefill TTFT vs context length (lower is better; derived)",
        "time-to-first-token (s, derived)",
        lambda p: p.ttft_s,
    )
    line_chart(
        "prefill-throughput",
        "Prefill throughput vs context length (higher is better)",
        "prefill tokens/sec",
        lambda p: p.throughput_tok_s,
    )

    # Cost bar chart only when priced. PROMPT-token cost (prefill throughput at each result's
    # largest context) — the metric matching the prefill/TTFT headline. Non-finite costs
    # (no prefill points) are skipped rather than plotted.
    if instance.usd_per_hour > 0:
        labels, costs = [], []
        for r in results:
            p_tput, _ = _prompt_tput_max_ctx(r)
            c = compute_cost(p_tput, instance.usd_per_hour).usd_per_million_tokens
            if math.isfinite(c) and c > 0:
                labels.append(r.label)
                costs.append(c)
        if costs:
            fig, ax = plt.subplots(figsize=(7.2, 3.4))
            bars = ax.bar(
                labels, costs, color=[PALETTE[i % len(PALETTE)] for i in range(len(labels))]
            )
            ax.set_ylabel("$ / million prompt tokens")
            ax.set_title("Cost per million PROMPT tokens at largest context (lower is better)")
            ax.bar_label(bars, fmt="$%.3f", padding=3)
            fig.tight_layout()
            png = _fig_png(fig)
            plt.close(fig)
            charts.append(Chart("cost", "Cost per million prompt tokens", png))

    return charts


# --- markdown -----------------------------------------------------------------


def build_markdown(model: ReportModel, charts: list[Chart], stem: str) -> str:
    lines: list[str] = []
    if model.demo:
        lines.append("> **DEMO - synthetic illustrative data, not a measured run.**\n")
    lines.append(f"# {model.title}\n")
    lines.append(f"## {model.headline_main}\n")
    for s in model.headline_subs:
        lines.append(f"- {s}")
    lines.append("")

    # metric cards as a compact table
    if model.metric_cards:
        lines.append("| " + " | ".join(v for v, _ in model.metric_cards) + " |")
        lines.append("|" + "---|" * len(model.metric_cards))
        lines.append("| " + " | ".join(lbl for _, lbl in model.metric_cards) + " |")
        lines.append("")

    for ch in charts:
        lines.append(f"### {ch.title}")
        lines.append(f"![{ch.title}]({stem}-{ch.name}.png)\n")

    # results summary table
    lines.append("## Runs\n")
    lines.append(
        "| label | model:variant | threads | build | kleidiai | gen tok/s | peak mem "
        "| $/M prompt tok | $/M gen tok | quality |"
    )
    lines.append("|---|---|---:|---|---|---:|---:|---:|---:|---:|")
    grouped = len({r.experiment for r in model.result_rows if r.experiment}) > 1
    last_exp = object()
    for r in model.result_rows:
        if grouped and r.experiment != last_exp:
            name = r.experiment or "ad-hoc"
            lines.append(f"| **{name}** |  |  |  |  |  |  |  |  |  |")
            last_exp = r.experiment
        mem = bytes_human(r.peak_rss) if r.peak_rss else "-"
        pcost = r.prompt_cost.format_usd_per_mtok() if r.prompt_cost else "-"
        lines.append(
            f"| {r.label} | {r.model} | {r.threads} | {r.build} | {_fmt_kleidiai(r.kleidiai)} | "
            f"{r.gen_tput:.0f} | {mem} | {pcost} | {r.cost.format_usd_per_mtok()} | "
            f"{_fmt_quality(r.quality_counts, r.quality_acc)} |"
        )
    lines.append("")

    # prefill scaling table
    lines.append("## Prefill scaling (TTFT seconds / prefill tok/s ± stddev)\n")
    lines.append(
        "_TTFT is derived as prompt_tokens ÷ prefill throughput; it excludes tokenization "
        "and model-load time. ± is the spread across repetitions._\n"
    )
    header = "| context | " + " | ".join(f"{lbl} TTFT | {lbl} tok/s" for lbl in model.labels) + " |"
    sep = "|---|" + "---:|---:|" * len(model.labels)
    lines.append(header)
    lines.append(sep)
    for row in model.prefill_table:
        cells = []
        for lbl in model.labels:
            c = row["cells"][lbl]
            cells.append(f"{c['ttft']} | {c['tput']}")
        lines.append(f"| {row['ctx']:,} | " + " | ".join(cells) + " |")
    lines.append("")

    if model.hotspots:
        lines.append("## Top hotspots (Arm Performix)\n")
        if model.profile_target:
            lines.append(f"_profiled: `{model.profile_target}`_\n")
        lines.append("| function | module | % |")
        lines.append("|---|---|---:|")
        for h in model.hotspots:
            lines.append(f"| `{h.function}` | {h.module} | {h.percent:.1f} |")
        lines.append("")

    lines.append(f"_{model.quality_note}_\n")
    thp = f" | THP: {model.thp}" if model.thp else ""
    lines.append(
        f"<sub>Instance: {model.instance_name} ({model.instance_cpu}) | CPU: {model.cpu_info}"
        f"{thp} | firstflight {model.tool_version} | generated {model.generated_at}. "
        f"Methodology: docs/METHODOLOGY.md</sub>"
    )
    return "\n".join(lines)


# --- html (jinja2, lazy) ------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ m.title }}</title>
<style>
:root{ --accent:#0B7285; --ink:#212529; --muted:#868e96; --line:#e9ecef; --bg:#f8f9fa; }
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.5}
.wrap{max-width:960px;margin:0 auto;padding:32px 20px 64px}
.demo{background:#E8590C;color:#fff;font-weight:600;text-align:center;padding:10px;border-radius:8px;margin-bottom:24px}
.kicker{letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:700;font-size:12px}
h1.head{font-size:30px;line-height:1.2;margin:6px 0 12px}
.subs{color:#495057;margin:0 0 24px;padding:0;list-style:none}
.subs li{padding:3px 0 3px 22px;position:relative}
.subs li::before{content:"";position:absolute;left:0;top:11px;width:8px;height:8px;border-radius:50%;background:var(--accent)}
.cards{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 28px}
.card{flex:1 1 150px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;
 box-shadow:0 1px 2px rgba(0,0,0,.04)}
.card .v{font-size:26px;font-weight:700;color:var(--accent)}
.card .l{font-size:12px;color:var(--muted);margin-top:2px}
section{background:#fff;border:1px solid var(--line);border-radius:12px;padding:20px;margin:0 0 20px}
section h2{margin:0 0 14px;font-size:16px}
img{max-width:100%;height:auto;display:block;margin:0 auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:600;border-bottom:2px solid var(--line)}
tbody tr:nth-child(even){background:#fcfcfd}
.note{color:var(--muted);font-size:13px;margin-top:8px}
footer{color:var(--muted);font-size:12px;margin-top:28px;border-top:1px solid var(--line);padding-top:14px}
.gridcharts{display:grid;gap:18px}
</style></head>
<body><div class="wrap">
{% if m.demo %}<div class="demo">DEMO - synthetic illustrative data, not a measured run</div>{% endif %}
<div class="kicker">Arm FirstFlight - Inference Optimization</div>
<h1 class="head">{{ m.headline_main }}</h1>
<ul class="subs">{% for s in m.headline_subs %}<li>{{ s }}</li>{% endfor %}</ul>
<div class="cards">{% for v,l in m.metric_cards %}<div class="card"><div class="v">{{ v }}</div><div class="l">{{ l }}</div></div>{% endfor %}</div>

<section class="gridcharts">{% for c in charts %}<img alt="{{ c.title }}" src="{{ c.data_uri }}">{% endfor %}</section>

<section><h2>Runs</h2><table>
<thead><tr><th>label</th><th>model:variant</th><th>threads</th><th>build</th><th>kleidiai</th><th>gen tok/s</th><th>peak mem</th><th>$/M prompt tok</th><th>$/M gen tok</th><th>quality</th></tr></thead>
<tbody>{% for r in m.result_rows %}{% if m.grouped and r.show_group %}<tr><td colspan="10" style="text-align:left;font-weight:700;background:#f1f3f5">{{ r.experiment or "ad-hoc" }}</td></tr>{% endif %}<tr><td>{{ r.label }}</td><td>{{ r.model }}</td><td>{{ r.threads }}</td><td>{{ r.build }}</td><td>{{ r.kleidiai_human }}</td><td>{{ "%.0f"|format(r.gen_tput) }}</td><td>{{ r.peak_human }}</td><td>{{ r.prompt_cost_human }}</td><td>{{ r.cost.format_usd_per_mtok() }}</td><td>{{ r.quality_human }}</td></tr>{% endfor %}</tbody>
</table>{% if not m.priced %}<div class="note">$/M tokens shows "set price" until you set the instance hourly price in configs/instances.yaml.</div>{% endif %}</section>

<section><h2>Prefill scaling</h2>
<div class="note">TTFT is derived as prompt_tokens &divide; prefill throughput; excludes tokenization and model-load time. &plusmn; is the spread across repetitions.</div>
<table>
<thead><tr><th>context</th>{% for l in m.labels %}<th>{{ l }} TTFT (s)</th><th>{{ l }} tok/s</th>{% endfor %}</tr></thead>
<tbody>{% for row in m.prefill_table %}<tr><td>{{ "{:,}".format(row.ctx) }}</td>{% for l in m.labels %}<td>{{ row.cells[l].ttft }}</td><td>{{ row.cells[l].tput }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></section>

{% if m.hotspots %}<section><h2>Top hotspots (Arm Performix)</h2>
{% if m.profile_target %}<div class="note">profiled: <code>{{ m.profile_target }}</code></div>{% endif %}
<table><thead><tr><th>function</th><th>module</th><th>% CPU</th></tr></thead>
<tbody>{% for h in m.hotspots %}<tr><td>{{ h.function }}</td><td>{{ h.module }}</td><td>{{ "%.1f"|format(h.percent) }}</td></tr>{% endfor %}</tbody>
</table></section>{% endif %}

<div class="note">{{ m.quality_note }}</div>
<footer>Instance: {{ m.instance_name }} ({{ m.instance_cpu }}) &middot; CPU: {{ m.cpu_info }}{% if m.thp %} &middot; THP: {{ m.thp }}{% endif %} &middot;
 firstflight {{ m.tool_version }} &middot; generated {{ m.generated_at }} &middot;
 Methodology: docs/METHODOLOGY.md</footer>
</div></body></html>
"""


def build_html(model: ReportModel, charts: list[Chart]) -> str:
    _require_report_deps()
    import jinja2

    # Attach human-readable strings + grouping markers for the template.
    model.grouped = len({r.experiment for r in model.result_rows if r.experiment}) > 1  # type: ignore[attr-defined]
    last_exp = object()
    for r in model.result_rows:
        r.peak_human = bytes_human(r.peak_rss) if r.peak_rss else "-"  # type: ignore[attr-defined]
        r.quality_human = _fmt_quality(r.quality_counts, r.quality_acc)  # type: ignore[attr-defined]
        r.kleidiai_human = _fmt_kleidiai(r.kleidiai)  # type: ignore[attr-defined]
        r.prompt_cost_human = (  # type: ignore[attr-defined]
            r.prompt_cost.format_usd_per_mtok() if r.prompt_cost else "-"
        )
        r.show_group = r.experiment != last_exp  # type: ignore[attr-defined]
        last_exp = r.experiment

    template = jinja2.Environment(autoescape=jinja2.select_autoescape(["html"])).from_string(
        _HTML_TEMPLATE
    )
    return template.render(m=model, charts=charts)


# --- orchestration ------------------------------------------------------------


@dataclass
class ReportPaths:
    html: Path
    markdown: Path
    charts: list[Path]


def render_report(
    *,
    results: list[SweepResult] | None = None,
    results_source: Path | None = None,
    out_dir: Path | None = None,
    instance_name: str | None = None,
    demo: bool = False,
) -> ReportPaths | None:
    """Render the report. Returns paths, or None when there are no results to render."""
    instances = load_instances()

    if demo:
        results, instance = synthetic_results()
        profiles = []
    else:
        results = results if results is not None else load_results(results_source)
        instance = instances.get(instance_name)
        profiles = []

    if not results:
        return None

    model = build_report_model(results, instance, demo=demo, profiles=profiles)
    charts = build_charts(results, instance)  # raises ReportError if deps missing

    out_dir = out_dir or reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "report-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + ("-demo" if demo else "")

    chart_paths: list[Path] = []
    for ch in charts:
        p = out_dir / f"{stem}-{ch.name}.png"
        p.write_bytes(ch.png)
        chart_paths.append(p)

    html_path = out_dir / f"{stem}.html"
    html_path.write_text(build_html(model, charts), encoding="utf-8")
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(build_markdown(model, charts, stem), encoding="utf-8")

    return ReportPaths(html=html_path, markdown=md_path, charts=chart_paths)


# --- synthetic demo data ------------------------------------------------------


def synthetic_results() -> tuple[list[SweepResult], InstanceSpec]:
    """Plausible-but-fake Arm Graviton4 numbers for a clearly-labeled DEMO report.

    NOT measured data. Used by `firstflight report --demo` so the report layout can be seen
    (and the demo video recorded) without an Arm box. Real reports come from `firstflight bench`.
    """
    from ..bench.result import EngineInfo, HostInfo, ModelInfo

    contexts = [128, 512, 2048, 8192, 16384, 32768]
    base_pp = {128: 1400, 512: 1350, 2048: 1200, 8192: 900, 16384: 700, 32768: 520}
    opt_pp = {128: 2150, 512: 2080, 2048: 1850, 8192: 1380, 16384: 1080, 32768: 800}

    def make(label, variant, pp, gen, peak, kleidiai, quality):
        host = HostInfo(
            platform="Linux-aarch64",
            arch="aarch64",
            cpu_count=8,
            is_arm_linux=True,
            cpu_info="Arm Neoverse-V2 (Graviton4) [SYNTHETIC]",
            kleidiai=kleidiai,
        )
        engine = EngineInfo(
            build_number=9873, build_commit="demo0000", backends="CPU", binary="llama-bench"
        )
        model = ModelInfo(
            id="qwen2.5-0.5b-instruct",
            variant=variant,
            filename=f"{variant}.gguf",
            size_bytes=491400032,
            n_params=494000000,
        )
        points = []
        for c in contexts:
            tput = pp[c]
            points.append(BenchPoint("pp", c, 0, float(tput), tput * 0.015, c / tput, 8))
        points.append(BenchPoint("tg", 0, 32, float(gen), gen * 0.02, None, 8))
        return SweepResult(
            label=label,
            timestamp="2026-06-26T12:00:00+00:00",
            workload="prefill-scaling",
            host=host,
            engine=engine,
            model=model,
            threads=8,
            repetitions=5,
            n_gen=32,
            peak_rss_bytes=peak,
            points=points,
            quality=quality,
        )

    q = {"method": "builtin-probe", "accuracy": 0.8, "n_correct": 32, "n_total": 40}
    baseline = make("baseline", "q4_k_m", base_pp, 55, 730 * 1024 * 1024, False, q)
    optimized = make("kleidiai-q4_0", "q4_0", opt_pp, 85, 690 * 1024 * 1024, True, dict(q))
    baseline.experiment = "kleidiai"
    optimized.experiment = "kleidiai"

    instance = InstanceSpec(
        name="aws-graviton4-c8g-2xlarge",
        arch="arm64",
        cpu="Arm Neoverse-V2 (Graviton4)",
        vcpus=8,
        usd_per_hour=0.319,  # real c8g.2xlarge on-demand price (us-east-1, 2026-07-05); DATA is synthetic
        notes="real instance price; synthetic perf data",
    )
    return [baseline, optimized], instance

"""Render the before/after report.

Reads `bench/results/*.json` and emits a one-page report in two forms:
  - standalone HTML (`bench/reports/*.html`): self-contained, charts inlined as base64 PNG
  - markdown (`bench/reports/*.md`): diff-friendly, references sibling PNG charts

Leads with the headline (best before/after prefill-TTFT delta, or the long-context prefill
number for a single run), then metric cards, charts, and tables for prefill scaling,
throughput, peak memory, and $/M tokens (via `cost.py` + `instances.yaml`).

matplotlib and jinja2 come from the `[report]` extra and are imported lazily, so importing
this module (and `firstflight report` with no results) works without them.
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
from ..engines.llama_cpp import kernel_tier
from ..profile.performix import Hotspot, ProfileResult
from ..util import bytes_human, reports_dir, results_dir

PALETTE = ["#0B7285", "#E8590C", "#5F3DC4", "#2B8A3E", "#A61E4D", "#1864AB"]


class ReportError(RuntimeError):
    """Something prevented rendering (e.g. the [report] extra isn't installed)."""


# --- loading ------------------------------------------------------------------


def _is_committed_example(path: Path) -> bool:
    """Synthetic example_*.json / profile_example.json shipped in the repo for illustration.

    These must never mix into a real report: they would force the DEMO banner and become the
    headline baseline. The demo path injects synthetic data instead of reading these files.
    """
    return path.name.startswith("example_") or path.name == "profile_example.json"


# Result files that are NOT llama-bench SweepResults (they have their own loaders/sections).
_OTHER_PREFIXES = ("profile_", "ttft_", "throughput_", "autotune_")


def load_results(source: Path | None = None) -> list[SweepResult]:
    """Real SweepResults from a directory (default bench/results), sorted by timestamp.

    Skips committed synthetic examples (see _is_committed_example) and non-sweep result files
    (profile_/ttft_/throughput_/autotune_).
    """
    source = source or results_dir()
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    out: list[SweepResult] = []
    for f in files:
        if _is_committed_example(f) or f.name.startswith(_OTHER_PREFIXES):
            continue
        try:
            out.append(SweepResult.load_json(f))
        except Exception:  # noqa: BLE001 - skip unparseable, keep going
            continue
    out.sort(key=lambda r: r.timestamp)
    return out


def load_ttft_results(source: Path | None = None) -> list:
    """Measured-TTFT results (ttft_*.json), oldest->newest."""
    from ..bench.ttft import TtftResult

    source = source or results_dir()
    if not source.is_dir():
        return []
    out = []
    for f in sorted(source.glob("ttft_*.json")):
        if _is_committed_example(f):
            continue
        try:
            out.append(TtftResult.load_json(f))
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda r: r.timestamp)
    return out


def load_throughput_results(source: Path | None = None) -> list:
    """Concurrency-sweep results (throughput_*.json), oldest->newest."""
    from ..bench.throughput import ThroughputResult

    source = source or results_dir()
    if not source.is_dir():
        return []
    out = []
    for f in sorted(source.glob("throughput_*.json")):
        if _is_committed_example(f):
            continue
        try:
            out.append(ThroughputResult.load_json(f))
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda r: r.timestamp)
    return out


def load_profiles(source: Path | None = None) -> list[ProfileResult]:
    """Performix profile JSONs (profile_*.json), sorted by timestamp."""
    source = source or results_dir()
    if not source.is_dir():
        return []
    out: list[ProfileResult] = []
    for f in sorted(source.glob("profile_*.json")):
        if _is_committed_example(f):
            continue
        try:
            out.append(ProfileResult.load_json(f))
        except Exception:  # noqa: BLE001
            continue
    out.sort(key=lambda p: p.timestamp)
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
    ppl: float | None = None  # perplexity over the fixed corpus (lower = better)


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
    hotspots: list[Hotspot] = field(default_factory=list)
    profile_target: str = ""
    thp: str = ""  # transparent-hugepage mode captured with the results (run evidence)
    ttft_runs: list = field(default_factory=list)  # measured-TTFT (bench/ttft.py) results
    throughput_runs: list = field(default_factory=list)  # concurrency sweeps (bench/throughput.py)
    duplicates_dropped: list = field(default_factory=list)  # older same-label runs superseded
    kernel_evidence: list = field(default_factory=list)  # per-label weight-buffer lines
    system_info: str = ""  # ggml system_info flags from a real run
    cpu_features: str = ""  # /proc/cpuinfo Features line


def _consolidate(results: list[SweepResult]) -> tuple[list[SweepResult], list[str]]:
    """Collapse duplicates by (experiment, label).

    Experiment runs: several results per label are interleaved ROUNDS (`--rounds N`) —
    aggregate to the median per point with the between-round spread as the stddev (the
    honest error bar on a shared runner). Ad-hoc runs (experiment ""): newest wins and the
    older ones are listed as superseded, since a stale ad-hoc rerun may not be the same setup.
    """
    by_key: dict[tuple[str, str], list[SweepResult]] = {}
    for r in results:
        by_key.setdefault((r.experiment, r.label), []).append(r)
    merged: list[SweepResult] = []
    dropped: list[str] = []
    for (exp, label), group in by_key.items():
        if len(group) == 1:
            merged.append(group[0])
        elif exp:
            merged.append(_median_of_rounds(group))
            dropped.append(f"{label}: median of {len(group)} rounds (spread = between-round)")
        else:
            for stale in group[:-1]:
                dropped.append(f"{label} @ {stale.timestamp}")
            merged.append(group[-1])
    return merged, dropped


def _baseline_and_rivals(results: list[SweepResult]) -> tuple[SweepResult, list[SweepResult]]:
    """Pick the headline baseline and the results it may be compared against.

    "baseline" wins; the attribution ladder's "generic" floor is the fallback name. Rivals
    come only from the baseline's own experiment: the global best would attribute a compound
    win (e.g. a KleidiAI build AND a bigger micro-batch) to a single change.
    """
    baseline = next(
        (r for r in results if r.label.lower() == "baseline"),
        next((r for r in results if r.label.lower() == "generic"), results[0]),
    )
    others = [
        r
        for r in results
        if r is not baseline and (not baseline.experiment or r.experiment == baseline.experiment)
    ]
    return baseline, others


def _headline_pair(results: list[SweepResult]):
    """(baseline, best, ctx) for annotation; best is None when no comparison exists."""
    if not results:
        return None, None, None
    baseline, others = _baseline_and_rivals(results)
    base_map = _prefill_map(baseline)
    best, best_ctx = None, None
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

    # Same adjacent-rung rule the headline uses. The charts are the most-shared artifact in
    # the whole report, so a TTFT chart titled "kleidiai vs generic" would put the bundled
    # comparison this project exists to reject onto the one image people screenshot.
    if len(others) >= 2 and best is not None and best_ctx is not None:
        top, rung_below = others[-1], others[-2]
        if best_ctx in _prefill_map(top) and best_ctx in _prefill_map(rung_below):
            baseline, best = rung_below, top
    return baseline, best, best_ctx


def _median_of_rounds(group: list[SweepResult]) -> SweepResult:
    """Collapse interleaved rounds of one config into a single result.

    Per point: median tok/s across rounds; stddev = between-round spread (pstdev), which is
    the error bar that matters on a shared runner. Newest round supplies the metadata.
    """
    import statistics
    from dataclasses import replace

    base = group[-1]
    order: list[tuple] = []
    per_key: dict[tuple, list[BenchPoint]] = {}
    for r in group:
        for p in r.points:
            k = (p.kind, p.n_prompt, p.n_gen)
            if k not in per_key:
                order.append(k)
            per_key.setdefault(k, []).append(p)
    points: list[BenchPoint] = []
    for k in order:
        pts = per_key[k]
        tputs = [p.throughput_tok_s for p in pts]
        med = statistics.median(tputs)
        spread = statistics.pstdev(tputs) if len(tputs) > 1 else pts[-1].throughput_stddev
        ttft = None
        if k[0] == "pp":
            ttft = (k[1] / med) if med > 0 else float("inf")
        points.append(
            replace(
                pts[-1],
                throughput_tok_s=med,
                throughput_stddev=spread,
                ttft_s=ttft,
                samples_ts=[t for p in pts for t in p.samples_ts] or [round(t, 3) for t in tputs],
            )
        )
    peak = max((r.peak_rss_bytes or 0) for r in group) or None
    quality = next((r.quality for r in reversed(group) if r.quality), None)
    return replace(base, points=points, peak_rss_bytes=peak, quality=quality)


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


def _fmt_spread(sd: float | None) -> str:
    """Render a repetition spread without rounding a small one away to '±0'.

    llama-bench's spread across repetitions is routinely far below 1 tok/s: the Q8_0 ladder
    measured 0.02 on 72 tok/s. Printed at zero decimals that becomes '±0', which reads as a
    perfectly reproducible measurement rather than a very tight one, and that is a stronger
    claim than the run can support.
    """
    if sd is None:
        return ""
    if sd >= 0.5:
        return f"±{sd:.0f}"
    if sd >= 0.005:
        return f"±{sd:.2f}"
    return "±0.00" if sd else "±0"


def _quality_ppl(q) -> float | None:
    """Perplexity for a run, or None. Present whenever the ladder ran, probe or no probe."""
    if isinstance(q, dict) and q.get("ppl") is not None:
        return float(q["ppl"])
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


def _fmt_quality_row(r: ResultRow) -> str:
    """Probe counts plus perplexity when measured, e.g. '32/40 · ppl 12.41'."""
    base = _fmt_quality(r.quality_counts, r.quality_acc)
    if r.ppl is not None:
        base = f"{base} · ppl {r.ppl:.2f}" if base != "-" else f"ppl {r.ppl:.2f}"
    return base


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
    profiles: list[ProfileResult] | None = None,
    ttft_results: list | None = None,
    throughput_results: list | None = None,
    title: str = "Arm FirstFlight - Inference Optimization Report",
) -> ReportModel:
    if not results:
        raise ReportError("no results to render")

    # Any synthetic result forces the DEMO banner, even outside --demo (e.g. committed examples).
    demo = demo or any("SYNTHETIC" in (r.host.cpu_info or "") for r in results)

    results, dropped = _consolidate(results)

    price = instance.usd_per_hour
    priced = price > 0

    # Two cost views per row: $/M generated tokens (gen throughput) and $/M prompt tokens
    # (prefill throughput at the row's largest context, which matches the prefill/TTFT headline).
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
                ppl=(
                    float(r.quality["ppl"])
                    if isinstance(r.quality, dict) and r.quality.get("ppl")
                    else None
                ),
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
                # tok/s with the spread across repetitions, so the noise is visible
                "tput": f"{pt.throughput_tok_s:.0f} {_fmt_spread(pt.throughput_stddev)}"
                if pt
                else "-",
            }
        prefill_table.append({"ctx": ctx, "ctx_label": _ctx_label(ctx), "cells": cells})

    # Headline: comparison if a baseline + another result exist, else single-run.
    baseline, others = _baseline_and_rivals(results)
    floor_pct = noise_floor_pct(results)
    headline_main, subs, cards = _headline(baseline, others, rows, instance, priced, floor_pct)

    # Measured-TTFT prompt-cache evidence gets a headline sub-bullet (newest run).
    ttft_runs = list(ttft_results or [])
    if ttft_runs:
        t = ttft_runs[-1]
        subs.append(
            f"measured prompt-cache TTFT: cold {t.cold.prompt_ms:.0f}ms -> warm "
            f"{t.warm.prompt_ms:.0f}ms ({t.reduction_pct:.0f}% less prefill on the warm turn)"
        )

    # The guardrail note must describe what the data SAYS, not what we hoped it would say.
    # Two earlier bugs lived here: perplexity-only runs were treated as having no quality
    # data at all (the accuracy probe is optional; perplexity is not), and both branches
    # asserted the speedup "did not degrade accuracy" without ever consulting the numbers.
    # On the Q8_0 ladder that assertion was simply false.
    ppls = [p for p in (_quality_ppl(r.quality) for r in results) if p]
    has_quality = any(_quality_acc(r.quality) is not None for r in results) or bool(ppls)
    if not has_quality:
        quality = (
            "Run `firstflight experiment` to add the quality columns (perplexity per rung, "
            "plus the optional exact-match probe)."
        )
    else:
        quality = (
            "Quality = a regression guardrail, not a leaderboard: perplexity per rung, plus a "
            "small exact-match probe via llama-completion where it was run."
        )
        # Flag drift rather than leaving the reader to diff the column by eye. Kernels that
        # only reorder arithmetic should land on the same perplexity; a rung that doesn't is
        # buying its speed with numerics, which is exactly what this column is for.
        if len(ppls) > 1 and min(ppls) > 0:
            drift = (max(ppls) - min(ppls)) / min(ppls) * 100.0
            quality += (
                f" Perplexity spread across rungs: {drift:.2f}% "
                f"({min(ppls):.4f} to {max(ppls):.4f})"
                + (
                    " - the fastest rung is NOT output-neutral; treat the speedup as a "
                    "trade, not a free win."
                    if drift >= 0.5
                    else " - consistent with the kernels being output-neutral."
                )
            )

    # Newest non-skipped Performix profile, if any.
    hotspots: list[Hotspot] = []
    profile_target = ""
    for p in reversed(profiles or []):
        if not p.skipped and p.hotspots:
            hotspots = p.hotspots
            profile_target = p.target
            break

    # Kernel evidence: which weight buffer each config actually loaded, plus the ggml
    # system_info / cpuinfo Features lines (identical across rows on one host; take newest).
    kernel_evidence = []
    for r in results:
        if not getattr(r.host, "kernel_buffer", ""):
            continue
        bits = [r.host.kernel_buffer]
        # Derive the tier from THIS run's own system_info instead of trusting the stored
        # field. Results recorded before the tier became build-derived carry the host's
        # capability, which labels an armv8-a floor build "I8MM" on an I8MM machine - in the
        # one section that claims to show what actually ran, not what the box can do.
        own_info = getattr(r.host, "system_info", "") or ""
        tier = kernel_tier(own_info) if own_info else getattr(r.host, "kernel_tier", "")
        if tier:
            bits.append(f"{tier}-tier kernels")
        repack = getattr(r.host, "repack", None)
        if repack is not None:
            bits.append(f"ggml repack {'on' if repack else 'off'}")
        kernel_evidence.append(f"{r.label}: " + ", ".join(bits))
    system_info = next(
        (r.host.system_info for r in reversed(results) if getattr(r.host, "system_info", "")), ""
    )
    cpu_features = next(
        (r.host.cpu_features for r in reversed(results) if getattr(r.host, "cpu_features", "")),
        "",
    )

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
        ttft_runs=ttft_runs,
        throughput_runs=list(throughput_results or []),
        duplicates_dropped=dropped,
        kernel_evidence=kernel_evidence,
        system_info=system_info,
        cpu_features=cpu_features,
    )


def _cost_for(rows: list[ResultRow], label: str) -> CostResult | None:
    return next((r.cost for r in rows if r.label == label), None)


def _quality_for(rows: list[ResultRow], label: str):
    """(accuracy, (n_correct, n_total)|None) for a labelled row, or (None, None)."""
    for r in rows:
        if r.label == label:
            return r.quality_acc, r.quality_counts
    return None, None


def noise_floor_pct(results: list[SweepResult]) -> float | None:
    """Measured same-build spread, as a percentage, from the noise-floor experiment.

    Two identical configs run under different labels: any apparent 'speedup' between them is
    the runner talking, not the treatment. Returns None when that experiment wasn't run.
    """
    ctrl = [r for r in results if r.experiment == "noise-floor"]
    if len(ctrl) < 2:
        return None
    maps = [_prefill_map(r) for r in ctrl]
    shared = sorted(set.intersection(*(set(m) for m in maps)))
    spreads = []
    for ctx in shared:
        vals = [m[ctx].throughput_tok_s for m in maps if m[ctx].throughput_tok_s > 0]
        if len(vals) >= 2 and min(vals) > 0:
            spreads.append((max(vals) - min(vals)) / min(vals) * 100.0)
    return max(spreads) if spreads else None


def _speed_phrase(speed: float) -> str:
    """ "3.60x faster" / "1.10x slower" - a bare multiplier drops the direction.

    Below 1.0 the honest form inverts the ratio: "0.91x faster" reads as a win to anyone
    skimming a chart title, when it is really a 10% regression. Chart titles and cards are
    exactly where a number travels furthest from its context, so the word ships with it.
    """
    if speed and speed < 1.0:
        return f"{1 / speed:.2f}x slower"
    return f"{speed:.2f}x faster"


def _speedup_span(base_map: dict, other_map: dict):
    """((ctx, min_speedup), (ctx, max_speedup)) across shared contexts, or None.

    A ladder's delta can invert with context: once the matmul is fast, quadratic attention
    takes over, so a rung that is 3.6x ahead at 1k can fall behind at 8k. Reporting only the
    largest context would hide the first fact; reporting only the best would hide the second.
    """
    pairs = [
        (ctx, other_map[ctx].throughput_tok_s / base_map[ctx].throughput_tok_s)
        for ctx in sorted(set(base_map) & set(other_map))
        if base_map[ctx].throughput_tok_s
    ]
    if len(pairs) < 2:
        return None
    return min(pairs, key=lambda p: p[1]), max(pairs, key=lambda p: p[1])


def noise_verdict(speed: float, floor_pct: float | None) -> str:
    """ "faster" | "slower" | "within noise" | "unknown floor".

    Three outcomes, not two. A 0.91x delta against a 0.3% floor is a 9% REGRESSION, thirty
    times the floor; calling that "within noise" (as a plain not-a-win test does) hides a
    real result. Only a delta whose magnitude sits under the floor is genuinely unresolvable.
    """
    if floor_pct is None:
        return "unknown floor"
    change_pct = (speed - 1.0) * 100.0
    if abs(change_pct) <= floor_pct:
        return "within noise"
    return "faster" if change_pct > 0 else "slower"


def _headline(baseline, others, rows, instance, priced, floor_pct=None):
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

    # On a multi-rung ladder the headline compares ADJACENT rungs. Running the top rung
    # straight against the unaccelerated floor folds every mechanism below it into one
    # number, which is the exact attribution error this harness exists to catch: KleidiAI
    # vs generic hands KleidiAI credit for ggml's repack as well. The cumulative delta is
    # still reported, as a sub-bullet, where it cannot be read as one mechanism's doing.
    cumulative = None
    if len(others) >= 2 and best is not None and best_ctx is not None:
        top, rung_below = others[-1], others[-2]
        top_map, below_map = _prefill_map(top), _prefill_map(rung_below)
        if (
            best_ctx in top_map
            and below_map.get(best_ctx, None)
            and below_map[best_ctx].throughput_tok_s
        ):
            cumulative = (baseline.label, base_map)
            baseline, base_map, best = rung_below, below_map, top

    if best is not None and best_ctx is not None:
        b = base_map[best_ctx]
        o = _prefill_map(best)[best_ctx]
        speed = (o.throughput_tok_s / b.throughput_tok_s) if b.throughput_tok_s else 0.0
        verdict = noise_verdict(speed, floor_pct)
        subs = [
            f"TTFT {_fmt_ttft(b.ttft_s)}s -> {_fmt_ttft(o.ttft_s)}s at {best_ctx:,} tokens "
            f"({best.label} vs {baseline.label})",
            f"prefill {b.throughput_tok_s:.0f} -> {o.throughput_tok_s:.0f} tok/s",
            f"generation {_gen_tput(baseline):.0f} -> {_gen_tput(best):.0f} tok/s",
        ]
        # The delta at one context is not the result: report how it moves across the sweep,
        # so a single multiplier can never stand in for a curve that crosses over.
        span = _speedup_span(base_map, _prefill_map(best))
        if span:
            (lo_ctx, lo), (hi_ctx, hi) = span
            if abs(hi - lo) > 0.02:
                subs.append(
                    f"context-dependent: {hi:.2f}x at {_ctx_label(hi_ctx)} -> {lo:.2f}x at "
                    f"{_ctx_label(lo_ctx)} (the delta is a curve, not a number)"
                )
        if cumulative:
            floor_label, floor_map = cumulative
            fp = floor_map.get(best_ctx)
            if fp and fp.throughput_tok_s:
                cum = o.throughput_tok_s / fp.throughput_tok_s
                subs.append(
                    f"cumulative vs {floor_label}: {cum:.2f}x at {_ctx_label(best_ctx)} "
                    f"(every rung combined, not {best.label} alone)"
                )
        if floor_pct is not None:
            phrase = {
                "faster": "clears it",
                "slower": "is a REGRESSION well outside it",
                "within noise": "does NOT clear it - not claimed either way",
            }[verdict]
            subs.append(
                f"measured noise floor (same build, twice): {floor_pct:.1f}% - this delta {phrase}"
            )
        cards = [
            (
                _speed_phrase(speed) if verdict != "within noise" else "within noise",
                f"prefill @ {_ctx_label(best_ctx)} vs {baseline.label}",
            ),
            (f"{_fmt_ttft(o.ttft_s)}s", f"TTFT @ {_ctx_label(best_ctx)} ({best.label})"),
            (f"{o.throughput_tok_s:.0f}", "prefill tok/s"),
        ]
        if priced:
            # Prompt cost from the same prefill points as the headline. Generation cost is
            # secondary.
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
            cards.append(("$0.00", "$/M prompt tok (free runner)"))
        bq, bqc = _quality_for(rows, baseline.label)
        oq, oqc = _quality_for(rows, best.label)
        if bq is not None and oq is not None:
            if bqc and oqc:
                # tolerate one probe item; the instrument isn't finer than that
                held = "held" if oqc[0] >= bqc[0] - 1 else "down"
                subs.append(f"quality {_fmt_quality(bqc, bq)} -> {_fmt_quality(oqc, oq)} ({held})")
                cards.append((_fmt_quality(oqc, oq), "quality (probe)"))
            else:
                held = "held" if oq >= bq - 0.02 else "down"
                subs.append(f"quality {bq:.0%} -> {oq:.0%} ({held})")
                cards.append((f"{oq:.0%}", "quality (probe)"))
        # When the delta crosses 1.0 somewhere in the sweep, no single context is the result.
        # Reporting the widest context alone would headline the crossover and read as a plain
        # regression; reporting the best would read as a plain win. Both are the same curve,
        # so the headline carries both ends and names the shape.
        if span:
            (lo_ctx, lo), (hi_ctx, hi) = span
            if lo < 1.0 < hi:
                return (
                    f"{hi:.2f}x faster prefill at {_ctx_label(hi_ctx)}, inverting to "
                    f"{lo:.2f}x at {_ctx_label(lo_ctx)} ({best.label} vs {baseline.label}): "
                    f"the advantage is context-dependent",
                    subs,
                    cards,
                )
        if verdict == "within noise":
            return (
                f"No claimed win: the delta at {_ctx_label(best_ctx)} ({speed:.2f}x) sits inside "
                f"the measured {floor_pct:.1f}% noise floor",
                subs,
                cards,
            )
        if verdict == "slower":
            return (
                f"{1 / speed:.2f}x SLOWER prefill at {best_ctx:,}-token context "
                f"({best.label} vs {baseline.label}), well outside the {floor_pct:.1f}% noise floor",
                subs,
                cards,
            )
        # Reached when the floor is unknown, so the direction still has to be honest here:
        # an unmeasured floor is not licence to call a 0.91x delta a speedup.
        return f"{_speed_phrase(speed)} prefill at {best_ctx:,}-token context", subs, cards

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
        cards.append(("$0.00", "$/M prompt tok (free runner)"))
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


def build_charts(
    results: list[SweepResult],
    instance: InstanceSpec,
    ttft_results: list | None = None,
    throughput_results: list | None = None,
    demo: bool = False,
) -> list[Chart]:
    _require_report_deps()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results, _ = _consolidate(results)  # same view of rounds/duplicates as the tables

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

    def stamp_synthetic(ax):
        # Baked into the pixels: this image gets copied out of context (README, galleries),
        # where the report's DEMO banner can't follow it.
        if demo:
            ax.text(
                0.5,
                0.5,
                "SYNTHETIC DEMO DATA",
                transform=ax.transAxes,
                fontsize=34,
                color="#fa5252",
                alpha=0.22,
                ha="center",
                va="center",
                rotation=20,
                fontweight="bold",
                zorder=0,
            )

    def line_chart(name, title, ylabel, value_fn):
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for i, r in enumerate(results):
            pts = sorted(r.prefill_points, key=lambda p: p.n_prompt)
            # Skip missing/non-finite values; plotting them as 0.0 draws a fake dip to zero.
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
        stamp_synthetic(ax)
        fig.tight_layout()
        png = _fig_png(fig)
        plt.close(fig)
        charts.append(Chart(name, title, png))

    # Hero chart: annotated before/after TTFT with the saved time shaded and the speedup
    # called out. Falls back to the plain line chart when there's nothing to compare.
    hero_done = False
    pair_base, pair_best, pair_ctx = _headline_pair(results)
    if pair_best is not None and pair_ctx is not None:
        bmap, omap = _prefill_map(pair_base), _prefill_map(pair_best)
        shared = sorted(
            c
            for c in set(bmap) & set(omap)
            if bmap[c].ttft_s
            and omap[c].ttft_s
            and math.isfinite(bmap[c].ttft_s)
            and math.isfinite(omap[c].ttft_s)
        )
        if len(shared) >= 2:
            xs = shared
            ys_b = [bmap[c].ttft_s for c in xs]
            ys_o = [omap[c].ttft_s for c in xs]
            speed = (
                omap[pair_ctx].throughput_tok_s / bmap[pair_ctx].throughput_tok_s
                if bmap[pair_ctx].throughput_tok_s
                else 0.0
            )
            saved = (bmap[pair_ctx].ttft_s or 0) - (omap[pair_ctx].ttft_s or 0)
            fig, ax = plt.subplots(figsize=(11, 6.2))
            ax.plot(xs, ys_b, marker="o", linewidth=2.6, color=PALETTE[0], label=pair_base.label)
            ax.plot(xs, ys_o, marker="o", linewidth=2.6, color=PALETTE[1], label=pair_best.label)
            ax.fill_between(xs, ys_b, ys_o, color=PALETTE[1], alpha=0.12)
            ax.annotate(
                f"{_speed_phrase(speed)}\n{abs(saved):.1f}s "
                f"{'saved' if saved >= 0 else 'added'} per prompt",
                xy=(pair_ctx, (ys_b[-1] + ys_o[-1]) / 2),
                xytext=(0.55, 0.55),
                textcoords="axes fraction",
                fontsize=15,
                fontweight="bold",
                color="#1971c2",
                arrowprops={"arrowstyle": "->", "color": "#1971c2"},
            )
            ax.annotate(
                f"{ys_b[-1]:.1f}s",
                xy=(xs[-1], ys_b[-1]),
                xytext=(-8, 8),
                textcoords="offset points",
                ha="right",
                fontweight="bold",
                color=PALETTE[0],
            )
            ax.annotate(
                f"{ys_o[-1]:.1f}s",
                xy=(xs[-1], ys_o[-1]),
                xytext=(-8, -14),
                textcoords="offset points",
                ha="right",
                fontweight="bold",
                color=PALETTE[1],
            )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("context length (prompt tokens)")
            ax.set_ylabel("time-to-first-token (s)")
            title = (
                f"Prefill TTFT: {_speed_phrase(speed)} at {_ctx_label(pair_ctx)} context "
                f"({pair_best.label} vs {pair_base.label})"
            )
            ax.set_title(title)
            if all_ctx:
                ax.set_xticks(all_ctx)
                ax.set_xticklabels([_ctx_label(x) for x in all_ctx])
                ax.minorticks_off()
            ax.legend(loc="upper left")
            stamp_synthetic(ax)
            fig.text(
                0.01,
                0.005,
                "TTFT derived as prompt_tokens / prefill throughput; excludes tokenization "
                "and model load. See docs/METHODOLOGY.md.",
                fontsize=8,
                color="#868e96",
            )
            fig.tight_layout()
            png = _fig_png(fig)
            plt.close(fig)
            charts.append(Chart("prefill-ttft", title, png))
            hero_done = True
    if not hero_done:
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

    # Which lever mattered: best prefill speedup per experiment axis, sorted, with the
    # no-change line at 1.0. The noise-floor bar sits at ~1.0 by construction and is the
    # yardstick every other bar has to clear. Negative results plot honestly left of 1.0.
    by_exp: dict[str, list[SweepResult]] = {}
    for r in results:
        if r.experiment:
            by_exp.setdefault(r.experiment, []).append(r)
    levers: list[tuple[str, float]] = []
    for exp, group in by_exp.items():
        if len(group) < 2:
            continue
        base = group[0]  # first-run config = that experiment's own "before"
        bmap = _prefill_map(base)
        best_ratio, best_label = None, ""
        for cand in group[1:]:
            cmap = _prefill_map(cand)
            shared = sorted(set(bmap) & set(cmap))
            if not shared:
                continue
            ctx = shared[-1]
            if bmap[ctx].throughput_tok_s > 0:
                ratio = cmap[ctx].throughput_tok_s / bmap[ctx].throughput_tok_s
                if best_ratio is None or ratio > best_ratio:
                    best_ratio, best_label = ratio, cand.label
        if best_ratio is not None:
            levers.append((f"{exp}  ({best_label} / {base.label})", best_ratio))
    if levers:
        levers.sort(key=lambda t: t[1])
        names = [n for n, _ in levers]
        vals = [v for _, v in levers]
        fig, ax = plt.subplots(figsize=(8.8, 1.0 + 0.55 * len(levers)))
        bars = ax.barh(names, vals, color=[PALETTE[1] if v >= 1.0 else "#e03131" for v in vals])
        ax.axvline(1.0, color="#495057", linewidth=1.2, linestyle="--")
        ax.bar_label(bars, fmt="%.2fx", padding=3)
        ax.set_xlabel("prefill speedup at largest shared context (1.0 = no change)")
        ax.set_title("Which lever mattered - best speedup per experiment axis")
        fig.tight_layout()
        png = _fig_png(fig)
        plt.close(fig)
        charts.append(Chart("levers", "Which lever mattered", png))

    # Prompt-cache evidence: cold vs warm server-measured prefill, one glance.
    if ttft_results:
        t = ttft_results[-1]
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        bars = ax.bar(
            ["cold (full prefix)", "warm (cache hit)"],
            [t.cold.prompt_ms, t.warm.prompt_ms],
            color=[PALETTE[0], PALETTE[1]],
        )
        ax.bar_label(bars, fmt="%.0f ms", padding=3)
        ax.set_ylabel("server-measured prefill time (ms)")
        title = f"Prompt cache: {t.reduction_pct:.0f}% of prefill skipped on the warm turn"
        ax.set_title(title)
        fig.tight_layout()
        png = _fig_png(fig)
        plt.close(fig)
        charts.append(Chart("prompt-cache", title, png))

    # Concurrency: aggregate tok/s as parallel requests scale.
    if throughput_results:
        run = throughput_results[-1]
        pts = sorted(run.points, key=lambda p: p.parallel)
        if pts:
            xs = [p.parallel for p in pts]
            fig, ax = plt.subplots(figsize=(7.2, 4.0))
            for label, ys, color in (
                ("total", [p.speed for p in pts], PALETTE[0]),
                ("prefill", [p.speed_pp for p in pts], PALETTE[1]),
                ("generation", [p.speed_tg for p in pts], PALETTE[2 % len(PALETTE)]),
            ):
                if any(y > 0 for y in ys):
                    ax.plot(xs, ys, marker="o", linewidth=2.2, color=color, label=label)
            ax.set_xticks(xs)
            ax.set_xlabel("parallel requests")
            ax.set_ylabel("aggregate tokens/sec")
            title = f"Throughput vs parallel requests ({run.model}:{run.variant})"
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            png = _fig_png(fig)
            plt.close(fig)
            charts.append(Chart("concurrency", title, png))

    # Cost bars only when priced. Prompt-token cost (prefill throughput at each result's largest
    # context), matching the prefill/TTFT headline. Non-finite costs are skipped, not plotted.
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
            f"{_fmt_quality_row(r)} |"
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

    if model.ttft_runs:
        lines.append("## Measured TTFT - prompt cache, cold vs warm\n")
        lines.append(
            "_llama-server's own `timings` (includes tokenization). Cold = full shared prefix "
            "prefilled; warm = same prefix, new question, KV cache re-used (`cache_prompt` + "
            "`--cache-reuse`)._\n"
        )
        lines.append(
            "| model:variant | cache-reuse | cold tokens | cold ms | warm tokens | warm ms | prefill saved |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for t in model.ttft_runs:
            lines.append(
                f"| {t.model}:{t.variant} | {t.cache_reuse} | {t.cold.prompt_n} | "
                f"{t.cold.prompt_ms:.0f} | {t.warm.prompt_n} | {t.warm.prompt_ms:.0f} | "
                f"{t.reduction_pct:.0f}% |"
            )
        lines.append("")

    if model.throughput_runs:
        lines.append("## Throughput vs parallel requests (llama-batched-bench)\n")
        lines.append("_Aggregate tokens/sec across all concurrent sequences._\n")
        for run in model.throughput_runs:
            lines.append(
                f"**{run.model}:{run.variant}** - {run.npp} prompt + {run.ntg} gen tokens per request\n"
            )
            lines.append("| parallel | prefill tok/s | gen tok/s | total tok/s |")
            lines.append("|---:|---:|---:|---:|")
            for p in sorted(run.points, key=lambda x: x.parallel):
                lines.append(
                    f"| {p.parallel} | {p.speed_pp:.1f} | {p.speed_tg:.1f} | {p.speed:.1f} |"
                )
            lines.append("")

    if model.kernel_evidence or model.system_info or model.cpu_features:
        lines.append("## Kernel evidence (from real runs, not build flags)\n")
        lines.append("_Which CPU weight buffer each config actually loaded, per the load log._\n")
        for ev in model.kernel_evidence:
            lines.append(f"- weight buffer - {ev}")
        if model.system_info:
            lines.append(f"- ggml system_info: `{model.system_info}`")
        if model.cpu_features:
            lines.append(f"- cpuinfo Features: `{model.cpu_features}`")
        lines.append("")

    if model.duplicates_dropped:
        lines.append(
            f"_Note: {len(model.duplicates_dropped)} run group(s) consolidated: "
            f"{', '.join(model.duplicates_dropped)}._\n"
        )

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
</table>{% if not m.priced %}<div class="note">Costs are $0.00: this run executed on a free CI runner (github-arm-runner). Select a paid instance in configs/instances.yaml to price the same throughput on rented hardware.</div>{% endif %}</section>

<section><h2>Prefill scaling</h2>
<div class="note">TTFT is derived as prompt_tokens &divide; prefill throughput; excludes tokenization and model-load time. &plusmn; is the spread across repetitions.</div>
<table>
<thead><tr><th>context</th>{% for l in m.labels %}<th>{{ l }} TTFT (s)</th><th>{{ l }} tok/s</th>{% endfor %}</tr></thead>
<tbody>{% for row in m.prefill_table %}<tr><td>{{ "{:,}".format(row.ctx) }}</td>{% for l in m.labels %}<td>{{ row.cells[l].ttft }}</td><td>{{ row.cells[l].tput }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></section>

{% if m.ttft_runs %}<section><h2>Measured TTFT &mdash; prompt cache, cold vs warm</h2>
<div class="note">llama-server's own <code>timings</code> (includes tokenization). Cold = full shared prefix prefilled; warm = same prefix, new question, KV cache re-used.</div>
<table><thead><tr><th>model:variant</th><th>cache-reuse</th><th>cold tokens</th><th>cold ms</th><th>warm tokens</th><th>warm ms</th><th>prefill saved</th></tr></thead>
<tbody>{% for t in m.ttft_runs %}<tr><td>{{ t.model }}:{{ t.variant }}</td><td>{{ t.cache_reuse }}</td><td>{{ t.cold.prompt_n }}</td><td>{{ "%.0f"|format(t.cold.prompt_ms) }}</td><td>{{ t.warm.prompt_n }}</td><td>{{ "%.0f"|format(t.warm.prompt_ms) }}</td><td>{{ "%.0f"|format(t.reduction_pct) }}%</td></tr>{% endfor %}</tbody>
</table></section>{% endif %}

{% if m.throughput_runs %}<section><h2>Throughput vs parallel requests</h2>
<div class="note">Aggregate tokens/sec across all concurrent sequences (llama-batched-bench).</div>
{% for run in m.throughput_runs %}<div class="note"><b>{{ run.model }}:{{ run.variant }}</b> &mdash; {{ run.npp }} prompt + {{ run.ntg }} gen tokens per request</div>
<table><thead><tr><th>parallel</th><th>prefill tok/s</th><th>gen tok/s</th><th>total tok/s</th></tr></thead>
<tbody>{% for p in run.points|sort(attribute="parallel") %}<tr><td>{{ p.parallel }}</td><td>{{ "%.1f"|format(p.speed_pp) }}</td><td>{{ "%.1f"|format(p.speed_tg) }}</td><td>{{ "%.1f"|format(p.speed) }}</td></tr>{% endfor %}</tbody>
</table>{% endfor %}</section>{% endif %}

{% if m.kernel_evidence or m.system_info %}<h2>Kernel evidence</h2><div class="note">{% for ev in m.kernel_evidence %}weight buffer &mdash; {{ ev }}<br>{% endfor %}{% if m.system_info %}<code>{{ m.system_info }}</code><br>{% endif %}{% if m.cpu_features %}<code>Features: {{ m.cpu_features }}</code>{% endif %}</div>{% endif %}
{% if m.duplicates_dropped %}<div class="note">Note: {{ m.duplicates_dropped|length }} run group(s) consolidated ({{ m.duplicates_dropped|join(", ") }}).</div>{% endif %}

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
        r.quality_human = _fmt_quality_row(r)  # type: ignore[attr-defined]
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
    """Render the report. Output paths, or None when there are no results."""
    instances = load_instances()

    if demo:
        results, instance = synthetic_results()
        ttft_res = [synthetic_ttft()]
        tp_res = [synthetic_throughput()]
        profiles = [synthetic_profile()]
    else:
        results = results if results is not None else load_results(results_source)
        instance = instances.get(instance_name)
        ttft_res = load_ttft_results(results_source)
        tp_res = load_throughput_results(results_source)
        profiles = load_profiles(results_source)

    if not results:
        return None

    model = build_report_model(
        results,
        instance,
        demo=demo,
        profiles=profiles,
        ttft_results=ttft_res,
        throughput_results=tp_res,
    )
    charts = build_charts(  # raises ReportError if deps missing
        results,
        instance,
        ttft_results=ttft_res,
        throughput_results=tp_res,
        demo=model.demo,  # model.demo also covers committed synthetic results loaded off disk
    )

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
    """Plausible-but-fake Arm Graviton4 numbers for a DEMO-labelled report.

    NOT measured data. `firstflight report --demo` uses this so the layout can be seen (and the
    demo video recorded) without an Arm box. Real reports come from `firstflight bench`.
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
        usd_per_hour=0.319,  # real c8g.2xlarge on-demand (us-east-1, 2026-07-05); perf is synthetic
        notes="real instance price; synthetic perf data",
    )
    return [baseline, optimized], instance


def synthetic_ttft():
    """Fake measured-TTFT prompt-cache pair for the DEMO report. NOT measured."""
    from ..bench.ttft import Timings, TtftResult

    return TtftResult(
        timestamp="2026-06-26T12:00:00+00:00",
        model="qwen2.5-0.5b-instruct [SYNTHETIC]",
        variant="q4_0",
        prefix_target_tokens=2048,
        cache_reuse=256,
        threads=8,
        cold=Timings(prompt_n=2071, prompt_ms=1620.0, predicted_n=16, predicted_ms=210.0),
        warm=Timings(prompt_n=23, prompt_ms=42.0, predicted_n=16, predicted_ms=205.0),
    )


def synthetic_throughput():
    """Fake concurrency sweep for the DEMO report. NOT measured."""
    from ..bench.throughput import ThroughputPoint, ThroughputResult

    pts = [
        ThroughputPoint(1, 1024, 32, 1450.0, 55.0, 590.0),
        ThroughputPoint(2, 1024, 32, 2500.0, 92.0, 1010.0),
        ThroughputPoint(4, 1024, 32, 3900.0, 138.0, 1580.0),
        ThroughputPoint(8, 1024, 32, 4600.0, 161.0, 1890.0),
    ]
    return ThroughputResult(
        timestamp="2026-06-26T12:00:00+00:00",
        model="qwen2.5-0.5b-instruct [SYNTHETIC]",
        variant="q4_0",
        npp=1024,
        ntg=32,
        threads=8,
        points=pts,
    )


def synthetic_profile() -> ProfileResult:
    """Fake Arm Performix hotspots for the DEMO report. NOT measured."""
    hotspots = [
        Hotspot("kai_matmul_clamp_f32_qai8dxp_qsi4c32p", 41.7, "libggml-cpu.so"),
        Hotspot("ggml_compute_forward_mul_mat", 18.3, "libggml-cpu.so"),
        Hotspot("ggml_vec_dot_q8_0_q8_0", 9.6, "libggml-cpu.so"),
        Hotspot("ggml_compute_forward_soft_max", 5.1, "libggml-cpu.so"),
        Hotspot("ggml_compute_forward_rope", 4.2, "libggml-cpu.so"),
        Hotspot("memcpy", 3.0, "libc.so.6"),
    ]
    return ProfileResult(
        skipped=False,
        timestamp="2026-06-26T12:00:00+00:00",
        target="llama-bench -m qwen2.5-0.5b-instruct-q4_0.gguf -p 2048 -n 0 -r 1 [SYNTHETIC]",
        hotspots=hotspots,
    )

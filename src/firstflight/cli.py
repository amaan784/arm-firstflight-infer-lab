"""`firstflight` command-line entrypoint.

Subcommands: setup-engine, info, smoke, download, run, bench, ttft, throughput, profile,
experiment, perplexity, report, autotune. `autotune` is opt-in (--enable). Everything else
no-ops off Arm or without a llama.cpp build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.table import Table

from . import __version__
from .util import console, is_arm_linux, platform_tag


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="firstflight")
def main() -> None:
    """Arm FirstFlight — CPU LLM inference optimization harness (prefill/TTFT focus)."""


@main.command()
def info() -> None:
    """Print environment + config summary."""
    from .config import load_instances, load_models, load_workloads
    from .engines import llama_cpp
    from .profile.performix import PerformixProfiler

    table = Table(title="firstflight environment", show_header=False, title_style="bold")
    table.add_column("k", style="cyan", no_wrap=True)
    table.add_column("v")

    table.add_row("version", __version__)
    table.add_row("platform", platform_tag())
    table.add_row("arm linux", "yes" if is_arm_linux() else "no (Arm-only tools will no-op)")
    try:  # ISA evidence up front (Linux): dotprod/i8mm/sve are the kernels' entry ticket
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            text = cpuinfo.read_text(encoding="utf-8", errors="replace")
            feats = next(
                (
                    ln.split(":", 1)[1].strip()
                    for ln in text.splitlines()
                    if ln.lower().startswith("features")
                ),
                "",
            )
            if feats:
                table.add_row("cpu features", feats[:160])
    except OSError:
        pass

    cli = llama_cpp.find_binary("cli")
    bench = llama_cpp.find_binary("bench")
    table.add_row("llama-cli", str(cli) if cli else "[yellow]not found[/]")
    table.add_row("llama-bench", str(bench) if bench else "[yellow]not found[/]")
    table.add_row("performix (apx)", "available" if PerformixProfiler().available() else "n/a")

    try:
        models = load_models()
        instances = load_instances()
        workloads = load_workloads()
        table.add_row("smoke model", f"{models.default_smoke}:{models.default_smoke_variant}")
        table.add_row("default instance", instances.default_instance)
        table.add_row("default workload", workloads.default_workload)
    except Exception as exc:  # bad config shouldn't crash `info`
        table.add_row("config", f"[red]error: {exc}[/]")

    console.print(table)


@main.command()
@click.option("--no-download", is_flag=True, help="Skip the model download step.")
@click.option("-n", "--n-predict", default=24, show_default=True, help="Tokens to generate.")
@click.option(
    "--timeout", default=90.0, show_default=True, help="Seconds before the run is aborted."
)
def smoke(no_download: bool, n_predict: int, timeout: float) -> None:
    """Download the tiny model + run llama.cpp once (proves the pipeline on any machine)."""
    from .runner import smoke as run_smoke

    sys.exit(run_smoke(download=not no_download, n_predict=n_predict, timeout=timeout))


@main.command()
@click.option("--force", is_flag=True, help="Re-download even if cached.")
@click.option("--model", "model_id", default=None, help="Model id (default: smoke model).")
@click.option("--variant", default=None, help="Quant variant (default: model's default).")
def download(force: bool, model_id: str | None, variant: str | None) -> None:
    """Download a model's GGUF (default: the smoke model). Resumable; retries on throttling."""
    from .config import load_models
    from .download import DownloadError, ensure_model
    from .util import safe

    models = load_models()
    if model_id is None:
        spec, mv = models.smoke()
        if variant:
            mv = spec.variant(variant)
    else:
        spec = models.get(model_id)
        mv = spec.variant(variant) if variant else next(iter(spec.variants.values()))

    try:
        path = ensure_model(spec, mv, force=force)
    except DownloadError as exc:
        console.print(f"[red]download failed:[/] {safe(str(exc))}")
        sys.exit(1)
    console.print(f"[green]OK[/] {path}")


@main.command("setup-engine")
@click.option("--tag", default=None, help="llama.cpp release tag (default: the pinned tag).")
@click.option(
    "--dest", type=click.Path(path_type=Path), default=None, help="Install dir (default: ./engine)."
)
@click.option("--force", is_flag=True, help="Re-download even if an engine is present.")
def setup_engine(tag, dest, force) -> None:
    """Download the prebuilt llama.cpp for THIS platform into ./engine (no compiler needed).

    After this, `firstflight smoke` / `bench` run real inference on any machine —
    Windows/Linux/macOS, x64/arm64.
    """
    from .engines import fetch
    from .util import safe

    try:
        fetch.setup_engine(tag=tag or fetch.DEFAULT_TAG, dest=dest, force=force)
    except fetch.EngineFetchError as exc:
        console.print(f"[red]setup-engine failed:[/] {safe(str(exc))}")
        sys.exit(1)


_MODEL_OPTS = [
    click.option(
        "--model",
        "model_id",
        default=None,
        help="Model id from models.yaml (default: smoke model).",
    ),
    click.option("--variant", default=None, help="Quant variant (default: model's default)."),
    click.option(
        "--threads", type=int, default=None, help="Thread count (default: llama-bench auto)."
    ),
    click.option("--no-download", is_flag=True, help="Require the model cached; don't download."),
    click.option("--dry-run", is_flag=True, help="Print the llama-bench command and exit."),
]


def _model_opts(fn):
    for opt in reversed(_MODEL_OPTS):
        fn = opt(fn)
    return fn


@main.command()
@click.option(
    "--prompt-len", type=int, default=2048, show_default=True, help="Single prefill context length."
)
@click.option("--gen", "n_gen", type=int, default=32, show_default=True, help="Generation tokens.")
@click.option("--label", default="run", show_default=True, help="Result label.")
@_model_opts
def run(prompt_len, n_gen, label, model_id, variant, threads, no_download, dry_run) -> None:
    """Run a single (context-length, gen) point — quick iteration."""
    from .runner import bench as run_bench

    sys.exit(
        run_bench(
            model_id=model_id,
            variant=variant,
            workload="quick-smoke",
            prompt_lengths=[prompt_len],
            threads=threads,
            n_gen=n_gen,
            label=label,
            download=not no_download,
            dry_run=dry_run,
        )
    )


@main.command()
@click.option(
    "--workload", default=None, help="Workload from workloads.yaml (default: its default)."
)
@click.option(
    "--repetitions", type=int, default=None, help="Repeats per point (default: workload)."
)
@click.option(
    "--gen", "n_gen", type=int, default=None, help="Generation tokens for the tg measurement."
)
@click.option(
    "--label", default="baseline", show_default=True, help="Result label (for before/after)."
)
@_model_opts
def bench(
    workload, repetitions, n_gen, label, model_id, variant, threads, no_download, dry_run
) -> None:
    """Prefill/TTFT + generation throughput sweep across context lengths -> bench/results/*.json."""
    from .runner import bench as run_bench

    sys.exit(
        run_bench(
            model_id=model_id,
            variant=variant,
            workload=workload,
            threads=threads,
            repetitions=repetitions,
            n_gen=n_gen,
            label=label,
            download=not no_download,
            dry_run=dry_run,
        )
    )


@main.command()
@click.option(
    "--prefix-tokens",
    type=int,
    default=2048,
    show_default=True,
    help="Approx. length of the shared prefix (the server reports exact counts).",
)
@click.option(
    "--cache-reuse",
    type=int,
    default=256,
    show_default=True,
    help="llama-server --cache-reuse chunk size (KV-shift reuse; 0 disables).",
)
@click.option("--model", "model_id", default=None, help="Model id (default: smoke model).")
@click.option("--variant", default=None, help="Quant variant (default: model's default).")
@click.option("--threads", type=int, default=None, help="Thread count (default: auto).")
@click.option("--port", type=int, default=None, help="Local server port (default: auto ephemeral).")
@click.option(
    "--mlock",
    is_flag=True,
    help="Keep model resident in RAM (steady demo TTFT; needs memlock limit).",
)
@click.option("--no-download", is_flag=True, help="Require the model cached; don't download.")
def ttft(prefix_tokens, cache_reuse, model_id, variant, threads, port, mlock, no_download) -> None:
    """MEASURED TTFT + prompt-cache demo: llama-server timings, cold vs warm shared prefix."""
    from .runner import ttft as run_ttft

    sys.exit(
        run_ttft(
            model_id=model_id,
            variant=variant,
            prefix_tokens=prefix_tokens,
            cache_reuse=cache_reuse,
            threads=threads,
            port=port,
            mlock=mlock,
            download=not no_download,
        )
    )


@main.command()
@click.option(
    "--prompt-tokens",
    "npp",
    type=int,
    default=2048,
    show_default=True,
    help="Prompt tokens per request.",
)
@click.option(
    "--gen", "ntg", type=int, default=32, show_default=True, help="Generated tokens per request."
)
@click.option(
    "--levels",
    default="1,2,4,8",
    show_default=True,
    help="Comma-separated parallel-request levels.",
)
@click.option("--model", "model_id", default=None, help="Model id (default: smoke model).")
@click.option("--variant", default=None, help="Quant variant (default: model's default).")
@click.option("--threads", type=int, default=None, help="Thread count (default: auto).")
@click.option("--no-download", is_flag=True, help="Require the model cached; don't download.")
def throughput(npp, ntg, levels, model_id, variant, threads, no_download) -> None:
    """Concurrency axis: aggregate tok/s at 1/2/4/8 parallel requests (llama-batched-bench)."""
    from .runner import throughput as run_tp

    sys.exit(
        run_tp(
            model_id=model_id,
            variant=variant,
            npp=npp,
            ntg=ntg,
            levels=[int(x) for x in levels.split(",") if x.strip()],
            threads=threads,
            download=not no_download,
        )
    )


@main.command()
@click.option(
    "--prompt-len", type=int, default=2048, show_default=True, help="Prefill context to profile."
)
@click.option("--model", "model_id", default=None, help="Model id (default: smoke model).")
@click.option("--variant", default=None, help="Quant variant (default: model's default).")
@click.option("--threads", type=int, default=None, help="Thread count (default: auto).")
@click.option("--no-download", is_flag=True, help="Require the model cached; don't download.")
def profile(prompt_len, model_id, variant, threads, no_download) -> None:
    """Profile a representative prefill run with Arm Performix (apx); no-op off Arm."""
    from .runner import profile as run_profile

    sys.exit(
        run_profile(
            model_id=model_id,
            variant=variant,
            prompt_len=prompt_len,
            threads=threads,
            download=not no_download,
        )
    )


@main.command()
@click.option(
    "--name", default=None, help="Experiment from experiments.yaml (default: its default)."
)
@click.option("--no-quality", is_flag=True, help="Skip the quality probe.")
@click.option("--no-download", is_flag=True, help="Require models cached; don't download.")
@click.option("--no-report", is_flag=True, help="Don't render the report afterwards.")
@click.option("--instance", "instance_name", default=None, help="Instance for the $/M-token cost.")
@click.option(
    "--rounds",
    type=int,
    default=1,
    show_default=True,
    help="Interleave the configs N times (A,B,A,B,...); the report medians per label.",
)
def experiment(name, no_quality, no_download, no_report, instance_name, rounds) -> None:
    """Run a before/after optimization experiment (quant/threads/pinning/KleidiAI) + quality."""
    from .runner import experiment as run_exp

    sys.exit(
        run_exp(
            name=name,
            quality=not no_quality,
            download=not no_download,
            render=not no_report,
            instance_name=instance_name,
            rounds=rounds,
        )
    )


@main.command()
@click.option(
    "--file",
    "corpus_file",
    type=click.Path(path_type=Path),
    default=None,
    help="Text corpus (default: the repo's README + docs, identical across configs).",
)
@click.option("--chunks", type=int, default=8, show_default=True, help="Perplexity chunks.")
@click.option("--model", "model_id", default=None, help="Model id (default: smoke model).")
@click.option("--variant", default=None, help="Quant variant (default: model's default).")
@click.option("--threads", type=int, default=None, help="Thread count (default: auto).")
@click.option("--no-download", is_flag=True, help="Require the model cached; don't download.")
def perplexity(corpus_file, chunks, model_id, variant, threads, no_download) -> None:
    """Perplexity guardrail via llama-perplexity — finer than the probe (lower = better)."""
    from .runner import perplexity as run_ppl

    sys.exit(
        run_ppl(
            model_id=model_id,
            variant=variant,
            corpus_file=corpus_file,
            chunks=chunks,
            threads=threads,
            download=not no_download,
        )
    )


@main.command()
@click.option(
    "--results-dir",
    "results_source",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory of result JSONs (default: bench/results).",
)
@click.option(
    "--instance",
    "instance_name",
    default=None,
    help="Instance for the $/M-token cost (default: instances.yaml default).",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: bench/reports).",
)
@click.option(
    "--demo",
    is_flag=True,
    help="Render a clearly-labeled synthetic DEMO report (no Arm box needed).",
)
def report(results_source, instance_name, out_dir, demo) -> None:
    """Render the before/after markdown + standalone HTML report from bench/results/*.json."""
    from .runner import report as run_report

    sys.exit(
        run_report(
            results_source=results_source,
            instance_name=instance_name,
            out_dir=out_dir,
            demo=demo,
        )
    )


@main.command()
@click.option("--enable", is_flag=True, help="Opt in to the agentic optimizer (required to run).")
@click.option(
    "--llm", is_flag=True, help="Use the Claude proposer ([agent] extra + ANTHROPIC_API_KEY)."
)
@click.option("--model", "model_id", default=None, help="Model id (default: smoke model).")
@click.option(
    "--target-context",
    type=int,
    default=2048,
    show_default=True,
    help="Context to optimize prefill for.",
)
@click.option("--max-iters", type=int, default=12, show_default=True, help="Max trials.")
@click.option(
    "--patience", type=int, default=3, show_default=True, help="Stop after N non-improving trials."
)
@click.option("--no-download", is_flag=True, help="Require models cached; don't download.")
@click.option("--no-profile", is_flag=True, help="Skip the Performix profiling step.")
def autotune(
    enable, llm, model_id, target_context, max_iters, patience, no_download, no_profile
) -> None:
    """Agent-in-the-loop optimizer (stretch). Strictly opt-in: pass --enable to run."""
    if not enable:
        console.print(
            "[yellow]autotune is opt-in[/] - pass [bold]--enable[/] to run it (stretch goal)."
        )
        console.print(
            "[dim]It loops: propose config -> benchmark -> repeat until no improvement. "
            "Default proposer is a heuristic grid (no API key); --llm uses Claude.[/]"
        )
        sys.exit(0)
    from .runner import autotune as run_autotune

    sys.exit(
        run_autotune(
            model_id=model_id,
            use_llm=llm,
            max_iters=max_iters,
            patience=patience,
            target_context=target_context,
            download=not no_download,
            do_profile=not no_profile,
        )
    )


if __name__ == "__main__":
    main()

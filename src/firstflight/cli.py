"""`firstflight` command-line entrypoint.

Subcommands: setup-engine · info · smoke · download · run · bench · ttft · throughput ·
report. All skip cleanly off Arm / without a llama.cpp build.
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

    table = Table(title="firstflight environment", show_header=False, title_style="bold")
    table.add_column("k", style="cyan", no_wrap=True)
    table.add_column("v")

    table.add_row("version", __version__)
    table.add_row("platform", platform_tag())
    table.add_row("arm linux", "yes" if is_arm_linux() else "no (Arm-only tools will no-op)")

    cli = llama_cpp.find_binary("cli")
    bench = llama_cpp.find_binary("bench")
    table.add_row("llama-cli", str(cli) if cli else "[yellow]not found[/]")
    table.add_row("llama-bench", str(bench) if bench else "[yellow]not found[/]")

    try:
        models = load_models()
        instances = load_instances()
        workloads = load_workloads()
        table.add_row("smoke model", f"{models.default_smoke}:{models.default_smoke_variant}")
        table.add_row("default instance", instances.default_instance)
        table.add_row("default workload", workloads.default_workload)
    except Exception as exc:  # config problems shouldn't crash `info`
        table.add_row("config", f"[red]error: {exc}[/]")

    console.print(table)


@main.command()
@click.option("--no-download", is_flag=True, help="Skip the model download step.")
@click.option("-n", "--n-predict", default=24, show_default=True, help="Tokens to generate.")
def smoke(no_download: bool, n_predict: int) -> None:
    """Download the tiny model + run llama.cpp once (proves the pipeline on any machine)."""
    from .runner import smoke as run_smoke

    sys.exit(run_smoke(download=not no_download, n_predict=n_predict))


@main.command()
@click.option("--force", is_flag=True, help="Re-download even if cached.")
def download(force: bool) -> None:
    """Download the default smoke model only."""
    from .config import load_models
    from .download import ensure_model

    models = load_models()
    spec, variant = models.smoke()
    path = ensure_model(spec, variant, force=force)
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
@click.option("--port", type=int, default=8033, show_default=True, help="Local server port.")
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


if __name__ == "__main__":
    main()

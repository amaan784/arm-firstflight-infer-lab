"""`firstflight` command-line entrypoint.

Subcommands: setup-engine · info · smoke · download.
`smoke` skips cleanly off Arm / without a llama.cpp build.
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


if __name__ == "__main__":
    main()

"""Top-level orchestration.

`smoke()` is the end-to-end sanity check that proves the pipeline on any machine.
`bench()` runs the prefill/TTFT sweep (Phase 1) via `firstflight.bench.prefill`, writes a
structured JSON result, and prints a summary. Both degrade gracefully (clean skip) when no
llama.cpp binary is present.
"""

from __future__ import annotations

from .config import load_models
from .download import ensure_model
from .engines import llama_cpp
from .util import console, safe


def _skip(message: str) -> int:
    console.print(f"\n[yellow]SKIP[/] {message}")
    console.print(
        "[dim]This is expected off-Arm or without a llama.cpp build. "
        "Point [bold]LLAMA_CPP_BIN[/] at a `llama-cli` binary or an extracted "
        "release directory to run the real inference.[/]\n"
    )
    return 0


def smoke(download: bool = True, n_predict: int = 24) -> int:
    """Download the tiny smoke model and run llama.cpp once.

    Returns a process exit code: 0 on success OR on a clean skip (no binary), 1 on a real
    failure (binary present but the run errored / produced no output).
    """
    console.rule("[bold]firstflight smoke[/]")

    models = load_models()
    spec, variant = models.smoke()
    console.print(f"Model: [bold]{spec.id}[/]:{variant.name}  ([dim]{spec.hf_repo}[/])")

    cli = llama_cpp.find_binary("cli")
    if cli is None:
        return _skip("no llama.cpp `llama-cli` binary found.")
    console.print(f"Engine: [green]{cli}[/]")

    if not download:
        return _skip("download disabled (--no-download) — nothing to run.")

    try:
        model_path = ensure_model(spec, variant)
    except Exception as exc:  # network/URL issues shouldn't crash the smoke target
        console.print(f"[red]Download failed:[/] {safe(str(exc))}")
        return 1

    console.print("Running one generation...")
    result = llama_cpp.run_once(cli, model_path, n_predict=n_predict)

    if not result.ok:
        console.print(f"[red]llama.cpp exited {result.returncode}[/]")
        if result.stderr.strip():
            console.print(f"[dim]{safe(result.stderr.strip()[-800:])}[/]")
        return 1

    completion = result.stdout.strip()
    if not completion:
        console.print("[red]No output produced - check the model and flags.[/]")
        return 1

    preview = completion[:240] + ("..." if len(completion) > 240 else "")
    console.print(f"\n[green]OK smoke passed[/] in {result.wall_s:.2f}s")
    console.print(f"[dim]completion:[/] {safe(preview)}\n")
    return 0

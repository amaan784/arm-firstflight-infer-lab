"""Top-level orchestration.

`smoke()` is the end-to-end sanity check that proves the pipeline on any machine.
`bench()` runs the prefill/TTFT sweep (Phase 1) via `firstflight.bench.prefill`, writes a
structured JSON result, and prints a summary. Both degrade gracefully (clean skip) when no
llama.cpp binary is present.
"""

from __future__ import annotations

from .bench import prefill
from .bench.result import result_filename
from .config import load_models, load_workloads
from .download import ensure_model
from .engines import llama_cpp
from .util import bytes_human, console, model_dir, results_dir, safe


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


def _resolve_model(models, model_id: str | None, variant: str | None):
    """Resolve (ModelSpec, variant_name) from optional overrides, defaulting to the smoke model."""
    if model_id is None:
        spec, mv = models.smoke()
        return spec, (variant or mv.name)
    spec = models.get(model_id)
    if variant:
        return spec, variant
    if models.default_smoke_variant in spec.variants:
        return spec, models.default_smoke_variant
    return spec, next(iter(spec.variants))


def _print_sweep_summary(result) -> None:
    from rich.table import Table

    title = f"prefill sweep - {result.model.id}:{result.model.variant} [{result.label}]"
    table = Table(title=title, title_style="bold")
    table.add_column("context (pp tokens)", justify="right", style="cyan")
    table.add_column("prefill tok/s", justify="right")
    table.add_column("+/- stddev", justify="right", style="dim")
    table.add_column("TTFT (s)", justify="right")
    for p in sorted(result.prefill_points, key=lambda x: x.n_prompt):
        ttft = "-" if p.ttft_s in (None, float("inf")) else f"{p.ttft_s:.3f}"
        table.add_row(
            str(p.n_prompt), f"{p.throughput_tok_s:.1f}", f"{p.throughput_stddev:.1f}", ttft
        )
    console.print(table)

    gen = result.gen_points
    if gen:
        console.print(
            f"generation: [bold]{gen[0].throughput_tok_s:.1f}[/] tok/s ({gen[0].n_gen} tok)"
        )
    if result.peak_rss_bytes:
        console.print(f"peak RSS: [bold]{bytes_human(result.peak_rss_bytes)}[/]")
    if result.engine.build_number:
        console.print(
            f"[dim]llama.cpp build b{result.engine.build_number} | {result.host.cpu_info}[/]"
        )


def bench(
    *,
    model_id: str | None = None,
    variant: str | None = None,
    workload: str | None = None,
    prompt_lengths: list[int] | None = None,
    threads: int | None = None,
    repetitions: int | None = None,
    n_gen: int | None = None,
    label: str = "baseline",
    download: bool = True,
    dry_run: bool = False,
) -> int:
    """Run the prefill/TTFT sweep and write a JSON result. Returns a process exit code."""
    console.rule(f"[bold]firstflight bench[/] [{label}]")

    models = load_models()
    workloads = load_workloads()
    spec, variant = _resolve_model(models, model_id, variant)
    mv = spec.variant(variant)
    wl = workloads.get(workload)
    lengths = prompt_lengths or wl.prompt_lengths
    reps = repetitions if repetitions is not None else wl.repeats
    gen = n_gen if n_gen is not None else (wl.gen_lengths[0] if wl.gen_lengths else 32)

    console.print(f"Model: [bold]{spec.id}[/]:{variant}   Workload: [bold]{wl.name}[/]")
    console.print(
        f"Context sweep: {lengths}   gen={gen}   repeats={reps}   threads={threads or 'auto'}"
    )

    bench_bin = llama_cpp.find_binary("bench")

    if dry_run:
        shown = bench_bin or "llama-bench"
        cmd = prefill.build_llama_bench_command(
            shown, model_dir() / mv.file, lengths, [gen], threads, reps, "json"
        )
        console.print("\n[dim]dry-run command:[/]")
        console.print("  " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
        return 0

    if bench_bin is None:
        return _skip("no llama.cpp `llama-bench` binary found.")

    if download:
        try:
            model_path = ensure_model(spec, mv)
        except Exception as exc:
            console.print(f"[red]Download failed:[/] {safe(str(exc))}")
            return 1
    else:
        model_path = model_dir() / mv.file
        if not model_path.exists():
            return _skip(f"model not present ({model_path.name}) and --no-download set.")

    console.print(f"Engine: [green]{bench_bin}[/]\nRunning sweep (this can take a while)...")
    try:
        result = prefill.run_sweep(
            bench_bin=bench_bin,
            model_path=model_path,
            model_id=spec.id,
            variant=variant,
            workload_name=wl.name,
            prompt_lengths=lengths,
            n_gen=gen,
            threads=threads,
            repetitions=reps,
            label=label,
        )
    except prefill.BenchError as exc:
        console.print(f"[red]Benchmark failed:[/] {safe(str(exc))}")
        return 1

    out = results_dir() / result_filename(spec.id, variant, label, result.timestamp)
    result.save_json(out)

    console.print()
    _print_sweep_summary(result)
    console.print(f"\n[green]OK[/] wrote {out}")
    return 0


def ttft(
    *,
    model_id: str | None = None,
    variant: str | None = None,
    prefix_tokens: int = 2048,
    cache_reuse: int = 256,
    threads: int | None = None,
    port: int = 8033,
    n_predict: int = 16,
    mlock: bool = False,
    download: bool = True,
) -> int:
    """MEASURED TTFT + prompt-cache demo via llama-server (see bench/ttft.py).

    Starts llama-server, sends the same long prefix with two different questions, and reports
    the server's own measured prompt-processing time cold vs warm. Clean skip without a
    llama-server binary.
    """
    import subprocess
    from datetime import UTC, datetime

    from .bench import ttft as tmod

    console.rule("[bold]firstflight ttft[/] (measured, via llama-server)")

    models = load_models()
    spec, variant = _resolve_model(models, model_id, variant)
    mv = spec.variant(variant)

    server_bin = llama_cpp.find_binary("server")
    if server_bin is None:
        return _skip("no llama.cpp `llama-server` binary found.")

    if download:
        try:
            model_path = ensure_model(spec, mv)
        except Exception as exc:
            console.print(f"[red]Download failed:[/] {safe(str(exc))}")
            return 1
    else:
        model_path = model_dir() / mv.file
        if not model_path.exists():
            return _skip(f"model not present ({model_path.name}) and --no-download set.")

    base = f"http://127.0.0.1:{port}"
    cmd = tmod.build_server_cmd(
        server_bin, model_path, port=port, threads=threads, cache_reuse=cache_reuse, mlock=mlock
    )
    console.print(f"Server: [green]{server_bin}[/]  (--cache-reuse {cache_reuse}, port {port})")
    console.print(f"Prefix target: ~{prefix_tokens} tokens; measuring cold vs warm turn...")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        tmod.wait_ready(base, proc=proc)
        prefix = tmod.build_prefix(prefix_tokens)
        cold, warm = tmod.measure_prompt_cache(base, prefix, n_predict=n_predict)
    except tmod.TtftError as exc:
        console.print(f"[red]TTFT measurement failed:[/] {safe(str(exc))}")
        return 1
    except Exception as exc:  # noqa: BLE001 - HTTP/etc: fail clean, never hang
        console.print(f"[red]TTFT measurement failed:[/] {safe(str(exc))}")
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    result = tmod.TtftResult(
        timestamp=datetime.now(UTC).isoformat(),
        model=spec.id,
        variant=variant,
        prefix_target_tokens=prefix_tokens,
        cache_reuse=cache_reuse,
        threads=threads,
        cold=cold,
        warm=warm,
    )

    from rich.table import Table

    table = Table(title="measured TTFT - shared prefix, cold vs warm", title_style="bold")
    table.add_column("turn", style="cyan")
    table.add_column("prompt tokens processed", justify="right")
    table.add_column("prompt time (ms)", justify="right")
    table.add_row("cold (turn 1)", str(cold.prompt_n), f"{cold.prompt_ms:.0f}")
    table.add_row("warm (turn 2)", str(warm.prompt_n), f"{warm.prompt_ms:.0f}")
    console.print(table)
    console.print(
        f"[green]prompt-cache effect:[/] [bold]{result.reduction_pct:.0f}%[/] less prefill "
        f"time on the warm turn ({cold.prompt_n} -> {warm.prompt_n} tokens processed)"
    )

    stamp = result.timestamp.replace(":", "").replace("-", "").replace(".", "")
    out = results_dir() / f"ttft_{stamp}.json"
    result.save_json(out)
    console.print(f"[green]OK[/] wrote {out}")
    return 0


def throughput(
    *,
    model_id: str | None = None,
    variant: str | None = None,
    npp: int = 2048,
    ntg: int = 32,
    levels: list[int] | None = None,
    threads: int | None = None,
    download: bool = True,
) -> int:
    """Concurrency axis: llama-batched-bench parallel-level sweep (see bench/throughput.py)."""
    from datetime import UTC, datetime

    from .bench import throughput as tp

    levels = levels or [1, 2, 4, 8]
    console.rule("[bold]firstflight throughput[/] (llama-batched-bench)")

    models = load_models()
    spec, variant = _resolve_model(models, model_id, variant)
    mv = spec.variant(variant)

    bin_path = llama_cpp.find_binary("batched")
    if bin_path is None:
        return _skip("no llama.cpp `llama-batched-bench` binary found.")

    if download:
        try:
            model_path = ensure_model(spec, mv)
        except Exception as exc:
            console.print(f"[red]Download failed:[/] {safe(str(exc))}")
            return 1
    else:
        model_path = model_dir() / mv.file
        if not model_path.exists():
            return _skip(f"model not present ({model_path.name}) and --no-download set.")

    console.print(f"Model: [bold]{spec.id}[/]:{variant}   npp={npp} ntg={ntg} parallel={levels}")
    console.print(f"Engine: [green]{bin_path}[/]\nRunning parallel sweep...")
    try:
        points = tp.run_sweep(
            bench_bin=bin_path,
            model_path=model_path,
            npp=npp,
            ntg=ntg,
            levels=levels,
            threads=threads,
        )
    except tp.ThroughputError as exc:
        console.print(f"[red]Throughput sweep failed:[/] {safe(str(exc))}")
        return 1

    result = tp.ThroughputResult(
        timestamp=datetime.now(UTC).isoformat(),
        model=spec.id,
        variant=variant,
        npp=npp,
        ntg=ntg,
        threads=threads,
        points=points,
    )

    from rich.table import Table

    table = Table(title="throughput vs parallel requests (aggregate tok/s)", title_style="bold")
    table.add_column("parallel", justify="right", style="cyan")
    table.add_column("prefill tok/s", justify="right")
    table.add_column("gen tok/s", justify="right")
    table.add_column("total tok/s", justify="right")
    for p in sorted(points, key=lambda x: x.parallel):
        table.add_row(str(p.parallel), f"{p.speed_pp:.1f}", f"{p.speed_tg:.1f}", f"{p.speed:.1f}")
    console.print(table)

    stamp = result.timestamp.replace(":", "").replace("-", "").replace(".", "")
    out = results_dir() / f"throughput_{stamp}.json"
    result.save_json(out)
    console.print(f"[green]OK[/] wrote {out}")
    return 0


def report(
    *,
    results_source=None,
    out_dir=None,
    instance_name: str | None = None,
    demo: bool = False,
) -> int:
    """Render the markdown + standalone HTML report. Returns a process exit code."""
    from .report import render

    console.rule("[bold]firstflight report[/]")
    try:
        paths = render.render_report(
            results_source=results_source,
            out_dir=out_dir,
            instance_name=instance_name,
            demo=demo,
        )
    except render.ReportError as exc:
        console.print(f"[red]Report error:[/] {safe(str(exc))}")
        return 1

    if paths is None:
        console.print(
            "[yellow]No results found.[/] Run `firstflight bench` first, "
            "or preview the layout with `firstflight report --demo`."
        )
        return 0

    console.print("[green]OK[/] wrote report:")
    console.print(f"  HTML:   {paths.html}")
    console.print(f"  MD:     {paths.markdown}")
    console.print(f"  charts: {len(paths.charts)} PNG(s)")
    return 0

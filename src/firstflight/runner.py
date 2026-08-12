"""Top-level orchestration.

`smoke()` is the end-to-end sanity check; it runs on any machine. `bench()` runs the
prefill/TTFT sweep via `firstflight.bench.prefill`, writes a JSON result, prints a
summary. Both skip cleanly when no llama.cpp binary is present.
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
        "Point [bold]LLAMA_CPP_BIN[/] at a `llama-completion` binary or an extracted "
        "release directory to run the real inference.[/]\n"
    )
    return 0


def smoke(download: bool = True, n_predict: int = 24, timeout: float = 90.0) -> int:
    """Download the tiny smoke model and run llama.cpp once.

    Exit code: 0 on success or on a clean skip (no binary), 1 on a real failure (binary
    present, run errored or produced no output). 90s default: 24 tokens on a 0.5B model
    takes seconds; five silent minutes on the first documented command is a bounce.
    """
    console.rule("[bold]firstflight smoke[/]")

    models = load_models()
    spec, variant = models.smoke()
    console.print(f"Model: [bold]{spec.id}[/]:{variant.name}  ([dim]{spec.hf_repo}[/])")

    cli = llama_cpp.find_binary("cli")
    if cli is None:
        return _skip("no llama.cpp `llama-completion` binary found.")
    console.print(f"Engine: [green]{cli}[/]")

    if not download:
        return _skip("download disabled (--no-download) — nothing to run.")

    try:
        model_path = ensure_model(spec, variant)
    except Exception as exc:  # network/URL issues shouldn't crash the smoke target
        console.print(f"[red]Download failed:[/] {safe(str(exc))}")
        return 1

    console.print("Running one generation...")
    result = llama_cpp.run_once(cli, model_path, n_predict=n_predict, timeout=timeout)

    if not result.ok:
        console.print(f"[red]llama.cpp exited {result.returncode}[/]")
        detail = result.stderr.strip()
        if "timed out" in result.stderr:
            # Already bounded, and written front-to-back (the engine's own first words
            # before it stalled). Tailing it would cut off the half that explains the hang.
            console.print(f"[dim]{safe(detail)}[/]")
            console.print(
                "[dim]If the engine printed a rejected flag above, that is the cause, not "
                "the machine. Otherwise re-run with --timeout 300.[/]"
            )
        elif detail:
            console.print(f"[dim]{safe(detail[-800:])}[/]")
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
    """(ModelSpec, variant_name) from optional overrides. Defaults to the smoke model."""
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
    """Run the prefill/TTFT sweep, write a JSON result. Returns a process exit code."""
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
            shown,
            model_dir() / mv.file,
            lengths,
            [gen],
            threads,
            reps,
            "json",
            mmap=False,
            delay=2,  # mirror run_sweep's steadiness flags so dry-run shows the real command
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
    port: int | None = None,
    n_predict: int = 16,
    mlock: bool = False,
    download: bool = True,
) -> int:
    """MEASURED TTFT + prompt-cache demo via llama-server (see bench/ttft.py).

    Sends the same long prefix with two different questions and reports the server's own
    prompt-processing time, cold vs warm. Clean skip without a llama-server binary.
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

    # Ephemeral port by default: a fixed port can race a stale/foreign server and measure
    # the wrong model. --port overrides.
    if port is None:
        port = tmod.free_port()
    base = f"http://127.0.0.1:{port}"
    # Context has to hold prefix + question + generation. The 4-chars/token sizing heuristic
    # undershoots, so pad; the server default of 4096 would truncate.
    ctx = int(prefix_tokens * 1.6) + n_predict + 256
    cmd = tmod.build_server_cmd(
        server_bin,
        model_path,
        port=port,
        threads=threads,
        cache_reuse=cache_reuse,
        mlock=mlock,
        ctx=ctx,
    )
    console.print(f"Server: [green]{server_bin}[/]  (--cache-reuse {cache_reuse}, port {port})")
    console.print(f"Prefix target: ~{prefix_tokens} tokens; measuring cold vs warm turn...")

    # stdout/stderr -> DEVNULL: an undrained PIPE fills the OS buffer and blocks the server
    # mid-request (observed). We only need the HTTP responses.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
            proc.wait(timeout=15)

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


def profile(
    *,
    model_id: str | None = None,
    variant: str | None = None,
    prompt_len: int = 2048,
    threads: int | None = None,
    download: bool = True,
) -> int:
    """Profile one prefill run with Arm Performix (apx). No-op off Arm."""
    from .profile.performix import PerformixProfiler, profile_filename

    console.rule("[bold]firstflight profile[/]")
    prof = PerformixProfiler()
    if not prof.available():
        console.print(f"[yellow]Performix unavailable:[/] {prof.unavailable_reason()}")
        console.print("[dim]Off-Arm this is expected - profiling no-ops with this message.[/]")
        return 0

    models = load_models()
    spec, variant = _resolve_model(models, model_id, variant)
    mv = spec.variant(variant)

    bench_bin = llama_cpp.find_binary("bench")
    if bench_bin is None:
        return _skip("no llama.cpp `llama-bench` binary to profile.")

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

    target = prefill.build_llama_bench_command(
        bench_bin, model_path, [prompt_len], [0], threads, 1, "json"
    )
    console.print(f"Profiling prefill @ {prompt_len} tokens with Arm Performix (apx)...")
    result = prof.profile(target)

    if result.skipped:
        console.print(f"[yellow]Profile skipped:[/] {safe(result.reason)}")
        return 0

    out = results_dir() / profile_filename(result.timestamp)
    result.save_json(out)

    from rich.table import Table

    table = Table(title="Top hotspots (Arm Performix)", title_style="bold")
    table.add_column("function", style="cyan")
    table.add_column("module", style="dim")
    table.add_column("%", justify="right")
    for h in result.hotspots:
        table.add_row(h.function, h.module, f"{h.percent:.1f}")
    console.print(table)
    console.print(f"\n[green]OK[/] wrote {out}")
    return 0


def experiment(
    *,
    name: str | None = None,
    quality: bool = True,
    download: bool = True,
    render: bool = True,
    instance_name: str | None = None,
    rounds: int = 1,
) -> int:
    """Before/after experiment: bench + quality-eval each config, model held fixed.

    Each config becomes a labelled SweepResult in bench/results/*.json; the report shows the
    speed delta AND the quality delta. Configs whose binary/model are missing are skipped
    with a message, so this still runs off the bench box.

    rounds > 1 interleaves the configs round-robin (A,B,A,B,...) and writes one result per
    round; the report collapses same-label rounds to the median with the between-round
    spread. Strictly-blocked runs would confound slow drift (co-tenants, thermals) with the
    treatment. Quality runs once per config (round 0): greedy decoding doesn't drift.
    """
    from .config import load_experiments
    from .eval import quality as qmod

    console.rule("[bold]firstflight experiment[/]")
    spec = load_experiments().get(name)
    models = load_models()
    wl = load_workloads().get(spec.workload)
    console.print(f"Experiment: [bold]{spec.name}[/] - {spec.description}")
    console.print(
        f"Workload: {wl.name}   configs: {len(spec.configs)}   quality: {quality}"
        + (f"   rounds: {rounds}" if rounds > 1 else "")
    )

    written: list = []
    for rnd in range(max(1, rounds)):
        if rounds > 1:
            console.print(f"\n[bold cyan]-- round {rnd + 1}/{rounds} --[/]")
        written += _experiment_round(
            spec, models, wl, quality=quality and rnd == 0, download=download, qmod=qmod
        )

    if not written:
        # Exit red: a green run with zero measurements would hide an env/plumbing mistake.
        console.print("\n[red]No configs ran[/] (binaries/models missing) - see messages above.")
        return 1

    console.print(f"\n[green]OK[/] {len(written)} result(s) in bench/results/")
    if render:
        console.print("Rendering report...")
        return report(instance_name=instance_name)
    return 0


def _experiment_round(spec, models, wl, *, quality: bool, download: bool, qmod) -> list:
    """One pass over the experiment's configs. Returns the result paths written."""
    written: list = []
    for cfg in spec.configs:
        model_id = cfg.model_id or spec.model
        try:
            mspec = models.get(model_id)
            mv = mspec.variant(cfg.variant)
        except KeyError as exc:
            console.print(f"[yellow]skip {cfg.label}:[/] {safe(str(exc))}")
            continue

        if cfg.bin and "${" in cfg.bin:
            # os.path.expandvars leaves ${VAR} untouched when the env var is unset.
            console.print(
                f"[yellow]skip {cfg.label}:[/] env var not set for bin override "
                f"({safe(cfg.bin)}) - export it or edit experiments.yaml."
            )
            continue

        bench_bin = llama_cpp.find_binary("bench", env_value=cfg.bin or None)
        if bench_bin is None:
            where = f" at {cfg.bin}" if cfg.bin else ""
            console.print(f"[yellow]skip {cfg.label}:[/] no llama-bench binary{safe(where)}.")
            continue

        if download:
            try:
                model_path = ensure_model(mspec, mv)
            except Exception as exc:
                console.print(f"[red]{cfg.label} download failed:[/] {safe(str(exc))}")
                continue
        else:
            model_path = model_dir() / mv.file
            if not model_path.exists():
                console.print(f"[yellow]skip {cfg.label}:[/] model not cached.")
                continue

        pin = f", mask={cfg.cpu_mask}" if cfg.cpu_mask else ""
        if cfg.cache_type_k or cfg.cache_type_v:
            pin += f", kv={cfg.cache_type_k or 'f16'}/{cfg.cache_type_v or 'f16'}"
        if cfg.ubatch_size:
            pin += f", ub={cfg.ubatch_size}"
        if cfg.flash_attn:
            pin += f", fa={cfg.flash_attn}"
        console.print(
            f"\n[bold]> {cfg.label}[/]  {model_id}:{cfg.variant}  "
            f"threads={cfg.threads or 'auto'}{pin}"
        )
        try:
            sweep = prefill.run_sweep(
                bench_bin=bench_bin,
                model_path=model_path,
                model_id=model_id,
                variant=cfg.variant,
                workload_name=wl.name,
                prompt_lengths=wl.prompt_lengths,
                n_gen=cfg.gen
                if cfg.gen is not None
                else (wl.gen_lengths[0] if wl.gen_lengths else 32),
                threads=cfg.threads,
                repetitions=wl.repeats,
                label=cfg.label,
                cpu_mask=cfg.cpu_mask,
                cpu_strict=cfg.cpu_strict,
                cache_type_k=cfg.cache_type_k,
                cache_type_v=cfg.cache_type_v,
                ubatch_size=cfg.ubatch_size,
                flash_attn=cfg.flash_attn,
            )
        except prefill.BenchError as exc:
            console.print(f"[red]{cfg.label} bench failed:[/] {safe(str(exc))}")
            continue

        sweep.experiment = spec.name  # so combined reports can group by experiment

        cli_bin = llama_cpp.find_binary("cli", env_value=cfg.bin or None)

        # Prove, don't assume, which kernel path is active for this binary+model.
        if cli_bin is not None:
            probe = llama_cpp.probe_backend(cli_bin, model_path, threads=cfg.threads)
            sweep.host.kleidiai = probe.kleidiai
            sweep.host.system_info = probe.system_info
            sweep.host.kernel_buffer = probe.kernel_buffer
            sweep.host.cpu_features = probe.cpu_features
            sweep.host.kernel_tier = probe.kernel_tier
            sweep.host.repack = probe.repack
            if probe.kleidiai is not None:
                console.print(f"  kleidiai active: {'yes' if probe.kleidiai else 'no'}")
            if probe.kernel_buffer:
                console.print(f"  weight buffer: {safe(probe.kernel_buffer)}")
            if probe.kernel_tier:
                console.print(f"  kernel tier:   {safe(probe.kernel_tier)}")

        if quality:
            if cli_bin is not None:
                qr = qmod.run_probe(cli_bin, model_path, threads=cfg.threads)
                sweep.quality = {
                    "method": qr.method,
                    "accuracy": qr.accuracy,
                    "n_correct": qr.n_correct,
                    "n_total": qr.n_total,
                }
                console.print(f"  quality: {qr.n_correct}/{qr.n_total} = {qr.accuracy:.0%}")
            else:
                console.print("  [dim]quality skipped: no llama-completion binary[/]")

            # Perplexity: the finer instrument (the probe can't resolve ~1% shifts).
            # Best-effort: a missing binary or parse failure never sinks the experiment.
            from .eval import perplexity as pmod

            ppl_bin = llama_cpp.find_binary("perplexity", env_value=cfg.bin or None)
            if ppl_bin is not None:
                try:
                    corpus = pmod.ensure_default_corpus(model_dir())
                    if corpus is not None:
                        pr = pmod.run_perplexity(ppl_bin, model_path, corpus, threads=cfg.threads)
                        sweep.quality = {**(sweep.quality or {}), "ppl": pr.ppl}
                        console.print(f"  perplexity: {pr.ppl:.2f} ({pr.corpus})")
                except pmod.PerplexityError as exc:
                    console.print(f"  [dim]perplexity skipped: {safe(str(exc)[:160])}[/]")

        out = results_dir() / result_filename(model_id, cfg.variant, cfg.label, sweep.timestamp)
        sweep.save_json(out)
        written.append(out)
        console.print(f"  [green]OK[/] wrote {out.name}")

    return written


def perplexity(
    *,
    model_id: str | None = None,
    variant: str | None = None,
    corpus_file=None,
    chunks: int = 8,
    threads: int | None = None,
    download: bool = True,
) -> int:
    """Perplexity over a fixed corpus via llama-perplexity. Skips cleanly without the binary."""
    from pathlib import Path

    from .eval import perplexity as pmod

    console.rule("[bold]firstflight perplexity[/]")
    ppl_bin = llama_cpp.find_binary("perplexity")
    if ppl_bin is None:
        return _skip("no llama.cpp `llama-perplexity` binary found.")

    models = load_models()
    spec, var_name = _resolve_model(models, model_id, variant)
    mv = spec.variant(var_name)
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

    corpus = Path(corpus_file) if corpus_file else pmod.ensure_default_corpus(model_dir())
    if corpus is None:
        console.print("[red]No corpus:[/] pass --file (the default docs corpus needs a checkout).")
        return 1

    console.print(f"Engine: [green]{ppl_bin}[/]   corpus: {corpus.name}   chunks: {chunks}")
    try:
        pr = pmod.run_perplexity(ppl_bin, model_path, corpus, chunks=chunks, threads=threads)
    except pmod.PerplexityError as exc:
        console.print(f"[red]perplexity failed:[/] {safe(str(exc))}")
        return 1
    console.print(f"\n[green]OK[/] PPL = [bold]{pr.ppl:.4f}[/] (lower is better)")
    return 0


def autotune(
    *,
    model_id: str | None = None,
    use_llm: bool = False,
    max_iters: int = 12,
    patience: int = 3,
    target_context: int = 2048,
    download: bool = True,
    do_profile: bool = True,
) -> int:
    """Agent-in-the-loop optimizer: propose -> benchmark -> loop until no improvement.

    Opt-in only; the CLI gates it behind --enable. No binary -> skip.
    """
    import json
    import os
    from datetime import UTC, datetime

    from .autotune import agent
    from .profile.performix import PerformixProfiler

    console.rule("[bold]firstflight autotune[/]")
    models = load_models()
    spec, _ = _resolve_model(models, model_id, None)
    variants = list(spec.variants.keys())
    if not variants:
        console.print(f"[yellow]No variants for {spec.id}.[/]")
        return 0

    cpu = os.cpu_count() or 4
    thread_options = sorted({t for t in (2, 4, 8, cpu) if 1 <= t <= cpu}) or [cpu]

    bench_bin = llama_cpp.find_binary("bench")
    if bench_bin is None:
        return _skip("no llama.cpp `llama-bench` binary to autotune.")

    hotspots: list = []
    if do_profile:
        prof = PerformixProfiler()
        if prof.available():
            try:
                mv0 = spec.variant(variants[0])
                mp0 = ensure_model(spec, mv0) if download else model_dir() / mv0.file
                target = prefill.build_llama_bench_command(
                    bench_bin, mp0, [target_context], [0], None, 1, "json"
                )
                pres = prof.profile(target)
                if not pres.skipped:
                    hotspots = [
                        {"function": h.function, "percent": h.percent} for h in pres.hotspots
                    ]
            except Exception:  # noqa: BLE001 - profiling is best-effort
                hotspots = []

    state = agent.AutotuneState(model_id=spec.id, target_context=target_context, hotspots=hotspots)
    proposer = (
        agent.LLMProposer(variants, thread_options)
        if use_llm
        else agent.HeuristicProposer(variants, thread_options)
    )

    def benchmark_fn(cfg: agent.TuneConfig) -> agent.TuneTrial:
        try:
            mv = spec.variant(cfg.variant)
            model_path = ensure_model(spec, mv) if download else model_dir() / mv.file
            sweep = prefill.run_sweep(
                bench_bin=bench_bin,
                model_path=model_path,
                model_id=spec.id,
                variant=cfg.variant,
                workload_name="autotune",
                prompt_lengths=[target_context],
                n_gen=0,
                threads=cfg.threads,
                repetitions=3,
                label=cfg.label(),
                cpu_mask=cfg.cpu_mask,
            )
            pt = next((p for p in sweep.prefill_points if p.n_prompt == target_context), None)
            score = pt.throughput_tok_s if pt else 0.0
            console.print(f"  {cfg.label()}: [bold]{score:.1f}[/] tok/s")
            return agent.TuneTrial(
                config=cfg, score=score, detail={"ttft_s": pt.ttft_s if pt else None}
            )
        except Exception as exc:  # noqa: BLE001 - a bad config scores 0, loop continues
            console.print(f"  {cfg.label()}: [red]failed[/] ({safe(str(exc))})")
            return agent.TuneTrial(config=cfg, score=0.0, detail={"error": str(exc)})

    proposer_name = "LLM (Claude)" if use_llm else "heuristic grid"
    console.print(
        f"Tuning [bold]{spec.id}[/] | variants={variants} | threads={thread_options} | "
        f"target={target_context} tok | proposer={proposer_name}"
    )
    state = agent.optimize(
        proposer, benchmark_fn, initial_state=state, max_iters=max_iters, patience=patience
    )

    best = state.best()
    if best is None:
        console.print("[yellow]No trials ran.[/]")
        return 0

    from rich.table import Table

    table = Table(title="autotune trajectory", title_style="bold")
    table.add_column("config", style="cyan")
    table.add_column("prefill tok/s", justify="right")
    for t in state.history:
        marker = "  <- best" if t is best else ""
        table.add_row(t.config.label(), f"{t.score:.1f}{marker}")
    console.print(table)
    console.print(f"\n[green]Best:[/] {best.config.label()} @ [bold]{best.score:.1f}[/] tok/s")

    out = results_dir() / f"autotune_{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    payload = {
        "model": spec.id,
        "target_context": target_context,
        "metric": state.metric,
        "proposer": proposer_name,
        "best": {
            "variant": best.config.variant,
            "threads": best.config.threads,
            "score": best.score,
        },
        "trials": [
            {
                "label": t.config.label(),
                "variant": t.config.variant,
                "threads": t.config.threads,
                "score": t.score,
            }
            for t in state.history
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(f"[green]OK[/] wrote {out}")
    return 0

"""Prefill / TTFT benchmark.

Prefill (prompt processing) is what agentic/RAG apps feel as time-to-first-token. Drives
`llama-bench` over a sweep of prompt lengths, then derives TTFT from prefill throughput.

llama-bench flags (verified 2026-06-26):
  -p/--n-prompt <n[,n...]>   prompt-processing tokens (prefill)   [default 512]
  -n/--n-gen    <n[,n...]>   generation tokens                    [default 128]
  -t/--threads  <n[,n...]>   threads
  -r/--repetitions <n>       repeats per config
  -o/--output   <csv|json|jsonl|md|sql>   results format          [default md]
  -C/--cpu-mask <hex> + --cpu-strict <0|1>  CPU affinity / pinning  (pinning experiment)
  -ctk/--cache-type-k <t> + -ctv/--cache-type-v <t>  KV-cache types [default f16]
  -ub/--ubatch-size <n>                    physical micro-batch     [default 512]
  -fa/--flash-attn <on|off|auto>           flash attention          [default auto]
                                           (all verified against the real b9873 binary 2026-07-07)
  --prio <-1|0|1|2|3>                      process/thread priority  [default 0]
  -mmp/--mmap <0|1>                        mmap the model           [default 1]
  --delay <seconds>                        pause between tests      [default 0]
                                           (verified on the real b9873 binary 2026-08-09)
  --no-warmup                              warm-up runs are ON by default; this disables them

One row per test: a prompt-processing (pp) row has n_prompt>0, n_gen==0; a token-generation
(tg) row has n_gen>0, n_prompt==0. The JSON is a top-level array of row objects; each row
also carries samples_ns / samples_ts (per-repetition raw samples).
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ..util import is_arm_linux, machine
from .memory import run_with_peak_rss
from .result import BenchPoint, EngineInfo, HostInfo, ModelInfo, SweepResult

# llama-bench JSON field names, verified vs real b9873. One place to fix if they drift.
# Source: tools/llama-bench/llama-bench.cpp (json printer) + tools/llama-bench/README.md
F_AVG_TS = "avg_ts"
F_STD_TS = "stddev_ts"  # across repetitions
F_N_PROMPT = "n_prompt"
F_N_GEN = "n_gen"
F_THREADS = "n_threads"
F_MODEL_FILENAME = "model_filename"
F_MODEL_SIZE = "model_size"  # bytes
F_MODEL_PARAMS = "model_n_params"
F_BUILD_NUMBER = "build_number"
F_BUILD_COMMIT = "build_commit"
F_CPU_INFO = "cpu_info"
F_BACKENDS = "backends"


class BenchError(RuntimeError):
    """llama-bench run failed: non-zero exit or unparseable output."""


# --- command building ---


def build_llama_bench_command(
    bench_bin: Path,
    model_path: Path,
    prompt_lengths: list[int],
    gen_lengths: list[int],
    threads: int | None = None,
    repetitions: int = 5,
    output: str = "json",
    cpu_mask: str = "",
    cpu_strict: bool = False,
    cache_type_k: str = "",
    cache_type_v: str = "",
    ubatch_size: int | None = None,
    flash_attn: str = "",
    prio: int | None = None,
    mmap: bool | None = None,
    delay: int | None = None,
) -> list[str]:
    """Build a `llama-bench` invocation for a prefill/generation sweep.

    Comma-joined lists cover the whole context sweep in one call. output="json" gives the
    array parsed into bench/results/*.json. Experiment axes: cpu_mask (hex) + cpu_strict =
    pinning; cache_type_k/cache_type_v (e.g. "q8_0") = quantized KV-cache; ubatch_size (-ub)
    = prefill micro-batch, i.e. bigger tiles for the GEMM kernels; flash_attn = on/off/auto.

    Run quantized KV-cache with flash_attn="on". Without FA the cache is dequantized every
    attention step and can come out slower than f16.
    """
    if not prompt_lengths:
        raise ValueError("prompt_lengths must be non-empty for a prefill sweep")
    cmd = [
        str(bench_bin),
        "-m",
        str(model_path),
        "-p",
        ",".join(str(p) for p in prompt_lengths),
        "-n",
        ",".join(str(g) for g in gen_lengths) if gen_lengths else "0",
        "-r",
        str(repetitions),
        "-o",
        output,
    ]
    if threads:
        cmd += ["-t", str(threads)]
    if cpu_mask:
        cmd += ["-C", cpu_mask]
        if cpu_strict:
            cmd += ["--cpu-strict", "1"]
    if cache_type_k:
        cmd += ["-ctk", cache_type_k]
    if cache_type_v:
        cmd += ["-ctv", cache_type_v]
    if ubatch_size:
        cmd += ["-ub", str(ubatch_size)]
    if flash_attn:
        cmd += ["-fa", flash_attn]
    # Measurement-steadiness flags (all verified on the real b9873 binary, 2026-08-09):
    # --prio 2 resists co-tenant scheduling noise, -mmp 0 makes peak RSS honest malloc
    # (no lazily-paged mapped file) at some load-time cost, --delay settles between tests.
    if prio is not None:
        cmd += ["--prio", str(prio)]
    if mmap is not None:
        cmd += ["-mmp", "1" if mmap else "0"]
    if delay:
        cmd += ["--delay", str(delay)]
    return cmd


# --- derivations ---


def ttft_seconds(prefill_tok_s: float, prompt_tokens: int) -> float:
    """TTFT derived from prefill throughput. +inf when throughput is non-positive.

    TTFT ~= prompt_tokens / prefill_throughput, since the whole prompt has to be processed
    before the first token comes out.
    """
    if prefill_tok_s <= 0:
        return float("inf")
    return prompt_tokens / prefill_tok_s


# --- parsing ---


def parse_rows(data: str | list) -> list[dict]:
    """`llama-bench -o json` output as row dicts."""
    obj = json.loads(data) if isinstance(data, str) else data
    if not isinstance(obj, list):
        raise BenchError(f"expected a JSON array from llama-bench, got {type(obj).__name__}")
    return [r for r in obj if isinstance(r, dict)]


def row_kind(row: dict) -> str:
    n_prompt = int(row.get(F_N_PROMPT, 0) or 0)
    n_gen = int(row.get(F_N_GEN, 0) or 0)
    if n_prompt > 0 and n_gen == 0:
        return "pp"
    if n_gen > 0 and n_prompt == 0:
        return "tg"
    return "mixed"


def rows_to_points(rows: list[dict]) -> list[BenchPoint]:
    """llama-bench rows as BenchPoints. TTFT derived for prefill rows."""
    points: list[BenchPoint] = []
    for row in rows:
        kind = row_kind(row)
        if kind == "mixed":
            continue  # neither a clean pp nor tg test
        n_prompt = int(row.get(F_N_PROMPT, 0) or 0)
        n_gen = int(row.get(F_N_GEN, 0) or 0)
        avg_ts = float(row.get(F_AVG_TS, 0.0) or 0.0)
        std_ts = float(row.get(F_STD_TS, 0.0) or 0.0)
        threads = int(row.get(F_THREADS, 0) or 0)
        ttft = ttft_seconds(avg_ts, n_prompt) if kind == "pp" else None
        raw = row.get("samples_ts") or []
        samples = [float(x) for x in raw if isinstance(x, (int, float))]
        points.append(
            BenchPoint(
                kind=kind,
                n_prompt=n_prompt,
                n_gen=n_gen,
                throughput_tok_s=avg_ts,
                throughput_stddev=std_ts,
                ttft_s=ttft,
                threads=threads,
                samples_ts=samples,
            )
        )
    return points


def _meta_from_rows(rows: list[dict]) -> tuple[EngineInfo, ModelInfo, str, int]:
    """engine/model/cpu metadata off the first row. Constant across rows."""
    first = rows[0] if rows else {}
    build_number = first.get(F_BUILD_NUMBER)
    engine = EngineInfo(
        build_number=int(build_number) if build_number not in (None, "") else None,
        build_commit=str(first.get(F_BUILD_COMMIT, "")),
        backends=str(first.get(F_BACKENDS, "")),
    )
    model = ModelInfo(
        id="",  # config id, filled by caller; not in llama-bench output
        variant="",
        filename=str(first.get(F_MODEL_FILENAME, "")),
        size_bytes=_maybe_int(first.get(F_MODEL_SIZE)),
        n_params=_maybe_int(first.get(F_MODEL_PARAMS)),
    )
    cpu_info = str(first.get(F_CPU_INFO, ""))
    threads = int(first.get(F_THREADS, 0) or 0)
    return engine, model, cpu_info, threads


def _maybe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_thp_mode() -> str:
    """Linux transparent-hugepage mode, e.g. 'madvise'. Empty off-Linux or unreadable.

    AWS's Graviton perf runbook recommends THP always/madvise for large weight working sets,
    so record the actual state to keep runs comparable.
    """
    try:
        text = Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text(encoding="ascii")
    except OSError:
        return ""
    for tok in text.split():
        if tok.startswith("[") and tok.endswith("]"):
            return tok[1:-1]
    return text.strip()


# --- the sweep ---


def run_sweep(
    *,
    bench_bin: Path,
    model_path: Path,
    model_id: str,
    variant: str,
    workload_name: str,
    prompt_lengths: list[int],
    n_gen: int,
    threads: int | None,
    repetitions: int,
    label: str = "baseline",
    cpu_mask: str = "",
    cpu_strict: bool = False,
    cache_type_k: str = "",
    cache_type_v: str = "",
    ubatch_size: int | None = None,
    flash_attn: str = "",
    timeout: float = 7200.0,
    prio: int | None = 2,
    mmap: bool | None = False,
    delay: int | None = 2,
) -> SweepResult:
    """Run the prefill/generation sweep. Raises BenchError if the run fails.

    Callers must have already checked that bench_bin exists. Off-Arm / no-binary is a CLI
    skip, not an exception.
    """
    cmd = build_llama_bench_command(
        bench_bin,
        model_path,
        prompt_lengths,
        [n_gen],
        threads,
        repetitions,
        "json",
        cpu_mask=cpu_mask,
        cpu_strict=cpu_strict,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        ubatch_size=ubatch_size,
        flash_attn=flash_attn,
        prio=prio,
        mmap=mmap,
        delay=delay,
    )
    try:
        returncode, stdout, stderr, peak = run_with_peak_rss(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # A stuck/slow sweep must not take down the whole run: raise BenchError so the
        # experiment loop skips this config and every later step (incl. report) still runs.
        raise BenchError(f"llama-bench timed out after {timeout:.0f}s: {cmd[0]}") from exc
    if returncode != 0:
        tail = stderr.strip()[-1200:]
        raise BenchError(f"llama-bench exited {returncode}\n{tail}")

    try:
        rows = parse_rows(stdout)
    except json.JSONDecodeError as exc:
        raise BenchError(f"could not parse llama-bench JSON: {exc}") from exc

    points = rows_to_points(rows)
    engine, model, cpu_info, row_threads = _meta_from_rows(rows)
    engine.binary = str(bench_bin)
    model.id = model_id
    model.variant = variant

    host = HostInfo(
        platform=f"{platform.system()}-{machine()}",
        arch=machine(),
        cpu_count=row_threads or (threads or 0),
        is_arm_linux=is_arm_linux(),
        cpu_info=cpu_info,
        thp=read_thp_mode(),
    )

    return SweepResult(
        label=label,
        timestamp=datetime.now(UTC).isoformat(),
        workload=workload_name,
        host=host,
        engine=engine,
        model=model,
        threads=threads or row_threads or 0,
        repetitions=repetitions,
        n_gen=n_gen,
        peak_rss_bytes=peak,
        points=points,
    )

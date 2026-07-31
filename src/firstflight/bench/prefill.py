"""Prefill / TTFT benchmark — the headline angle.

Long-context *prefill* (prompt processing) is what agentic/RAG apps feel as time-to-first-
token. We drive `llama-bench` across a sweep of prompt lengths so prefill scaling is visible,
and derive TTFT from prefill throughput.

llama-bench flags (verified 2026-06-26):
  -p/--n-prompt <n[,n...]>   prompt-processing tokens (prefill)   [default 512]
  -n/--n-gen    <n[,n...]>   generation tokens                    [default 128]
  -t/--threads  <n[,n...]>   threads
  -r/--repetitions <n>       repeats per config
  -o/--output   <csv|json|jsonl|md|sql>   results format          [default md]
  -C/--cpu-mask <hex> + --cpu-strict <0|1>  CPU affinity / pinning  (Phase 4)
  -ctk/--cache-type-k <t> + -ctv/--cache-type-v <t>  KV-cache types [default f16]
  -ub/--ubatch-size <n>                    physical micro-batch     [default 512]
  -fa/--flash-attn <on|off|auto>           flash attention          [default auto]
                                           (all verified against the real b9873 binary 2026-07-07)
  --no-warmup                              warm-up runs are ON by default; this disables them

llama-bench emits one row per test: a prompt-processing (pp) row has n_prompt>0, n_gen==0;
a token-generation (tg) row has n_gen>0, n_prompt==0. The JSON is a top-level array of row
objects; each row also carries samples_ns / samples_ts arrays (per-repetition raw samples).
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

from ..util import is_arm_linux, machine
from .memory import run_with_peak_rss
from .result import BenchPoint, EngineInfo, HostInfo, ModelInfo, SweepResult

# --- llama-bench JSON field names (verified Phase 1; centralized for easy correction) ------
# Source: tools/llama-bench/llama-bench.cpp (json printer) + tools/llama-bench/README.md
F_AVG_TS = "avg_ts"  # average tokens/sec
F_STD_TS = "stddev_ts"  # stddev of tokens/sec across repetitions
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
    """A llama-bench run failed (non-zero exit / unparseable output)."""


# --- command building ---------------------------------------------------------


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
) -> list[str]:
    """Construct a `llama-bench` invocation for a prefill/generation sweep.

    Comma-joined lists let one call cover the whole context sweep. `output="json"` yields a
    machine-readable array we parse into bench/results/*.json. Experiment axes: `cpu_mask`
    (hex) + `cpu_strict` = pinning; `cache_type_k`/`cache_type_v` (e.g. "q8_0") = quantized
    KV-cache; `ubatch_size` (-ub) = prefill micro-batch (bigger tiles for the GEMM kernels);
    `flash_attn` = "on"/"off"/"auto". NOTE: quantized KV-cache should run with flash_attn="on"
    — without FA the cache is dequantized every attention step and can be slower than f16.
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
    return cmd


# --- derivations --------------------------------------------------------------


def ttft_seconds(prefill_tok_s: float, prompt_tokens: int) -> float:
    """Derive time-to-first-token from prefill throughput.

    TTFT ≈ prompt_tokens / prefill_throughput (the prompt must be processed before the
    first token is emitted). Returns +inf when throughput is non-positive.
    """
    if prefill_tok_s <= 0:
        return float("inf")
    return prompt_tokens / prefill_tok_s


# --- parsing ------------------------------------------------------------------


def parse_rows(data: str | list) -> list[dict]:
    """Parse `llama-bench -o json` output into a list of row dicts."""
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
    """Convert llama-bench rows into BenchPoints, deriving TTFT for prefill rows."""
    points: list[BenchPoint] = []
    for row in rows:
        kind = row_kind(row)
        if kind == "mixed":
            continue  # skip rows that are neither a clean pp nor tg test
        n_prompt = int(row.get(F_N_PROMPT, 0) or 0)
        n_gen = int(row.get(F_N_GEN, 0) or 0)
        avg_ts = float(row.get(F_AVG_TS, 0.0) or 0.0)
        std_ts = float(row.get(F_STD_TS, 0.0) or 0.0)
        threads = int(row.get(F_THREADS, 0) or 0)
        ttft = ttft_seconds(avg_ts, n_prompt) if kind == "pp" else None
        points.append(
            BenchPoint(
                kind=kind,
                n_prompt=n_prompt,
                n_gen=n_gen,
                throughput_tok_s=avg_ts,
                throughput_stddev=std_ts,
                ttft_s=ttft,
                threads=threads,
            )
        )
    return points


def _meta_from_rows(rows: list[dict]) -> tuple[EngineInfo, ModelInfo, str, int]:
    """Pull engine/model/cpu metadata (constant across rows) from the first row."""
    first = rows[0] if rows else {}
    build_number = first.get(F_BUILD_NUMBER)
    engine = EngineInfo(
        build_number=int(build_number) if build_number not in (None, "") else None,
        build_commit=str(first.get(F_BUILD_COMMIT, "")),
        backends=str(first.get(F_BACKENDS, "")),
    )
    model = ModelInfo(
        id="",  # filled by caller (config id, not in llama-bench output)
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
    """Linux transparent-hugepage mode, e.g. 'madvise' — captured as run evidence.

    AWS's Graviton perf runbook recommends THP always/madvise for large weight working sets;
    recording the actual state makes runs comparable. Returns "" off-Linux/unreadable.
    """
    try:
        text = Path("/sys/kernel/mm/transparent_hugepage/enabled").read_text(encoding="ascii")
    except OSError:
        return ""
    for tok in text.split():
        if tok.startswith("[") and tok.endswith("]"):
            return tok[1:-1]
    return text.strip()


# --- the sweep ----------------------------------------------------------------


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
    timeout: float = 3600.0,
) -> SweepResult:
    """Run the prefill/generation sweep and return a structured result.

    Raises BenchError on a failed run. Callers should have already verified `bench_bin`
    exists (off-Arm / no-binary is a clean CLI skip, not an exception).
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
    )
    returncode, stdout, stderr, peak = run_with_peak_rss(cmd, timeout=timeout)
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

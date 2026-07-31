"""Concurrency / server-throughput axis via `llama-batched-bench`.

Everything else in the harness is single-stream, but the pitch targets agentic/RAG serving —
a CONCURRENT workload. `llama-batched-bench` (shipped in every llama.cpp build, incl. the
prebuilt archives `setup-engine` fetches) measures throughput at parallel levels: it prefills
`-npp` prompt tokens and generates `-ntg` tokens for `-npl` parallel sequences.

Flags verified against the real b9873 binary (2026-07-30): `-npp n0,n1,...`, `-ntg`, `-npl`,
and `--output-format {md,jsonl}` — we use **jsonl** (one JSON object per completed run).
TODO(confirm): the exact jsonl field names on the box; parsing below is tolerant (accepts the
known aliases, keeps every raw row verbatim in the saved JSON so nothing is lost even if a
key is missed).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path


class ThroughputError(RuntimeError):
    """llama-batched-bench failed (non-zero exit / unparseable output)."""


def build_batched_bench_command(
    bench_bin: Path,
    model_path: Path,
    *,
    npp: int = 2048,
    ntg: int = 32,
    levels: list[int] | None = None,
    threads: int | None = None,
) -> list[str]:
    """Construct a `llama-batched-bench` invocation for a parallel-level sweep.

    `-c` (context) must hold every parallel sequence: max(level) * (npp + ntg), with margin.
    """
    levels = levels or [1, 2, 4, 8]
    ctx = max(levels) * (npp + ntg) + 256
    cmd = [
        str(bench_bin),
        "-m",
        str(model_path),
        "-c",
        str(ctx),
        "-npp",
        str(npp),
        "-ntg",
        str(ntg),
        "-npl",
        ",".join(str(x) for x in levels),
        "--output-format",
        "jsonl",
    ]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


@dataclass
class ThroughputPoint:
    """One parallel level's result. speed = total tokens/sec across all sequences."""

    parallel: int
    pp: int
    tg: int
    speed_pp: float  # prompt-processing tokens/sec (aggregate)
    speed_tg: float  # generation tokens/sec (aggregate)
    speed: float  # overall tokens/sec (aggregate)
    raw: dict = field(default_factory=dict)


def _num(row: dict, *keys, default=0.0) -> float:
    for k in keys:
        if row.get(k) is not None:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                continue
    return default


def parse_jsonl(text: str) -> list[ThroughputPoint]:
    """Parse `--output-format jsonl` lines into points (tolerant key mapping, raw preserved)."""
    points: list[ThroughputPoint] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        points.append(
            ThroughputPoint(
                parallel=int(_num(row, "pl", "n_pl", "npl", "batch", default=0)),
                pp=int(_num(row, "pp", "n_pp", "npp", default=0)),
                tg=int(_num(row, "tg", "n_tg", "ntg", default=0)),
                speed_pp=_num(row, "speed_pp", "s_pp", "pp_speed"),
                speed_tg=_num(row, "speed_tg", "s_tg", "tg_speed"),
                speed=_num(row, "speed", "s", "speed_total"),
                raw=row,
            )
        )
    return points


@dataclass
class ThroughputResult:
    """A parallel-level sweep. Saved to bench/results/throughput_*.json."""

    timestamp: str
    model: str
    variant: str
    npp: int
    ntg: int
    threads: int | None
    points: list[ThroughputPoint] = field(default_factory=list)
    kind: str = "batched-throughput"
    notes: str = "llama-batched-bench --output-format jsonl; speeds are aggregate tok/s"

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def run_sweep(
    *,
    bench_bin: Path,
    model_path: Path,
    npp: int,
    ntg: int,
    levels: list[int],
    threads: int | None,
    timeout: float = 3600.0,
) -> list[ThroughputPoint]:
    """Execute the parallel-level sweep. Raises ThroughputError on failure."""
    cmd = build_batched_bench_command(
        bench_bin, model_path, npp=npp, ntg=ntg, levels=levels, threads=threads
    )
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL
    )
    if proc.returncode != 0:
        raise ThroughputError(
            f"llama-batched-bench exited {proc.returncode}\n{proc.stderr.strip()[-1200:]}"
        )
    points = parse_jsonl(proc.stdout)
    if not points:
        raise ThroughputError(
            "no jsonl rows parsed from llama-batched-bench output "
            f"(first 400 chars): {proc.stdout[:400]}"
        )
    return points

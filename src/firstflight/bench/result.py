"""Structured benchmark result schema + JSON (de)serialization.

A `SweepResult` is the unit written to `bench/results/*.json` and read by the report
generator. It is intentionally self-describing (schema_version, host, engine,
model, config) so a committed result is fully interpretable without rerunning anything.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import __version__

SCHEMA_VERSION = 1


@dataclass
class BenchPoint:
    """One measured point: a prefill (pp) or generation (tg) test at a given size."""

    kind: str  # "pp" (prefill / prompt-processing) or "tg" (token generation)
    n_prompt: int
    n_gen: int
    throughput_tok_s: float
    throughput_stddev: float
    ttft_s: float | None  # derived for pp points (n_prompt / throughput); None for tg
    threads: int


@dataclass
class HostInfo:
    platform: str
    arch: str
    cpu_count: int
    is_arm_linux: bool
    cpu_info: str = ""
    kleidiai: bool | None = None  # detected KleidiAI-active (experiments); None until known
    thp: str = ""  # Linux transparent-hugepage mode captured at run time ("" off-Linux)


@dataclass
class EngineInfo:
    name: str = "llama.cpp"
    build_number: int | None = None
    build_commit: str = ""
    backends: str = ""
    binary: str = ""


@dataclass
class ModelInfo:
    id: str
    variant: str
    filename: str = ""
    size_bytes: int | None = None
    n_params: int | None = None


@dataclass
class SweepResult:
    label: str  # "baseline" / "optimized" / free-form, for before-vs-after experiments
    timestamp: str  # ISO-8601 UTC
    workload: str
    host: HostInfo
    engine: EngineInfo
    model: ModelInfo
    threads: int
    repetitions: int
    n_gen: int
    peak_rss_bytes: int | None
    points: list[BenchPoint] = field(default_factory=list)
    quality: dict | None = None  # optional quality-eval summary: accuracy, n_correct...
    experiment: str = ""  # which experiment produced this result ("" = ad-hoc bench/run)
    schema_version: int = SCHEMA_VERSION
    tool: str = "firstflight"
    tool_version: str = __version__

    # --- convenience views -------------------------------------------------

    @property
    def prefill_points(self) -> list[BenchPoint]:
        return [p for p in self.points if p.kind == "pp"]

    @property
    def gen_points(self) -> list[BenchPoint]:
        return [p for p in self.points if p.kind == "tg"]

    # --- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> SweepResult:
        return cls(
            label=d["label"],
            timestamp=d["timestamp"],
            workload=d["workload"],
            host=HostInfo(**d["host"]),
            engine=EngineInfo(**d["engine"]),
            model=ModelInfo(**d["model"]),
            threads=d["threads"],
            repetitions=d["repetitions"],
            n_gen=d["n_gen"],
            peak_rss_bytes=d.get("peak_rss_bytes"),
            points=[BenchPoint(**p) for p in d.get("points", [])],
            quality=d.get("quality"),
            experiment=str(d.get("experiment", "") or ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            tool=d.get("tool", "firstflight"),
            tool_version=d.get("tool_version", __version__),
        )

    @classmethod
    def load_json(cls, path: Path) -> SweepResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def result_filename(model_id: str, variant: str, label: str, timestamp: str) -> str:
    """Stable, sortable filename for a result (timestamp first by caller's choice)."""
    safe_ts = timestamp.replace(":", "").replace("-", "").replace(".", "")
    safe = f"{model_id}_{variant}_{label}".replace("/", "-").replace(" ", "-")
    return f"{safe}_{safe_ts}.json"

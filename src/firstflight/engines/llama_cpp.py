"""llama.cpp engine adapter.

Binary discovery plus one generation run (`run_once`) for the smoke test. The prefill/TTFT
sweep via `llama-bench` lives in `firstflight.bench.prefill` and reuses
`find_binary("bench")` from here.

Upstream: https://github.com/ggml-org/llama.cpp (org is `ggml-org`, formerly `ggerganov`)
Verified flags (2026-06-26): llama-bench prefill=`-p/--n-prompt`, gen=`-n/--n-gen`,
JSON output=`-o json`. CPU build: `cmake -B build && cmake --build build --config Release`.
KleidiAI: add `-DGGML_CPU_KLEIDIAI=ON` (accelerates Q4_0/Q8_0 weights).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Binary name candidates by role. `llama-cli` is the modern CLI, `main` the legacy name.
# An explicit LLAMA_CPP_BIN dir accepts the legacy names; the bare-PATH search is strict
# (modern names only) so `main` can't match something like Windows `main.CPL`.
CLI_NAMES = ["llama-cli", "main", "llama"]
BENCH_NAMES = ["llama-bench"]
SERVER_NAMES = ["llama-server"]
BATCHED_NAMES = ["llama-batched-bench"]
PERPLEXITY_NAMES = ["llama-perplexity"]
KIND_NAMES = {
    "cli": CLI_NAMES,
    "bench": BENCH_NAMES,
    "server": SERVER_NAMES,
    "batched": BATCHED_NAMES,
    "perplexity": PERPLEXITY_NAMES,
}
PATH_NAMES = {
    "cli": ["llama-cli"],
    "bench": ["llama-bench"],
    "server": ["llama-server"],
    "batched": ["llama-batched-bench"],
    "perplexity": ["llama-perplexity"],
}


def _filenames(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        out.extend([n, f"{n}.exe"])  # .exe for Windows release zips
    return out


def _search_dir(d: Path, filenames: list[str]) -> Path | None:
    for fn in filenames:
        hits = sorted(d.rglob(fn))
        if hits:
            return hits[0]
    return None


def find_binary(kind: str = "cli", env_value: str | None = None) -> Path | None:
    """Locate a llama.cpp binary.

    Search order:
      1. $LLAMA_CPP_BIN, either a binary path or a directory (e.g. an extracted release
         tarball). Directories are searched recursively.
      2. The local `./engine` dir populated by `firstflight setup-engine`.
      3. PATH.

    `kind` is "cli" (generation), "bench" (llama-bench), or "server" (llama-server).
    None if not found, so callers can degrade off-Arm / without a build.
    """
    names = KIND_NAMES.get(kind, BENCH_NAMES)
    filenames = _filenames(names)

    env = env_value if env_value is not None else os.environ.get("LLAMA_CPP_BIN")
    if env:
        p = Path(env)
        if p.is_file():
            if p.name in filenames or p.stem in names:
                return p
            return _search_dir(p.parent, filenames)
        if p.is_dir():
            return _search_dir(p, filenames)
        return None  # explicit override that doesn't exist -> don't silently fall through

    from ..util import engine_dir  # local import to avoid cycles

    local = engine_dir()
    if local.is_dir():
        found = _search_dir(local, filenames)
        if found:
            return found

    for fn in _filenames(PATH_NAMES.get(kind, [])):
        found = shutil.which(fn)
        if found:
            return Path(found)
    return None


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    cmd: list[str]

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def build_run_cmd(
    cli_bin: Path,
    model_path: Path,
    prompt: str,
    n_predict: int = 24,
    threads: int | None = None,
    seed: int = 42,
    temp: float | None = None,
) -> list[str]:
    """Single-generation `llama-cli` command (used by the smoke test).

    Flags verified against the real b9873 `llama-cli --help` (2026-07-07): `-t/--threads`,
    `-n/--n-predict`, `-s/--seed`, `-no-cnv/--no-conversation`, `--no-display-prompt`,
    `--temp`. temp=0.0 is greedy; the quality probe uses it so it measures the model, not
    the sampler.
    """
    threads = threads or os.cpu_count() or 4
    cmd = [
        str(cli_bin),
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "-t",
        str(threads),
        "-s",
        str(seed),
        "-no-cnv",  # single-shot completion, no chat loop
        "--no-display-prompt",  # print only the completion
    ]
    if temp is not None:
        cmd += ["--temp", str(temp)]
    return cmd


# Printed at model load when KleidiAI kernels are active (verified 2026-06-26, docs/build.md):
#   load_tensors: CPU_KLEIDIAI model buffer size = ... MiB
KLEIDIAI_MARKER = "CPU_KLEIDIAI"
# ggml's own aarch64 Q4_0 repack path (GGML_CPU_REPACK, ON by default) loads into its own
# buffer. That is the rung below KleidiAI, and telling them apart is the whole ladder.
REPACK_MARKER = "CPU_REPACK"

# Any extra CPU buffer type (CPU_KLEIDIAI, the aarch64 repack buffer, ...) plus its size.
# Captured as printed, not matched against a hardcoded list, so a renamed upstream buffer
# type still gets recorded.
_BUFFER_RE = re.compile(r"load_tensors:\s+(CPU\w*)\s+model buffer size\s*=\s*([\d.]+)\s*MiB")
_SYSINFO_RE = re.compile(r"system_info:\s*(.+)")


@dataclass(frozen=True)
class BackendProbe:
    """Evidence of which CPU kernel path actually loaded, from a real 1-token run."""

    kleidiai: bool | None  # None = probe run failed
    system_info: str = ""  # ggml's feature line: NEON / DOTPROD / MATMUL_INT8 / SVE ...
    kernel_buffer: str = ""  # e.g. "CPU_KLEIDIAI 168.2 MiB": which weight buffer loaded
    cpu_features: str = ""  # /proc/cpuinfo Features line (Linux), "" elsewhere
    kernel_tier: str = ""  # highest ISA tier the run could reach: I8MM / DOTPROD / SVE2 / NEON
    repack: bool | None = None  # ggml's own aarch64 repack buffer loaded (the rung below KleidiAI)


# KleidiAI picks kernels in this order (Arm's llama.cpp integration docs): SME2 -> I8MM ->
# DotProd. SME2 is client-silicon only, so on a Neoverse server the top reachable tier is
# I8MM. Reporting the tier beats a yes/no: it names which kernels the speedup came from.
_TIER_ORDER = [
    ("SME2", ("SME2",)),
    ("I8MM", ("MATMUL_INT8", "I8MM")),
    ("DOTPROD", ("DOTPROD",)),
    ("SVE2", ("SVE2",)),
    ("NEON", ("NEON",)),
]


def kernel_tier(system_info: str, cpu_features: str = "") -> str:
    """Highest kernel tier the reported ISA flags can reach. "" when nothing is recognized."""
    hay = f"{system_info} {cpu_features}".upper()
    for tier, needles in _TIER_ORDER:
        for n in needles:
            # ggml prints "MATMUL_INT8 = 1"; /proc/cpuinfo prints bare tokens like "i8mm"
            if f"{n} = 1" in hay or f" {n} " in f" {hay} ":
                return tier
    return ""


def probe_backend(
    cli_bin: Path, model_path: Path, threads: int | None = None, timeout: float = 300.0
) -> BackendProbe:
    """Run 1 token and collect kernel-path evidence from the load log.

    The Arm-leverage claim shouldn't rest on a build flag: this records what the binary
    itself printed (system_info flags + which CPU buffer type the weights loaded into).
    """
    try:
        res = run_once(
            cli_bin, model_path, prompt="hi", n_predict=1, threads=threads, timeout=timeout
        )
    except Exception:  # noqa: BLE001 - probing is best-effort
        return BackendProbe(kleidiai=None)
    if not res.ok:
        return BackendProbe(kleidiai=None)
    out = res.stderr + "\n" + res.stdout

    m = _SYSINFO_RE.search(out)
    system_info = " ".join(m.group(1).split())[:400] if m else ""

    # Prefer the accelerated buffer (anything beyond plain "CPU") if one loaded.
    buffers = _BUFFER_RE.findall(out)
    kernel_buffer = ""
    for name, mib in buffers:
        if name != "CPU":
            kernel_buffer = f"{name} {mib} MiB"
            break
    if not kernel_buffer and buffers:
        kernel_buffer = f"{buffers[0][0]} {buffers[0][1]} MiB"

    cpu_features = ""
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("features"):
                    cpu_features = " ".join(line.split(":", 1)[1].split())[:400]
                    break
    except OSError:
        pass

    return BackendProbe(
        kleidiai=KLEIDIAI_MARKER in out,
        system_info=system_info,
        kernel_buffer=kernel_buffer,
        cpu_features=cpu_features,
        kernel_tier=kernel_tier(system_info, cpu_features),
        repack=REPACK_MARKER in out,
    )


def detect_kleidiai(
    cli_bin: Path, model_path: Path, threads: int | None = None, timeout: float = 300.0
) -> bool | None:
    """Whether KleidiAI kernels are actually active for this binary+model."""
    return probe_backend(cli_bin, model_path, threads=threads, timeout=timeout).kleidiai


def run_once(
    cli_bin: Path,
    model_path: Path,
    prompt: str = "The Arm Neoverse CPU is",
    n_predict: int = 24,
    threads: int | None = None,
    seed: int = 42,
    temp: float | None = None,
    timeout: float = 300.0,
) -> RunResult:
    """Run one generation, capture output + wall time (smoke test).

    stdin is DEVNULL so llama-cli can't wait for interactive input: on the real b9873
    Windows binary an inherited console stdin hangs the run.
    """
    cmd = build_run_cmd(cli_bin, model_path, prompt, n_predict, threads, seed, temp)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        # A hung generation becomes a failed RunResult. Every caller already handles the
        # non-ok path (smoke prints the error, probe scores the item wrong, detect -> None).
        return RunResult(
            returncode=-1,
            stdout="",
            stderr=f"llama-cli timed out after {timeout:.0f}s",
            wall_s=time.perf_counter() - start,
            cmd=cmd,
        )
    wall = time.perf_counter() - start
    return RunResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        wall_s=wall,
        cmd=cmd,
    )

"""llama.cpp engine adapter.

Implements binary discovery + a single generation run (`run_once`) for the smoke test.
The prefill/TTFT benchmark sweep via `llama-bench` lives in `firstflight.bench.prefill`
and reuses `find_binary("bench")` from here.

Upstream: https://github.com/ggml-org/llama.cpp  (org is now `ggml-org`, formerly `ggerganov`)
Verified flags (2026-06-26): llama-bench prefill=`-p/--n-prompt`, gen=`-n/--n-gen`,
JSON output=`-o json`. CPU build: `cmake -B build && cmake --build build --config Release`.
KleidiAI: add `-DGGML_CPU_KLEIDIAI=ON` (accelerates Q4_0/Q8_0 weights).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Binary name candidates by role. `llama-cli` is the modern CLI; `main` is the legacy name.
# When searching an explicit LLAMA_CPP_BIN directory we accept the legacy names too, but the
# bare-PATH search is strict (exact modern names only) so a generic name like `main` can't
# match an unrelated system file (e.g. Windows `main.CPL`).
CLI_NAMES = ["llama-cli", "main", "llama"]
BENCH_NAMES = ["llama-bench"]
SERVER_NAMES = ["llama-server"]
BATCHED_NAMES = ["llama-batched-bench"]
KIND_NAMES = {
    "cli": CLI_NAMES,
    "bench": BENCH_NAMES,
    "server": SERVER_NAMES,
    "batched": BATCHED_NAMES,
}
PATH_NAMES = {
    "cli": ["llama-cli"],
    "bench": ["llama-bench"],
    "server": ["llama-server"],
    "batched": ["llama-batched-bench"],
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
      1. $LLAMA_CPP_BIN — may be a binary path OR a directory (e.g. an extracted release
         tarball); directories are searched recursively.
      2. The local `./engine` dir populated by `firstflight setup-engine`.
      3. PATH (via `shutil.which`).

    `kind` is "cli" (generation), "bench" (llama-bench), or "server" (llama-server).
    Returns None if not found, so callers can degrade gracefully off-Arm / without a build.
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
    """Construct a single-generation `llama-cli` command (used by the smoke test).

    All flags verified against the real b9873 `llama-cli --help` (2026-07-07):
    `-t/--threads`, `-n/--n-predict`, `-s/--seed`, `-no-cnv/--no-conversation`,
    `--no-display-prompt`, `--temp`. Pass `temp=0.0` for greedy (deterministic) decoding —
    the quality probe uses this so it measures the model, not the sampler.
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


def detect_kleidiai(
    cli_bin: Path, model_path: Path, threads: int | None = None, timeout: float = 300.0
) -> bool | None:
    """Whether KleidiAI kernels are ACTIVE for this binary+model (proof, not assumption).

    Runs a 1-token generation and scans the load log for the CPU_KLEIDIAI buffer line.
    Returns True/False, or None if the probe run itself failed.
    """
    try:
        res = run_once(
            cli_bin, model_path, prompt="hi", n_predict=1, threads=threads, timeout=timeout
        )
    except Exception:  # noqa: BLE001 - detection is best-effort
        return None
    if not res.ok:
        return None
    return KLEIDIAI_MARKER in res.stderr or KLEIDIAI_MARKER in res.stdout


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
    """Run a single generation and capture output + wall time (smoke test).

    stdin is closed (DEVNULL): llama-cli must never wait for interactive input — verified on
    the real b9873 Windows binary, where an inherited console stdin hangs the run.
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
        # A hung generation becomes a clean failed RunResult — every caller already handles
        # the non-ok path (smoke prints the error, probe scores the item wrong, detect->None).
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

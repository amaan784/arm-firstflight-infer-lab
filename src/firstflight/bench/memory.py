"""Peak resident-memory capture for a benchmarked subprocess.

`run_with_peak_rss()` isolates peak RSS to the SPECIFIC child via GNU `/usr/bin/time -v` when
available (correct even when many configs run in one `firstflight` process — the Phase-4
experiment/autotune case). It falls back to process-wide `resource.getrusage(RUSAGE_CHILDREN)`
(a running max — accurate for a single run) and to None where neither is available (Windows).

`ru_maxrss` units differ by platform (Linux: KiB; macOS: bytes) — normalized to bytes here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import resource  # POSIX only
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

_MAXRSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def parse_time_v_maxrss(stderr: str) -> int | None:
    """Pull peak RSS (bytes) from GNU `/usr/bin/time -v` stderr ('... (kbytes): N')."""
    m = _MAXRSS_RE.search(stderr or "")
    return int(m.group(1)) * 1024 if m else None


def _gnu_time() -> str | None:
    """Locate GNU `time` (NOT the shell builtin). Only Linux `/usr/bin/time` supports `-v`;
    macOS `/usr/bin/time` is BSD, so there we only trust a Homebrew `gtime`."""
    if sys.platform.startswith("linux"):
        for cand in ("/usr/bin/time", "/usr/local/bin/time"):
            if Path(cand).exists():
                return cand
        return shutil.which("time")
    return shutil.which("gtime")


def peak_child_rss_bytes() -> int | None:
    """Peak RSS (bytes) of waited-for child processes, or None when unavailable.

    NOTE: with RUSAGE_CHILDREN this is a running max across ALL children of this process and
    never decreases — correct for one subprocess, but for multi-config runs prefer
    `run_with_peak_rss` (per-child isolation via GNU time).
    """
    if resource is None:
        return None
    maxrss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if maxrss <= 0:
        return None
    if sys.platform == "darwin":
        return int(maxrss)  # macOS reports bytes
    return int(maxrss) * 1024  # Linux reports kibibytes


def run_with_peak_rss(cmd: list, *, timeout: float) -> tuple[int, str, str, int | None]:
    """Run `cmd`, returning (returncode, stdout, stderr, peak_rss_bytes).

    Peak RSS is isolated to this child via GNU `/usr/bin/time -v` when available, else falls
    back to process-wide RUSAGE_CHILDREN, else None. The child's stdout is untouched (so JSON
    parsing is unaffected); `time`'s report is appended to stderr.
    """
    argv = [str(c) for c in cmd]
    gtime = _gnu_time()
    # stdin=DEVNULL: benchmark children must never block on interactive input.
    if gtime:
        proc = subprocess.run(
            [gtime, "-v", *argv],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
        peak = parse_time_v_maxrss(proc.stderr)
        if peak is not None:
            return proc.returncode, proc.stdout, proc.stderr, peak
        # `time` wasn't GNU / didn't emit the line — fall through to the rusage fallback.
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL
    )
    return proc.returncode, proc.stdout, proc.stderr, peak_child_rss_bytes()

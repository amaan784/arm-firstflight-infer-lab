"""Peak resident memory of a benchmarked subprocess.

run_with_peak_rss() pins peak RSS to the specific child via GNU `/usr/bin/time -v` when it's
there, which matters once many configs run inside one firstflight process (Phase-4
experiment/autotune). Falls back to process-wide resource.getrusage(RUSAGE_CHILDREN), a
running max that's only right for a single run, then to None (Windows).

ru_maxrss units differ by platform: Linux KiB, macOS bytes. Normalized to bytes here.
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
    """Peak RSS in bytes from GNU `time -v` stderr ('... (kbytes): N'). None if absent."""
    m = _MAXRSS_RE.search(stderr or "")
    return int(m.group(1)) * 1024 if m else None


def _gnu_time() -> str | None:
    """GNU `time`, not the shell builtin. Only Linux /usr/bin/time supports -v; macOS
    /usr/bin/time is BSD, so there only a Homebrew `gtime` counts."""
    if sys.platform.startswith("linux"):
        for cand in ("/usr/bin/time", "/usr/local/bin/time"):
            if Path(cand).exists():
                return cand
        return shutil.which("time")
    return shutil.which("gtime")


def peak_child_rss_bytes() -> int | None:
    """Peak RSS in bytes of waited-for children, None when unavailable.

    RUSAGE_CHILDREN is a running max over ALL children of this process and never decreases.
    Fine for one subprocess; for multi-config runs use run_with_peak_rss instead.
    """
    if resource is None:
        return None
    maxrss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if maxrss <= 0:
        return None
    if sys.platform == "darwin":
        return int(maxrss)  # macOS: bytes
    return int(maxrss) * 1024  # Linux: KiB


def run_with_peak_rss(cmd: list, *, timeout: float) -> tuple[int, str, str, int | None]:
    """Run `cmd` -> (returncode, stdout, stderr, peak_rss_bytes).

    Peak RSS comes from GNU `time -v` when available, else RUSAGE_CHILDREN, else None. The
    child's stdout is left alone so JSON parsing still works; `time` writes to stderr.
    """
    argv = [str(c) for c in cmd]
    gtime = _gnu_time()
    # stdin=DEVNULL so a benchmark child can never block waiting on input.
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
        # not GNU time, or it didn't emit the line. fall through to rusage.
    proc = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False, stdin=subprocess.DEVNULL
    )
    return proc.returncode, proc.stdout, proc.stderr, peak_child_rss_bytes()

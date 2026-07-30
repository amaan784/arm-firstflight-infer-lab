"""Small shared helpers: paths, platform detection, console output.

Kept dependency-light (only `rich`, a core dep) so the smoke path stays portable.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape as _markup_escape


def _init_console() -> Console:
    # Prefer UTF-8 where the stream allows it; on legacy Windows code pages we additionally
    # keep console output ASCII-safe (see `ascii_safe`) so nothing ever crashes on encode.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    return Console()


console = _init_console()


def ascii_safe(text: str) -> str:
    """Replace characters the local console can't encode — portable status output."""
    return text.encode("ascii", "replace").decode("ascii")


def safe(text: str) -> str:
    """ASCII-safe AND rich-markup-escaped — use for any DYNAMIC text (exceptions, model
    output, filenames) interpolated into `console.print`, so `[...]` in the text isn't eaten
    as a markup tag.
    """
    return _markup_escape(ascii_safe(text))


# --- platform detection -------------------------------------------------------

_ARM_MACHINES = {"aarch64", "arm64"}


def machine() -> str:
    """Normalized CPU architecture string, e.g. 'aarch64', 'x86_64'."""
    return platform.machine().lower()


def platform_system() -> str:
    """Normalized OS name: 'windows', 'linux', or 'darwin'."""
    return platform.system().lower()


def is_arm64() -> bool:
    """True on Arm (aarch64/arm64) hosts."""
    return machine() in _ARM_MACHINES


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def is_arm_linux() -> bool:
    """The environment where Arm-only tooling (Performix, KleidiAI) is expected."""
    return is_arm64() and is_linux()


def platform_tag() -> str:
    return f"{platform.system()}-{machine()}"


# --- paths --------------------------------------------------------------------


def repo_root() -> Path:
    """Best-effort repository root (the dir containing `configs/` and `pyproject.toml`).

    Works for an editable install (src/firstflight/util.py -> repo root) and falls back
    to the current working directory otherwise.
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir() and (parent / "pyproject.toml").exists():
            return parent
    return here.parents[2] if len(here.parents) >= 3 else Path.cwd()


def config_dir() -> Path:
    """Directory holding models.yaml / instances.yaml / workloads.yaml.

    Resolution order: $FIRSTFLIGHT_CONFIG_DIR -> the repo checkout's configs/ -> the copies
    packaged inside the wheel (firstflight/configs, shipped via force-include) so a plain
    non-editable `pip install` works outside a checkout.
    """
    env = os.environ.get("FIRSTFLIGHT_CONFIG_DIR")
    if env:
        return Path(env)
    repo = repo_root() / "configs"
    if repo.is_dir():
        return repo
    try:
        from importlib.resources import files

        packaged = Path(str(files("firstflight") / "configs"))
        if packaged.is_dir():
            return packaged
    except Exception:  # noqa: BLE001 - fall through to the repo path (clear error downstream)
        pass
    return repo


def model_dir() -> Path:
    """Where downloaded GGUFs are cached. Override with FIRSTFLIGHT_MODEL_DIR."""
    env = os.environ.get("FIRSTFLIGHT_MODEL_DIR")
    path = Path(env) if env else repo_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def engine_dir() -> Path:
    """Where `firstflight setup-engine` places prebuilt llama.cpp binaries (gitignored).

    `find_binary` searches here automatically, after $LLAMA_CPP_BIN and before PATH.
    Override with FIRSTFLIGHT_ENGINE_DIR.
    """
    env = os.environ.get("FIRSTFLIGHT_ENGINE_DIR")
    return Path(env) if env else repo_root() / "engine"


def results_dir() -> Path:
    path = repo_root() / "bench" / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reports_dir() -> Path:
    path = repo_root() / "bench" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- formatting ---------------------------------------------------------------


def bytes_human(n: float) -> str:
    """Human-readable byte size, e.g. 491400032 -> '468.6 MiB'."""
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < step or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= step
    return f"{n:.1f} TiB"

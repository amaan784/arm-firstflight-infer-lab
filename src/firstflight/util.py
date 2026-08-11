"""Shared helpers: paths, platform detection, console output.

Only depends on `rich` (a core dep) so the smoke path stays portable.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape as _markup_escape


def _init_console() -> Console:
    # UTF-8 where the stream allows it. Legacy Windows code pages also go through `ascii_safe`,
    # so an encode error can't crash us.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:
            pass
    return Console()


console = _init_console()


def ascii_safe(text: str) -> str:
    """Replace characters the local console can't encode."""
    return text.encode("ascii", "replace").decode("ascii")


def safe(text: str) -> str:
    """ASCII-safe and rich-markup-escaped. For any dynamic text (exceptions, model output,
    filenames) going into `console.print`, so `[...]` isn't eaten as a markup tag.
    """
    return _markup_escape(ascii_safe(text))


# --- platform detection -------------------------------------------------------

_ARM_MACHINES = {"aarch64", "arm64"}


def machine() -> str:
    """CPU architecture, lowercased: 'aarch64', 'x86_64'."""
    return platform.machine().lower()


def platform_system() -> str:
    """OS name, lowercased: 'windows', 'linux', 'darwin'."""
    return platform.system().lower()


def is_arm64() -> bool:
    """True on aarch64/arm64 hosts."""
    return machine() in _ARM_MACHINES


def is_linux() -> bool:
    return platform.system().lower() == "linux"


def is_arm_linux() -> bool:
    """Where the Arm-only tooling (Performix, KleidiAI) is expected to work."""
    return is_arm64() and is_linux()


def platform_tag() -> str:
    return f"{platform.system()}-{machine()}"


# --- paths --------------------------------------------------------------------


def repo_root() -> Path:
    """Working root: the checkout when there is one, else cwd.

    Order: $FIRSTFLIGHT_WORK_DIR -> the dir holding configs/ + pyproject.toml (editable
    install) -> cwd. The old fallback resolved INSIDE site-packages on a non-editable
    install, so models/reports landed invisibly inside the venv and died with it.
    """
    env = os.environ.get("FIRSTFLIGHT_WORK_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs").is_dir() and (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def config_dir() -> Path:
    """Directory holding models.yaml / instances.yaml / workloads.yaml.

    Order: $FIRSTFLIGHT_CONFIG_DIR -> the checkout's configs/ -> the copies inside the wheel
    (firstflight/configs, shipped via force-include), so a non-editable `pip install` works
    outside a checkout.
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
    except Exception:  # noqa: BLE001 - fall back to the repo path; error surfaces downstream
        pass
    return repo


def model_dir() -> Path:
    """Where downloaded GGUFs are cached. Override with FIRSTFLIGHT_MODEL_DIR."""
    env = os.environ.get("FIRSTFLIGHT_MODEL_DIR")
    path = Path(env) if env else repo_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def engine_dir() -> Path:
    """Where `firstflight setup-engine` puts prebuilt llama.cpp binaries (gitignored).

    `find_binary` searches here after $LLAMA_CPP_BIN and before PATH. Override with
    FIRSTFLIGHT_ENGINE_DIR.
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

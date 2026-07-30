"""Fetch a prebuilt llama.cpp release for the current platform (zero-compile bootstrap).

`firstflight setup-engine` downloads the official release asset for this OS/arch from
github.com/ggml-org/llama.cpp, extracts it into `./engine` (gitignored), and `find_binary`
picks it up automatically — so `firstflight smoke` runs REAL inference on any machine
(Windows/Linux/macOS, x64/arm64) without a compiler.

Asset filename patterns follow the release naming convention (pattern verified 2026-06-26 for
ubuntu-arm64; the exact per-platform names are re-verified against the live Releases page).
Note: prebuilt assets are plain CPU builds — the KleidiAI comparison still uses the
`-DGGML_CPU_KLEIDIAI=ON` source build (see arm-bench.yml / scripts/setup_arm_vm.sh).
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from ..util import bytes_human, console, engine_dir, machine, platform_system

# Pinned release tag (floats; bump deliberately). Keep in sync with arm-bench.yml/Dockerfile.
# b9873 (2026-07-04) verified live 2026-07-05: all ASSET_PATTERNS below exist on this release,
# and the win-cpu-x64 zip ships llama-cli.exe / llama-bench.exe at the archive root.
DEFAULT_TAG = "b9873"

# (system, arch) -> asset filename pattern. {tag} is substituted.
ASSET_PATTERNS: dict[tuple[str, str], str] = {
    ("windows", "amd64"): "llama-{tag}-bin-win-cpu-x64.zip",
    ("windows", "arm64"): "llama-{tag}-bin-win-cpu-arm64.zip",
    ("linux", "x86_64"): "llama-{tag}-bin-ubuntu-x64.tar.gz",
    ("linux", "aarch64"): "llama-{tag}-bin-ubuntu-arm64.tar.gz",  # verified 2026-06-26
    ("darwin", "arm64"): "llama-{tag}-bin-macos-arm64.tar.gz",
    ("darwin", "x86_64"): "llama-{tag}-bin-macos-x64.tar.gz",
}

RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{asset}"


class EngineFetchError(RuntimeError):
    """Could not download/extract a prebuilt engine for this platform."""


def asset_for(system: str | None = None, arch: str | None = None, tag: str = DEFAULT_TAG) -> str:
    """The release asset filename for a platform. Raises EngineFetchError if unsupported."""
    system = (system or platform_system()).lower()
    arch = (arch or machine()).lower()
    # normalize common aliases
    arch = {"x86_64": "x86_64", "amd64": "amd64", "arm64": "arm64", "aarch64": "aarch64"}.get(
        arch, arch
    )
    if system == "windows" and arch == "x86_64":
        arch = "amd64"
    if system == "linux" and arch == "arm64":
        arch = "aarch64"
    if system == "darwin" and arch == "aarch64":
        arch = "arm64"
    pattern = ASSET_PATTERNS.get((system, arch))
    if pattern is None:
        raise EngineFetchError(
            f"no prebuilt llama.cpp asset known for {system}/{arch} — build from source instead "
            "(cmake -B build && cmake --build build --config Release)"
        )
    return pattern.format(tag=tag)


def download_url(tag: str = DEFAULT_TAG, system: str | None = None, arch: str | None = None) -> str:
    return RELEASE_URL.format(tag=tag, asset=asset_for(system, arch, tag))


def _extract(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest)  # noqa: S202 - official release archive, pinned tag


def setup_engine(
    tag: str = DEFAULT_TAG,
    dest: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download + extract the prebuilt llama.cpp for this platform. Returns the engine dir.

    Idempotent: if the dest already contains a llama-cli/llama-bench, it is reused unless
    `force`. Raises EngineFetchError on unsupported platforms or download failures.
    """
    from . import llama_cpp  # local import to avoid a cycle at module load

    dest = dest or engine_dir()
    if not force:
        existing = llama_cpp.find_binary("cli", env_value=str(dest)) if dest.exists() else None
        if existing:
            console.print(f"[green]OK[/] engine already present: {existing}")
            return dest

    asset = asset_for(tag=tag)
    url = RELEASE_URL.format(tag=tag, asset=asset)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / asset

    console.print(f"Fetching llama.cpp [bold]{tag}[/] ({asset})")
    with requests.get(url, stream=True, timeout=60, allow_redirects=True) as resp:
        if resp.status_code == 404:
            raise EngineFetchError(
                f"release asset not found: {url}\n"
                "The tag or asset name may have changed — check "
                "https://github.com/ggml-org/llama.cpp/releases and pass --tag."
            )
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with archive.open("wb") as fh, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                bar.update(len(chunk))

    console.print(f"Extracting ({bytes_human(archive.stat().st_size)})...")
    _extract(archive, dest)
    archive.unlink(missing_ok=True)

    found = llama_cpp.find_binary("cli", env_value=str(dest))
    bench = llama_cpp.find_binary("bench", env_value=str(dest))
    if not found and not bench:
        raise EngineFetchError(f"extracted {asset} but found no llama-cli/llama-bench under {dest}")
    console.print(f"[green]OK[/] engine ready: {found or bench}")
    return dest

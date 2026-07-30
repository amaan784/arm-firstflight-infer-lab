"""Download GGUF model files on demand.

Streams the Hugging Face `resolve` URL with `requests` (a core dep) so the smoke path
needs no extra packages. An optional faster/authenticated path via `huggingface_hub`
can be added behind the `[hub]` extra later.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

from .config import ModelSpec, ModelVariant
from .util import bytes_human, console, model_dir

_CHUNK = 1 << 20  # 1 MiB


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(
    spec: ModelSpec,
    variant: ModelVariant,
    dest_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Return a local path to the GGUF, downloading it if missing.

    Verifies sha256 when one is configured. Downloads atomically (`.part` then rename) so
    an interrupted run never leaves a corrupt file in the cache.
    """
    dest_dir = dest_dir or model_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / variant.file

    if target.exists() and not force:
        if variant.sha256:
            actual = _sha256(target)
            if actual == variant.sha256:
                console.print(f"[green]OK[/] cached & verified: {target.name}")
                return target
            console.print(f"[yellow]![/] checksum mismatch for {target.name}, re-downloading")
        else:
            console.print(f"[green]OK[/] cached: {target.name}")
            return target

    if not variant.url:
        raise ValueError(
            f"No download URL for {spec.id}:{variant.name}. Fill it in configs/models.yaml."
        )

    console.print(f"Downloading [bold]{spec.id}:{variant.name}[/] -> {target.name}")
    tmp = target.with_suffix(target.suffix + ".part")
    with requests.get(variant.url, stream=True, timeout=60, allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            tmp.open("wb") as fh,
            tqdm(total=total, unit="B", unit_scale=True, desc=variant.file) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
                bar.update(len(chunk))

    if variant.sha256:
        actual = _sha256(tmp)
        if actual != variant.sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {variant.file}: expected {variant.sha256}, got {actual}"
            )

    tmp.replace(target)
    console.print(f"[green]OK[/] downloaded {target.name} ({bytes_human(target.stat().st_size)})")
    return target

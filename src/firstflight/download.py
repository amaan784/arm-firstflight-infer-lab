"""Download GGUF model files on demand.

Streams the Hugging Face `resolve` URL with `requests` (a core dep) so the smoke path
needs no extra packages. A faster/authenticated `huggingface_hub` path can go behind the
`[hub]` extra later.

Downloads resume and retry. Shared CI runners get throttled by HF (observed 2-8MB/s and
outright 429s), and a 400MB transfer that restarts from zero on every hiccup never lands.
The `.part` file is kept between attempts and continued with a Range request.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests
from tqdm import tqdm

from .config import ModelSpec, ModelVariant
from .util import bytes_human, console, model_dir

_CHUNK = 1 << 20  # 1 MiB
_ATTEMPTS = 5
_BACKOFF = 5.0  # seconds, doubled per attempt


class DownloadError(RuntimeError):
    """A model download failed after exhausting retries."""


def _fetch_with_resume(url: str, tmp: Path, label: str) -> None:
    """Stream `url` into `tmp`, resuming and retrying until complete.

    Raises DownloadError once attempts are exhausted; leaves `tmp` in place so the next
    call continues rather than restarting.
    """
    last_error: Exception | None = None
    for attempt in range(1, _ATTEMPTS + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(
                url, stream=True, timeout=60, allow_redirects=True, headers=headers
            ) as resp:
                if have and resp.status_code == 200:
                    # Server ignored the Range header: start over rather than corrupt.
                    have = 0
                    tmp.unlink(missing_ok=True)
                elif have and resp.status_code != 206:
                    resp.raise_for_status()
                else:
                    resp.raise_for_status()

                total = int(resp.headers.get("content-length", 0)) + have
                mode = "ab" if have else "wb"
                with (
                    tmp.open(mode) as fh,
                    tqdm(
                        total=total or None,
                        initial=have,
                        unit="B",
                        unit_scale=True,
                        desc=label,
                    ) as bar,
                ):
                    for chunk in resp.iter_content(chunk_size=_CHUNK):
                        fh.write(chunk)
                        bar.update(len(chunk))
            return
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            got = tmp.stat().st_size if tmp.exists() else 0
            if attempt == _ATTEMPTS:
                break
            wait = _BACKOFF * (2 ** (attempt - 1))
            console.print(
                f"[yellow]download interrupted[/] ({type(exc).__name__}) at "
                f"{bytes_human(got)}; retry {attempt + 1}/{_ATTEMPTS} in {wait:.0f}s"
            )
            time.sleep(wait)
    raise DownloadError(f"{label}: giving up after {_ATTEMPTS} attempts ({last_error})")


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
    """Local path to the GGUF, downloading it if missing.

    Verifies sha256 when one is configured. Writes to `.part` then renames, so an
    interrupted run never leaves a corrupt file in the cache.
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
    if force:
        tmp.unlink(missing_ok=True)
    _fetch_with_resume(variant.url, tmp, variant.file)

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

"""Perplexity guardrail via `llama-perplexity` (lower = better).

The 40-item probe is a smoke alarm; it cannot resolve the ~1% quality shifts a quant change
actually causes (binomial noise at n=40 is bigger than the effect). Perplexity over a fixed
corpus can. Relative deltas between configs are the signal, so the corpus just has to be
identical across the configs being compared.

Default corpus: the repo's own README + docs/*.md concatenated — real English, already
committed, byte-identical for every config in a run. Substitute any text file with --file.

TODO(confirm) on the box: the exact summary-line format of llama-perplexity at the pinned
tag. The parser accepts any `PPL = <float>` occurrence and takes the last one.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_PPL_RE = re.compile(r"PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)")


class PerplexityError(RuntimeError):
    """llama-perplexity failed, timed out, or produced no parseable PPL value."""


@dataclass(frozen=True)
class PerplexityResult:
    ppl: float
    chunks: int
    corpus: str
    raw_tail: str = ""  # last lines of output, for provenance


def ensure_default_corpus(dest_dir: Path) -> Path | None:
    """Write the default corpus (README + docs/*.md) into dest_dir; None off-checkout."""
    from ..util import repo_root

    root = repo_root()
    sources = [root / "README.md"] + sorted((root / "docs").glob("*.md"))
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in sources if p.is_file()]
    if not texts:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "_corpus_docs.txt"
    dest.write_text("\n\n".join(texts), encoding="utf-8")
    return dest


def build_perplexity_command(
    ppl_bin: Path,
    model_path: Path,
    corpus: Path,
    *,
    threads: int | None = None,
    chunks: int = 8,
    ctx: int = 512,
) -> list[str]:
    cmd = [
        str(ppl_bin),
        "-m",
        str(model_path),
        "-f",
        str(corpus),
        "--chunks",
        str(chunks),
        # Pin the eval context: deterministic chunking and ~5x cheaper than the model
        # default. Relative deltas between configs are the signal, and they share -c.
        "-c",
        str(ctx),
    ]
    if threads:
        cmd += ["-t", str(threads)]
    return cmd


def run_perplexity(
    ppl_bin: Path,
    model_path: Path,
    corpus: Path,
    *,
    threads: int | None = None,
    chunks: int = 8,
    ctx: int = 512,
    timeout: float = 1800.0,
) -> PerplexityResult:
    """Run llama-perplexity and parse the final PPL value."""
    cmd = build_perplexity_command(
        ppl_bin, model_path, corpus, threads=threads, chunks=chunks, ctx=ctx
    )
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise PerplexityError(f"llama-perplexity timed out after {timeout:.0f}s") from exc
    out = (proc.stderr or "") + "\n" + (proc.stdout or "")
    if proc.returncode != 0:
        raise PerplexityError(f"llama-perplexity exited {proc.returncode}: {out.strip()[-400:]}")
    found = _PPL_RE.findall(out)
    if not found:
        raise PerplexityError(f"no 'PPL =' value in output; tail: {out.strip()[-400:]}")
    return PerplexityResult(
        ppl=float(found[-1]), chunks=chunks, corpus=corpus.name, raw_tail=out.strip()[-200:]
    )

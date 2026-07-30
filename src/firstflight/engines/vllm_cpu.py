"""Optional vLLM CPU backend (server-throughput story). Stub until needed.

vLLM's CPU build can serve an OpenAI-compatible endpoint; this adapter would drive a
concurrent-request throughput benchmark to complement llama.cpp's single-stream prefill
numbers. Kept optional so the core llama.cpp path ships without a heavy dependency.
"""

from __future__ import annotations


def available() -> bool:
    """Whether a usable vLLM CPU install is present. (Stub: always False for now.)"""
    return False


def run_throughput(*_args, **_kwargs):  # noqa: ANN002, ANN003
    raise NotImplementedError("vLLM CPU backend is an optional Phase 1+ extension — not yet wired.")

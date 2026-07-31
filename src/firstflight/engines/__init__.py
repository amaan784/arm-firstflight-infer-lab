"""Inference engine adapters.

`llama_cpp` is the sole engine: llama.cpp via `llama-cli` (generation/probe), `llama-bench`
(single-stream benchmark), `llama-server` (measured TTFT / prompt-cache), and
`llama-batched-bench` (concurrency throughput) — all discovered by `llama_cpp.find_binary`.
"""

from __future__ import annotations

__all__ = ["llama_cpp"]

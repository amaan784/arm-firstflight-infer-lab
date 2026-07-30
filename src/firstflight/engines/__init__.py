"""Inference engine adapters.

`llama_cpp` is the primary engine (llama.cpp via `llama-cli` / `llama-bench`).
`vllm_cpu` is an optional throughput-story backend (stub until needed).
"""

from __future__ import annotations

__all__ = ["llama_cpp", "vllm_cpu"]

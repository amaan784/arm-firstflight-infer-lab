"""Arm FirstFlight — Inference Optimization Lab.

Reproducible benchmark + optimization harness for CPU LLM inference on Arm
Neoverse cloud servers, focused on prefill / time-to-first-token (TTFT).

The package degrades gracefully off Arm: Arm-only tools (Arm Performix, KleidiAI
kernels) are wrapped behind clean interfaces that no-op with a clear message when
absent, so the smoke test runs on any machine.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]

"""Arm FirstFlight: inference optimization lab.

Benchmark + optimization harness for CPU LLM inference on Arm Neoverse cloud
servers, focused on prefill / time-to-first-token (TTFT).

Off Arm, the Arm-only tools (Performix, KleidiAI kernels) sit behind interfaces
that no-op with a message, so the smoke test still runs.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]

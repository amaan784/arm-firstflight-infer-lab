# Arm FirstFlight — v1: Foundation + smoke

> **Cumulative review build. v1 is the base layer: package + config + engine bootstrap.** Independently runnable.
> Stages: v1 foundation+smoke · v2 benchmark · v3 report · v4 profiling · v5 experiments+quality · v6 autotuner · v7 full integration

The foundation: a pinned, installable Python package (`firstflight`) with the **config layer**
(`configs/*.yaml` + typed loaders), platform/Arm detection, the **$/M-token cost math** (real
verified instance prices), llama.cpp **binary discovery**, **`setup-engine`** (auto-download the
prebuilt llama.cpp for THIS platform — Win/Linux/mac, x64/arm64, no compiler), **KleidiAI-active
detection**, and on-demand **model download**. `firstflight smoke` proves the pipeline with real
inference anywhere; without an engine it skips cleanly with a message.

## Install & run (any machine)
```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
firstflight setup-engine         # prebuilt llama.cpp for THIS platform
firstflight info
firstflight smoke                # real model download + real generation
pytest
```
Later stages add: benchmark sweep (**v2**), report (**v3**), Performix profiling (**v4**),
experiments + quality (**v5**), autotuner (**v6**), full integration (**v7**).

## Commands
`setup-engine · info · smoke · download`

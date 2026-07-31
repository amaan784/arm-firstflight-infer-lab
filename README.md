# Arm FirstFlight — v2: Benchmark core

> **Cumulative review build. v2 = v1 + the benchmark engine.** Independently runnable.
> Stages: v1 foundation+smoke · v2 benchmark · v3 report · v4 profiling · v5 experiments+quality · v6 autotuner · v7 full integration

This stage adds **`firstflight bench`**: drives `llama-bench` across a context-length sweep from
`configs/workloads.yaml`, capturing prefill (pp) + generation (tg) throughput, **derived TTFT**,
per-point variance, and **peak memory** (per-child isolation via GNU time when available) ->
structured results in `bench/results/*.json`. `firstflight run` does a single point; `--dry-run`
prints the command.

## Install & run (any machine)
```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
firstflight setup-engine         # prebuilt llama.cpp -> real sweeps on any machine
firstflight bench --dry-run
pytest
```
The next stage (**v3**) renders these JSON results into the standalone HTML report.

## Commands
`setup-engine · info · smoke · download · run · bench · ttft · throughput`

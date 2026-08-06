# Arm FirstFlight — v6: Agentic autotuner

> **Cumulative review build. v6 = v5 + the agentic autotuner.** Independently installable & runnable.
> Stages: v1 foundation+smoke · v2 benchmark · v3 report · v4 profiling · v5 experiments+quality · v6 autotuner · v7 full integration

This stage adds **`firstflight autotune --enable`**: an agent-in-the-loop optimizer that
proposes a config -> benchmarks it -> loops until no improvement. Default proposer is a
deterministic heuristic grid (no API key); `--llm` uses Claude via the optional `[agent]` extra.

## Install & run (any machine)
```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
# or conda: conda create -n <env-name> python=3.12 -y && conda activate <env-name>
pip install -e ".[report,dev]"
firstflight setup-engine         # prebuilt llama.cpp for THIS platform (no compiler)
firstflight report --demo        # full standalone HTML/markdown report (synthetic data)
firstflight autotune             # prints the opt-in note; --enable runs the loop
pytest
```
The final stage (**v7**) adds the integration layer: docs, free Arm CI (`arm-bench.yml` with the
one-click KleidiAI before/after job + run-summary report), Dockerfile, remote-VM scripts, and the
committed sample report.

## Commands
`setup-engine · info · smoke · download · run · bench · ttft · throughput · report · profile · experiment · autotune`

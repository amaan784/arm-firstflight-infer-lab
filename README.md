# Arm FirstFlight — v5: Optimization experiments + quality eval

> **Cumulative review build. v5 = v4 + the optimization experiments & quality guardrail.** Independently runnable.
> Stages: v1 foundation+smoke · v2 benchmark · v3 report · v4 profiling · v5 experiments+quality · v6 autotuner · v7 full integration

This stage adds the **actual optimization axis**: `firstflight experiment` benchmarks configs from
`configs/experiments.yaml` (quant scheme / threads / CPU pinning / KleidiAI build) **holding model +
instance fixed**, runs a small quality probe per config, **detects whether KleidiAI is actually
active** (load-log proof), then renders the before/after report.

## Install & run (any machine)
```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
# or conda: conda create -n <env-name> python=3.12 -y && conda activate <env-name>
pip install -e ".[report,dev]"
firstflight setup-engine                           # prebuilt llama.cpp (no compiler)
firstflight experiment --no-download --no-report   # runs configs (skips cleanly off Arm)
firstflight report --demo                          # standalone HTML/markdown report
pytest
```
The next stage (**v6**) adds the opt-in agentic autotuner.

## Commands
`setup-engine · info · smoke · download · run · bench · ttft · throughput · report · profile · experiment`

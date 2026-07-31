# Arm FirstFlight — v3: Report generator

> **Cumulative review build. v3 = v2 + the report generator.** Independently runnable.
> Stages: v1 foundation+smoke · v2 benchmark · v3 report · v4 profiling · v5 experiments+quality · v6 autotuner · v7 full integration

This stage adds **`firstflight report`**: a clean **standalone HTML + markdown** report from
`bench/results/*.json` — headline-led, matplotlib charts (prefill-TTFT scaling, throughput, cost),
a **$/M-token** column (real instance prices in `configs/instances.yaml`), and a quality column
(populated from v5 onward). `--demo` previews the layout with clearly-labeled synthetic data.

## Install & run (any machine)
```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[report,dev]"
firstflight report --demo        # full standalone HTML/markdown report
firstflight bench --dry-run      # prints the llama-bench command
pytest
```
The next stage (**v4**) adds Arm Performix profiling that feeds a hotspots section into this report.

## Commands
`setup-engine · info · smoke · download · run · bench · ttft · throughput · report`

# Arm FirstFlight — v4: Arm Performix profiling

> **Cumulative review build. v4 = v3 + Arm Performix profiling.** Independently runnable.
> Stages: v1 foundation+smoke · v2 benchmark · v3 report · v4 profiling · v5 experiments+quality · v6 autotuner · v7 full integration

This stage adds **`firstflight profile`**: it runs Arm Performix (`apx` recipe `code_hotspots`)
on a representative prefill run and surfaces the top hotspot functions into the report. Off Arm /
without `apx` it no-ops cleanly.

## Install & run (any machine)
```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[report,dev]"
firstflight profile              # no-op off Arm; runs apx on an Arm box
firstflight report --demo        # report now includes a hotspots section
pytest
```

## Commands
`setup-engine · info · smoke · download · run · bench · ttft · throughput · report · profile`

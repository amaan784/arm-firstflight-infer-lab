# bench/results

Structured benchmark output (`*.json`) lands here. The committed `example_*.json` and
`profile_example.json` are synthetic schema examples for browsing the format. The report
loader skips them by name, and both CI jobs delete them before measuring, so they cannot
end up in a real report.

## Committed real runs

`run-<github-run-id>/` holds the results of an actual Arm CI run, kept as the evidence
behind the report in `bench/reports/`. Re-render any of them with:

```
firstflight report --results-dir bench/results/run-31656321896
```

They live in a subdirectory on purpose. The loader globs `*.json` one level deep, so a
committed run is invisible to a default `firstflight report` and cannot mix its numbers
into a later measurement — the failure mode where a report silently averages two different
machines.

| run | what it measured |
|---|---|
| `run-31656321896` | KleidiAI attribution ladder (generic / repack / kleidiai) + noise floor, qwen2.5-1.5b-instruct q4_0, 4 threads, `ubuntu-24.04-arm` (Neoverse N2) |
| `run-31784946201` | the same ladder at q8_0, `rag-context` workload, 3 repetitions. No noise-floor control: `run_q8_only` skips it, so deltas here are compared against the 0.3% floor measured at q4_0 rather than one of their own. |

### A known stale field in `run-31656321896`

Those JSONs carry `host.kernel_tier: "I8MM"` on every rung, including the armv8-a floor.
The tier was derived partly from `/proc/cpuinfo` (a property of the host) instead of purely
from ggml's `system_info` (a property of the build), so every rung inherited the runner's
capability. `run-31784946201` has the corrected values (`generic-q8` reads `NEON`).

The files are left exactly as CI wrote them. They are measurement artifacts, and editing one
after the fact to read better is the failure this project is about. The report does not use
the stale field: `build_report_model` re-derives the tier from each run's own `system_info`
at render time, which is why the committed report correctly shows `generic: NEON-tier
kernels`. `tests/test_report_logic.py::test_kernel_evidence_rederives_tier_from_each_runs_own_system_info`
pins that behaviour.

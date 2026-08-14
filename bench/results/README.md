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

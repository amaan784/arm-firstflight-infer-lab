# Measurement Methodology

Credibility is the point. This document states exactly how every number in the report is
produced so anyone can reproduce — or challenge — it. ML-engineer judges reward rigor.

## What we measure (and why)

The headline metric is **prefill latency / time-to-first-token (TTFT)** as a function of
context length. Agentic and RAG workloads stuff long contexts into the prompt, so the cost
users *feel* before any token appears is dominated by **prompt processing (prefill)**, not
generation. We make long-context prefill scaling a first-class axis.

| Metric | Definition | Source |
| --- | --- | --- |
| Prefill throughput | tokens/sec processing the prompt | `llama-bench -p` |
| Generation throughput | tokens/sec generating output | `llama-bench -n` |
| **TTFT** | `prompt_tokens / prefill_throughput` | derived (`bench/prefill.py:ttft_seconds`) |
| Peak memory | max RSS during the run | GNU `time -v` per child (`bench/memory.py`) |
| Quality delta | 40-item exact-match probe (or lm-eval), before vs after | `eval/quality.py` |
| $/M tokens | `usd_per_hour / (tok_s · 3600) · 1e6` | `cost.py` + `instances.yaml` |
| Measured TTFT | llama-server's own `timings.prompt_ms`, cold vs warm prefix | `bench/ttft.py` |
| Concurrency throughput | aggregate tok/s at 1/2/4/8 parallel requests | `bench/throughput.py` (llama-batched-bench) |

## How the numbers are produced (llama-bench)

`firstflight bench` drives a single `llama-bench` invocation with `-o json` and parses the
top-level JSON array of per-test rows (schema confirmed against
`tools/llama-bench/llama-bench.cpp`, 2026-06-26):

- The context sweep goes in `-p` (e.g. `-p 128,512,2048,8192,16384,32768`); generation in
  `-n` (e.g. `-n 32`). A **prefill (pp)** row has `n_prompt>0, n_gen==0`; a **generation (tg)**
  row has `n_gen>0, n_prompt==0`.
- Throughput is `avg_ts` (tokens/sec) with `stddev_ts` across repetitions; `-r/--repetitions`
  controls repeats (default 5). **TTFT is derived** as `n_prompt / avg_ts`.
- llama-bench performs **warm-up runs by default** (disable with `--no-warmup`), so warm-up is
  handled inside the tool; `configs/workloads.yaml:warmup` is informational.
- Thread count is `-t`; CPU **affinity/pinning** uses `-C/--cpu-mask` + `--cpu-strict`
  (exercised by the experiments).
- **Peak memory** is the benchmarked child's peak RSS, measured **per child** by wrapping the
  run in GNU `time -v` (`Maximum resident set size`, `bench/memory.py`) so each config gets its
  own attribution; `resource.getrusage(RUSAGE_CHILDREN)` is the fallback where GNU time is
  absent (that high-water mark spans all children, so it is labeled as such).

llama-bench sweeps over prompt sizes and threads are also Arm's own prescribed cloud
benchmarking method — Arm's Graviton4/Axion Learning Paths run e.g.
`llama-bench -m [model] -p 128,256,512 -n 128 -t 4,8,16` and read `pp` (prompt processing =
our prefill) and `tg` tokens/s
([example](https://learn.arm.com/learning-paths/servers-and-cloud-computing/arcee-foundation-model-on-aws/)).

## Measured TTFT (llama-server) and the prompt-cache demo

Derived TTFT (above) is an approximation. `firstflight ttft` **measures** it: llama-server's
native `/completion` response carries a `timings` object with `prompt_ms` (wall-clock prompt
processing) and `prompt_n` (tokens actually prefilled) — the server's own stopwatch, including
tokenization.

The same command demonstrates **prefix caching**, the dominant TTFT lever for agentic/RAG
serving: two requests share a long prefix (a system prompt / RAG context) with different
questions. llama-server's `cache_prompt` (default true) re-uses the KV cache for the shared
prefix, and `--cache-reuse N` extends the reuse to shifted chunks. The cold turn prefills the
full prefix; the warm turn processes only the new question — `prompt_n` collapses from
thousands of tokens to dozens, and `prompt_ms` with it. Both turns are reported side by side
(`bench/results/ttft_*.json`), from the server's measurements, not ours.

## Protocol (rigor checklist)

- **Same instance for before & after.** The optimization delta is only meaningful when the
  model and hardware are held fixed; only the optimization variable changes.
- **Warm-ups handled inside the tool.** llama-bench performs warm-up runs by default
  (disable with `--no-warmup`), priming caches and the allocator before timed runs; the
  `warmup` field in configs/workloads.yaml is informational, not an executed count.
- **Repeats + variance.** `repeats` timed runs per config (default 5). We report llama-bench's
  mean (`avg_ts`) **± stddev** across repetitions, never a single cherry-picked number.
- **Fixed seed.** Generation uses a fixed seed for run-to-run comparability.
- **Pinned environment.** llama.cpp pinned to a specific `b####` release tag; deps pinned in
  `pyproject.toml`; reproducible image in `docker/Dockerfile.arm64`. Our pin (b9873) is newer
  than the tag Arm's own KleidiAI build documentation last tested (b7610 — "Newer versions
  should also work but are not tested",
  [Arm Learning Path](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/performance_llama_cpp_sme2/)),
  and `-DGGML_CPU_KLEIDIAI=ON` is exactly Arm's documented enable flag.
- **Threads + affinity recorded.** Thread count and CPU pinning are part of the config and
  logged with every result (a tuned experiment variable).

## The optimization axes

Holding model + instance fixed, we compare:

1. **Quantization scheme** — Q4_0 vs Q4_K_M vs Q8_0 (`quant-sweep`).
2. **Threads + CPU pinning** — thread count sweep and affinity on/off (`thread-sweep`/`pinning`).
3. **KV-cache type** — default f16 vs quantized q8_0 cache (`llama-bench -ctk/-ctv`, flags
   verified against the real b9873 binary; flash-attn pinned on for both) (`kv-cache`).
4. **Prefill micro-batch** — `-ub` 256/512/1024/2048 (`prefill-batch`) and **flash attention**
   on/off (`flash-attn`).
5. **Build targeting** — default build vs `-mcpu=native`, the official AWS/Arm recipe
   (`build-flags`).
6. **KleidiAI on vs off** — baseline llama.cpp build vs `-DGGML_CPU_KLEIDIAI=ON` (`kleidiai`).
7. **Prompt/prefix caching** — llama-server `cache_prompt` + `--cache-reuse` (measured by
   `firstflight ttft`, not llama-bench).

> **KleidiAI caveat (verified 2026-06-26):** Arm's KleidiAI microkernels in current llama.cpp
> accelerate **Q4_0 and Q8_0** weights only (a one-time repack at model load) — **not** Q4_K_M
> / k-quants. So the KleidiAI on/off comparison uses a **Q4_0** model. Confirm KleidiAI is
> actually active by checking the load log for:
> `load_tensors: CPU_KLEIDIAI model buffer size = … MiB`.
> Re-verify supported quant types against the pinned tag's
> `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`, since support changes across releases.

`firstflight experiment [--name <exp>]` drives this: each config in `configs/experiments.yaml`
(varying `variant` / `threads` / `cpu_mask` / `bin`) becomes a labelled `SweepResult`, run on
the **same instance** with the same workload, then the report compares them. The KleidiAI axis
points two configs at two llama.cpp **builds** via `bin: ${LLAMA_BASELINE_BIN}` /
`${LLAMA_KLEIDIAI_BIN}` (since KleidiAI is a build-time flag, not runtime).

Every speed result is paired with a **quality delta** so we prove the speedup didn't tank
accuracy. Two paths:

- **Built-in probe (default):** a small fixed exact-match Q&A set run through `llama-cli` —
  self-contained, no torch, runs on the bench box. A regression guardrail, not a leaderboard.
- **lm-evaluation-harness (optional, `[eval]` extra):** for a real MMLU/GSM8K subset, start a
  `llama-server` and point lm-eval at it with `--model local-chat-completions` (the default;
  `local-completions` targets the raw completion endpoint) (the light
  `lm-eval[api]` install, no torch — the `gguf` backend is known-broken). Exact task names /
  endpoint are `TODO(confirm)` on the box; see `eval/quality.py:build_lm_eval_cmd`.

## Autotuner (stretch, opt-in)

`firstflight autotune --enable` closes the loop: propose a config → benchmark it → repeat until
no improvement (or the search space is exhausted, or `--max-iters`). The default proposer is a
deterministic grid over quant variant × thread count (no API key); `--llm` swaps in a Claude
proposer that reads the Performix hotspots + trajectory and returns the next config as JSON,
falling back to the grid on any failure/repeat. It optimizes prefill throughput at a target
context and writes the best config + full trajectory to `bench/results/autotune_*.json`.
Strictly opt-in so the core ships without it.

## Profiling (Arm Performix)

`firstflight profile` runs Arm Performix (`apx`) against a representative prefill command and
surfaces the **top self-time hotspot functions** into the report. The real `apx` flow (sourced
from Arm's open-source MCP server, github.com/arm/mcp) is:

```
apx recipe run code_hotspots --workload="<llama-bench …>" --json --target=localhost \
    --host-key-policy=ignore --deploy-tools      # -> run_id
apx run render <run_id>                          # -> session_id
apx render query <session_id> "<top-10 self-time SQL>"   # -> ┃-delimited hotspot table
```

The `code_hotspots` recipe returns `function_name`, `node_type`, `periodic_samples_self`,
`periodic_samples_self_percent`. `firstflight` parses that (JSON, `┃`-table, or CSV) into the
report's "Top hotspots (Arm Performix)" section. Off Arm — or without `apx` — profiling no-ops
with a clear message and the report omits the section.

`TODO(confirm)` on the live box (the official CLI Reference Guide, doc 111566, is a JS-rendered
SPA): the exact JSON keys for `run_id`/`session_id` and whether `recipe run --json` already
embeds the hotspot table. The **commands** above come from Arm's own tooling, not invention.

## Honest limitations

- GitHub Arm runners are shared/virtualized; absolute numbers there are indicative. The
  remote-VM path (`scripts/setup_arm_vm.sh`) gives a dedicated instance for headline numbers.
- The quality eval is a *small* slice (a guardrail, not a leaderboard).
- Instance prices in `configs/instances.yaml` are real published on-demand prices with
  verification dates, but prices float by region and over time — re-check them for the exact
  instance/region on the day you benchmark.
- The committed sample report (`bench/reports/`) and example results (`bench/results/example_*.json`)
  use **synthetic, illustrative** data (banner-labeled in the report) until a real Arm run lands —
  reproduce them with `make report-demo`, or replace them via `make bench && make report` on Arm.

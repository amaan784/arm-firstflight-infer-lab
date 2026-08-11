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
  absent (that high-water mark spans all children, so it is labeled as such). Benchmark sweeps
  run with `-mmp 0` (mmap off; flag verified on the real b9873 binary 2026-08-09), so peak RSS
  is honest allocated memory rather than lazily-paged file mappings — at some model-load-time
  cost outside the timed regions. Sweeps also set `--prio 2` and `--delay 2` (steadier numbers
  on shared runners). Treat peak RSS as capacity context, not a tuned metric.

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
- **A measured noise floor, and a gate that uses it.** The `noise-floor` experiment runs the
  *same build with the same config* under two labels. Any apparent speedup between them is the
  machine talking, not the treatment, so that spread is the bar a real delta must clear. The
  report computes it and **refuses to headline a speedup that sits inside it** — printing
  "within noise / not claimed" instead of a multiplier. A harness that can't report *no win*
  isn't measuring, it's advertising.
- **Interleaved rounds (`experiment --rounds N`).** Repeats inside one llama-bench process
  share cache/allocator state and can't see slow drift. Rounds re-run the whole config list
  round-robin (A,B,C,A,B,C,...); the report takes the **median per rung** and shows the
  **between-round spread** — the error bar that matters on a shared runner. CI runs the
  headline ladder with 3 rounds.
- **Noise floor (`noise-floor` experiment).** The same build measured twice under two labels.
  Its apparent "speedup" is pure runner noise (co-tenancy, thermals) and is published next to
  the headline — the yardstick a real delta must clear.
- **Fixed seed.** llama-cli paths (smoke, quality probe) pin seed 42 + greedy decoding;
  llama-bench itself takes no seed flag (its measurement is timing, not sampling).
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
5. **Build targeting** — a generic armv8-a floor (`GGML_NATIVE=OFF`, `GGML_CPU_REPACK=OFF`)
   vs the native default (`build-flags`). A plain Release build already carries the
   `-mcpu=native`-equivalent that the official AWS/Arm recipe prescribes — `GGML_NATIVE`
   defaults ON (verified in b9873's ggml/CMakeLists.txt, 2026-08-08) — so default-vs-native
   would be native-vs-native and measure only noise.
6. **KleidiAI attribution ladder** — generic floor → ggml's own aarch64 **repack** kernels
   (`GGML_CPU_REPACK`, ON by default: the named, controlled middle rung) → `-DGGML_CPU_KLEIDIAI=ON`
   (`kleidiai`). Every rung is a distinct mechanism, so each delta has one cause.
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
  Scoring matches whole words in the **first line** of the completion only (a rambling model
  must not get extra chances to hit the gold string). Honest limits: at n=40 the 95% binomial
  interval is roughly ±12 points around 80%, so the probe catches *collapse*, not a 1-2%
  drift — and in the KleidiAI ladder all rungs load the *identical* Q4_0 file, so the probe
  is flat there by construction. That's what perplexity is for:
- **Perplexity (`firstflight perplexity` / automatic in experiments):** `llama-perplexity`
  over a fixed corpus (default: this repo's own docs — identical text across the configs
  being compared, which is all a relative delta needs). Resolves the ~1% shifts the probe
  can't; reported per config as `ppl` in the runs table, lower = better.
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

## The negative control

`kleidiai-null-control` deliberately runs the KleidiAI build against a **Q4_K_M** model, which
its kernels cannot accelerate — upstream llama.cpp logs exactly that
([PR #25701](https://github.com/ggml-org/llama.cpp/pull/25701), merged 2026-07-21). The probe
must come back **KleidiAI inactive** for that row, and any timing difference must therefore
*not* be attributed to KleidiAI. It is a test of the instrument rather than of the silicon: if
this control ever reports KleidiAI active on a k-quant, the detection is broken and every other
KleidiAI claim in the report is void.

## Both quants, because the mechanism differs

The ladder runs at **Q4_0** (`kleidiai`) and again at **Q8_0** (`kleidiai-q8_0`). ggml's own
aarch64 repack path targets Q4_0, so on that quant the `repack` rung already captures most of
the available headroom and KleidiAI's own contribution looks small. Q8_0 leaves KleidiAI's
kernels room to show a clean delta. Reporting one quant alone would make the mechanism look
like whichever story that quant happened to tell.

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

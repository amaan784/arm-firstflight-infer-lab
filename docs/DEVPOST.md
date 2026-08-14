# Devpost submission draft (ready to paste)

> Numbers below are the real measured results from two `arm-bench` runs: 31656321896
> (Q4_0 ladder + noise floor) and 31784946201 (Q8_0 ladder). Add the video link before
> submitting; everything else is ready to paste.

## Project name
**Arm FirstFlight Attribution Ladder**

## Elevator pitch (tagline)
Every KleidiAI benchmark I found compares against a build that's already Arm-optimized. So I
built the real baseline. At Q4_0 KleidiAI adds nothing. At Q8_0 it's worth 1.23x.

## Project overview: why FirstFlight exists

Nearly every "we enabled KleidiAI and got N×" result (including ones I set out to reproduce)
compares against a baseline that is already accelerated. In llama.cpp's own
`ggml/CMakeLists.txt`, `GGML_NATIVE` and `GGML_CPU_REPACK` both default ON, so the "before"
build already has native targeting and ggml's aarch64 Q4_0 repack GEMM. The standard on/off
test hands KleidiAI credit for both. Worse, on Q4_K_M, the quant most people download,
KleidiAI's kernels never engage at all; upstream added a warning for this case in
[PR #25701](https://github.com/ggml-org/llama.cpp/pull/25701) (merged 2026-07-21).

So the number everyone quotes is unattributable. FirstFlight fixes the measurement:

- **Builds the true floor** (`GGML_NATIVE=OFF`, `GGML_CPU_ARM_ARCH=armv8-a`,
  `GGML_CPU_REPACK=OFF`) and runs a three-rung ladder against it, at Q4_0 and again at
  Q8_0. ggml's repack targets Q4_0, so measuring only that quant understates KleidiAI: the
  two ladders give opposite answers, and that contrast is the result.
- **Proves engagement** from the model-load log: which weight buffer loaded, which ISA tier
  ran (I8MM / DOTPROD / SVE2 / NEON). No marker, no claim.
- **Runs a negative control**: the KleidiAI build on Q4_K_M, where the probe must report
  INACTIVE. This tests the instrument, not the silicon. (Implemented and gated behind a
  workflow input; not part of either linked run.)
- **Refuses wins inside its own noise**: the same build measured twice sets the floor, and any
  delta that doesn't clear it prints `within noise` instead of a multiplier, with the headline
  reading `No claimed win`.

## Inspiration
Agentic and RAG apps live or die on time-to-first-token, and on long contexts that wait is
almost entirely prefill. Meanwhile more and more inference runs on Arm cloud CPUs (Graviton,
Axion, Cobalt) because they are cheap per token. We wanted to measure how much faster and
cheaper Arm-specific optimization makes this workload, and whether it costs any accuracy.

## What it does: functionality and final output
`firstflight` is a pip-installable harness for quantized CPU inference on standard CPU-only
Arm64 cloud instances. It:
- **Measures** prefill/TTFT scaling across context lengths (1k → 8k tokens on the free
  runner, configurable higher) with warm-ups,
  repeats, and variance (fixed seeds + greedy decoding in the generation/quality probes),
  plus TTFT taken from llama-server's own timings and concurrency throughput at 1–8
  parallel requests.
- **Optimizes** along seven Arm-specific axes: KleidiAI microkernels (as an attribution
  ladder over ggml's own aarch64 repack path), quant scheme chosen for the silicon (Q4_0 vs
  the default Q4_K_M), thread pinning, quantized KV-cache, prefill micro-batch, compiler
  build targeting (generic armv8-a vs native), and prompt/prefix caching.
- **Proves it**: KleidiAI activation is detected from the load log (not assumed), and every
  rung is checked for output drift (perplexity on the headline run, plus an optional 40-item
  exact-match probe). Costs use real dated prices; the free CI runner is priced at $0/hr, so
  the headline run reports no dollar figures rather than borrowing another instance's.
- **Reports it**: a self-contained one-page HTML report (headline delta, charts, before/after
  tables, hotspots), generated automatically and rendered into the GitHub Actions run summary.

**Headline result:** on a free `ubuntu-24.04-arm` runner (Azure Cobalt 100, Neoverse N2),
Qwen2.5-1.5B-Instruct Q4_0, 4 threads: **ggml's own aarch64 repack path delivers 3.60x
faster prefill at 1,024 tokens, and KleidiAI adds 1.00x on top of it.** TTFT at 1k drops
38.9s → 10.8s from repack alone; KleidiAI moves it to 10.7s. Perplexity is flat across all
three rungs at Q4_0 (37.4325 / 37.4181 / 37.4181, a 0.04% spread), so at this quant the
kernel swap is output-neutral. That does not hold at Q8_0 - see below.

Against a measured noise floor of 0.3%, that 1.00x is a genuine null, not a failed
measurement: KleidiAI demonstrably engaged (weight buffer `CPU_KLEIDIAI` 702.86 MiB vs
`CPU_REPACK` 885.41 MiB vs `CPU_Mapped` 1011.16 MiB for the floor). It simply had nothing
left to win at Q4_0, because ggml's repack had already taken it.

**That is the whole point.** The standard "KleidiAI on vs off" A/B would have reported the
full 3.60x as a KleidiAI win. Only the generic armv8-a floor rung separates the two, and
building that floor is what this harness does.

Two findings we did not go looking for:

- **The advantage inverts with context.** The generic build is nearly flat as context grows
  (26.4 → 22.6 tok/s) while the accelerated builds collapse (94.8 → 20.4), crossing over at
  8,192 tokens where they land 10% behind the floor. Peak RSS is identical at 1.3 GiB on
  every rung, so it is not memory pressure. We report the measurement and flag the
  mechanism as open rather than guessing.
- **Prefix caching beats every kernel we tested.** Cold TTFT 8,492 ms → warm 54 ms on the
  same prefix, taken from llama-server's own timings. 99% of prefill skipped, against 0-1%
  for the kernel swap this entire ladder exists to measure.

**Then we tested the prediction.** ggml's repack targets Q4_0, so if the null really is
"repack got there first", KleidiAI should have room at Q8_0. We ran that ladder:

| prompt tokens | generic | repack | KleidiAI | **KleidiAI vs repack** |
|---:|---:|---:|---:|---:|
| 2,048 | 28.2 | 58.8 | 72.5 | **1.23x** |
| 4,096 | 26.9 | 35.8 | 40.3 | **1.13x** |
| generation | 18.1 | 39.2 | 45.2 | **1.15x** |

**At Q8_0 KleidiAI earns 1.23x over repack.** At Q4_0 its build reports `repack=on`; at Q8_0
it reports `repack=off, kleidiai=on` and loads `CPU_KLEIDIAI` where repack cannot follow.
Different code path, not more of the same one.

So the standard benchmark is wrong in **both** directions: it credits KleidiAI for work it
did not do at Q4_0, and it never tests the quant where it does.

**And the guardrail caught a cost.** Perplexity across the Q8_0 rungs spreads 1.43% (31.6267
to 32.0774), KleidiAI the outlier, against 0.04% at Q4_0 where repack and KleidiAI agreed to
six significant figures. A 23% speedup that shifts model output by 1.4% is a trade, and the
report says so on its face. A speed-only benchmark ships that blind.

**Scope, stated plainly:** one model (Qwen2.5-1.5B-Instruct), two quants, one host (4-vCPU
Neoverse N2), 1k-8k context. The Q8_0 run skips the noise-floor control, so its deltas are
weighed against the 0.3% floor measured at Q4_0, and its perplexity is one measurement per
rung with no error bar. The Q4_K_M negative control and the quant/KV-cache/micro-batch sweeps
are implemented but gated: they add ~11h, past the CI job's 300-minute timeout. Everything
claimed above was measured in the linked runs; nothing was extrapolated.

**What the final output is.** Not an optimized model. The deliverable is three things: a
reproducible measurement harness (`pip install -e .`, one CI click, no Arm hardware needed); a
standalone HTML + Markdown + JSON report per run, committed alongside the raw results; and the
finding itself, which is that ggml's repack delivers 3.60x at Q4_0 while KleidiAI adds 1.00x,
and KleidiAI delivers 1.23x at Q8_0 where repack cannot reach. Anyone benchmarking llama.cpp
on Arm can rerun the ladder against their own model and get an attribution, not a single
unexplained multiplier.

Full report, raw result JSONs and the CI run are linked below; every number here re-renders
from committed data with one command.

> **Gallery note (don't paste this):** embed the report screenshot and the before/after table
> as images in the Devpost gallery. The rules say judges "are not required to test the Project
> and may choose to judge based solely on the text description, images, and video", so the
> submission must carry the evidence on its own.

## How we built it
A Python CLI over llama.cpp's own tools (`llama-bench`, `llama-completion`, `llama-server`,
`llama-batched-bench`). Every external fact (flags, JSON schemas, GGUF URLs, runner labels,
prices) is verified against live sources or real binaries and dated in the repo. The CI
workflow builds llama.cpp three ways (a generic armv8-a floor with repack+native OFF, the
native+repack default, and KleidiAI) on a free `ubuntu-24.04-arm` runner and runs the full
evidence suite (attribution ladder in 2 interleaved rounds, plus a same-build noise-floor
control) in one dispatch. Arm Performix is
wrapped behind the documented `apx` recipe flow (sourced from Arm's own MCP server), and an
opt-in agent closes the propose→measure loop.

## Challenges we ran into
- KleidiAI only accelerates Q4_0/Q8_0; the popular Q4_K_M default silently opts out of
  Arm acceleration. We turned that into a measured axis and documented it with sources.
- A quantized KV-cache without flash attention can be slower than f16, so our KV experiment
  pins FA on for both sides of the comparison.
- We caught and fixed several measurement bugs along the way: peak-RSS attribution, sampler
  noise in the quality probe (now greedy), a cost metric that mismatched the prefill headline,
  and a prompt-cache server harness that could deadlock on a full stdout pipe.

## Accomplishments we're proud of
- A judge can go from zero to a full Arm before/after report with one workflow click, no
  hardware required, and the report appears in the run summary itself.
- The harness reports a null as a null. KleidiAI came out at 1.00x over ggml's repack at
  Q4_0, and rather than reaching for a friendlier baseline we published it against a measured
  0.3% noise floor with the weight-buffer evidence proving the kernels really did engage.
- The repo was committed in stages, each one installable and passing its own tests, so the
  history doubles as a walkthrough for anyone assembling a similar benchmark rig.

## What we learned
Prefill is compute-bound and rewards Arm's matrix instructions heavily; generation is
bandwidth-bound and rewards cache/quant choices; and the single biggest TTFT lever for agent
serving is prefix caching, not a kernel. We measure it server-side.

## What's next
Land Performix hotspot attribution on a real box run, extend the headline runs to the verified
1.5B model, and grow the shipped adoption kit (`docs/ADOPT.md` + the workflow template) into
an upstream-able starter other llama.cpp-on-Arm projects can drop in.

## Setup Instructions (build / run / validate on Arm64)

**Any machine:** real inference in three commands (no compiler; run in a fresh venv or conda
env):
```bash
pip install -e ".[report,dev]"
firstflight setup-engine     # downloads the prebuilt llama.cpp for YOUR platform
firstflight smoke            # real model download + one real generation; `pytest` = 119 tests
```

**Path A, Arm64 in the cloud at zero cost (the judge path):** the repo's `arm-bench`
workflow runs on a free GitHub-hosted `ubuntu-24.04-arm` runner (Azure Cobalt 100, Arm
Neoverse N2). GitHub → Actions → **arm-bench** → *Run workflow* builds llama.cpp three ways
(generic armv8-a floor / native+repack default / KleidiAI), runs the attribution ladder +
noise-floor control + the measured-TTFT prompt-cache demo + the concurrency sweep, and
renders the before/after report into the run summary. A green run is linked
below. (To dispatch it yourself: fork the public repo, since `Run workflow` needs write
access; enable workflows on the fork; and run it there on the same free Arm runners.)

**Path B, Arm VM (bigger models):** on Ubuntu 24.04 aarch64 (AWS Graviton or another
Arm-based cloud instance):
```bash
bash scripts/setup_arm_vm.sh && . .venv/bin/activate && make bench && make report
```

**Validate (not just run):** the report's `kleidiai` column shows the KleidiAI kernels were
active (grepped from the model-load log, never assumed); the quality column shows the
40-item exact-match probe held (n/N before → after); every result JSON records host, build,
threads, and THP mode so any number can be traced to its exact configuration.

## Built with
`python` · `llama.cpp` · `kleidiai` · `arm-performix` · `github-actions` (ubuntu-24.04-arm) ·
`matplotlib` · `aws-graviton`

## Links
- Repo: https://github.com/amaan784/arm-firstflight-infer-lab (MIT)
- One-click Arm run (serves as the rules' required "functioning demo / test build" access):
  https://github.com/amaan784/arm-firstflight-infer-lab/actions/runs/31656321896
- Report from that run:
  https://github.com/amaan784/arm-firstflight-infer-lab/blob/main/bench/reports/report-20260814-031936.md
- Raw results:
  https://github.com/amaan784/arm-firstflight-infer-lab/tree/main/bench/results/run-31656321896
- Video: ⟨paste link⟩ (script: `docs/DEMO_SCRIPT.md`)

# Devpost submission draft (ready to paste)

> Fill the `⟨⟩` placeholders with the REAL numbers from your `arm-bench` →
> `kleidiai-before-after` CI run before submitting. Everything else is ready.

## Project name
**Arm FirstFlight — Inference Optimization Lab**

## Elevator pitch (tagline)
Your KleidiAI speedup is measured against the wrong baseline: a stock llama.cpp build is
already Arm-accelerated. FirstFlight builds the real floor, splits the win by mechanism, and
refuses to claim what it can't prove.

## Why FirstFlight

Nearly every "we enabled KleidiAI and got N×" result (including ones I set out to reproduce)
compares against a baseline that is already accelerated. In llama.cpp's own
`ggml/CMakeLists.txt`, `GGML_NATIVE` and `GGML_CPU_REPACK` both default ON, so the "before"
build already has native targeting and ggml's aarch64 Q4_0 repack GEMM. The standard on/off
test hands KleidiAI credit for both. Worse, on Q4_K_M, the quant most people download,
KleidiAI's kernels never engage at all; upstream added a warning for this case in
[PR #25701](https://github.com/ggml-org/llama.cpp/pull/25701) (merged 2026-07-21).

So the number everyone quotes is unattributable. FirstFlight fixes the measurement:

- **Builds the true floor** (`GGML_NATIVE=OFF`, `GGML_CPU_ARM_ARCH=armv8-a`,
  `GGML_CPU_REPACK=OFF`) and runs a three-rung ladder against it, at both Q4_0 and Q8_0.
  ggml's repack targets Q4_0, so measuring only that quant hides KleidiAI's contribution.
- **Proves engagement** from the model-load log: which weight buffer loaded, which ISA tier
  ran (I8MM / DOTPROD / SVE2 / NEON). No marker, no claim.
- **Runs a negative control**: the KleidiAI build on Q4_K_M, where the probe must report
  INACTIVE. This tests the instrument, not the silicon.
- **Refuses wins inside its own noise**: the same build measured twice sets the floor, and any
  delta that doesn't clear it prints `within noise` instead of a multiplier, with the headline
  reading `No claimed win`.

## Inspiration
Agentic and RAG apps live or die on time-to-first-token, and on long contexts that wait is
almost entirely prefill. Meanwhile more and more inference runs on Arm cloud CPUs (Graviton,
Axion, Cobalt) because they are cheap per token. We wanted to measure how much faster and
cheaper Arm-specific optimization makes this workload, and whether it costs any accuracy.

## What it does
`firstflight` is a pip-installable harness for quantized CPU inference on standard CPU-only
Arm64 cloud instances. It:
- **Measures** prefill/TTFT scaling across context lengths (128 → 32k tokens) with warm-ups,
  repeats, and variance (fixed seeds + greedy decoding in the generation/quality probes),
  plus TTFT taken from llama-server's own timings and concurrency throughput at 1–8
  parallel requests.
- **Optimizes** along seven Arm-specific axes: KleidiAI microkernels (as an attribution
  ladder over ggml's own aarch64 repack path), quant scheme chosen for the silicon (Q4_0 vs
  the default Q4_K_M), thread pinning, quantized KV-cache, prefill micro-batch, compiler
  build targeting (generic armv8-a vs native), and prompt/prefix caching.
- **Proves it**: KleidiAI activation is detected from the load log (not assumed), every config
  runs a 40-item exact-match quality guardrail, and costs use real dated AWS prices.
- **Reports it**: a self-contained one-page HTML report (headline delta, charts, before/after
  tables, hotspots), generated automatically and rendered into the GitHub Actions run summary.

**Headline result (⟨replace with real numbers⟩):** ⟨X⟩× faster prefill at a ⟨ctx⟩-token
context. TTFT ⟨A⟩s → ⟨B⟩s on ⟨instance/runner⟩; quality held (probe ⟨n/N⟩ → ⟨n/N⟩,
perplexity ⟨P1⟩ → ⟨P2⟩). Measured up the attribution ladder: generic armv8-a →
ggml's aarch64 repack → KleidiAI.

> **Fill-in note (don't paste this):** the free-runner CI ladder measures up to 16k tokens
> at $0/hr; use those numbers as-is ("on a free `ubuntu-24.04-arm` runner, Azure Cobalt
> 100"). Only claim 32k context or $/M-token dollar figures if you ran Path B on a
> paid instance (e.g. c8g.2xlarge), because judges can check the reproduction path.

> **Gallery note (don't paste this):** embed the report screenshot and the before/after table
> as images in the Devpost gallery. The rules say judges "are not required to test the Project
> and may choose to judge based solely on the text description, images, and video", so the
> submission must carry the evidence on its own.

## How we built it
A Python CLI over llama.cpp's own tools (`llama-bench`, `llama-cli`, `llama-server`,
`llama-batched-bench`). Every external fact (flags, JSON schemas, GGUF URLs, runner labels,
prices) is verified against live sources or real binaries and dated in the repo. The CI
workflow builds llama.cpp three ways (a generic armv8-a floor with repack+native OFF, the
native+repack default, and KleidiAI) on a free `ubuntu-24.04-arm` runner and runs the full
evidence suite (attribution ladder in 3 interleaved rounds, plus a same-build noise-floor
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
- Every claim is either measured on the same instance with variance shown, or labeled
  synthetic until the real run lands. The report enforces this with an automatic DEMO banner.
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
firstflight smoke            # real model download + one real generation; `pytest` = 107 tests
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
  ⟨link to a green arm-bench workflow run⟩
- Video: ⟨link⟩ (script: `docs/DEMO_SCRIPT.md`)

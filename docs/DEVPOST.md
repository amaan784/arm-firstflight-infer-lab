# Devpost submission draft — ready to paste

> Fill the `⟨⟩` placeholders with the REAL numbers from your `arm-bench` →
> `kleidiai-before-after` CI run before submitting. Everything else is ready.

## Project name
**Arm FirstFlight — Inference Optimization Lab**

## Elevator pitch (tagline)
Measure, optimize, and prove CPU LLM inference on Arm Neoverse — seven measured optimization
axes, a quality guardrail, and a one-click reproducible before/after report.

## Inspiration
Agentic and RAG apps live or die on **time-to-first-token**, and on long contexts that wait is
pure **prefill**. Meanwhile more and more inference runs on Arm cloud CPUs (Graviton, Axion,
Cobalt) because the economics are unbeatable. We wanted to answer, with evidence rather than
vibes: *how much faster and cheaper does Arm-specific optimization actually make this — and
does it cost any accuracy?*

## What it does
`firstflight` is a pip-installable harness — a production-ready developer workflow for
CPU-based inference through quantization on standard CPU-only Arm64 cloud instances — that:
- **Measures** prefill/TTFT scaling across context lengths (128 → 32k tokens) with warm-ups,
  repeats, and variance (fixed seeds + greedy decoding in the generation/quality probes) —
  plus *measured* TTFT from llama-server's own timings and concurrency throughput at 1–8
  parallel requests.
- **Optimizes** along seven Arm-specific axes: KleidiAI microkernels (build flag), quant
  scheme chosen for the silicon (Q4_0 vs the default Q4_K_M), thread pinning, quantized
  KV-cache, prefill micro-batch, `-mcpu=native` build targeting, and prompt/prefix caching.
- **Proves it**: KleidiAI activation is detected from the load log (not assumed), every config
  runs a 40-item exact-match quality guardrail, and costs use real dated AWS prices.
- **Reports it**: a self-contained one-page HTML report — headline delta, charts, before/after
  tables, hotspots — generated automatically and rendered into the GitHub Actions run summary.

**Headline result (⟨replace with real CI numbers⟩):** ⟨X⟩× faster prefill at a 32k-token
context — TTFT ⟨A⟩s → ⟨B⟩s, $⟨C⟩ → $⟨D⟩ per million prompt tokens on c8g.2xlarge, quality
held ⟨n/N⟩ → ⟨n/N⟩ — by switching to a KleidiAI Q4_0 build.

> **Gallery note (don't paste this):** embed the report screenshot and the before/after table
> as images in the Devpost gallery. The rules say judges "are not required to test the Project
> and may choose to judge based solely on the text description, images, and video" — the
> submission must carry the evidence standalone.

## How we built it
Python CLI over llama.cpp's own tools (`llama-bench`, `llama-cli`, `llama-server`,
`llama-batched-bench`) — every external fact (flags, JSON schemas, GGUF URLs, runner labels,
prices) verified against live sources or real binaries and dated in the repo. The CI workflow
builds llama.cpp **three ways** (baseline, `-DGGML_CPU_KLEIDIAI=ON`, `-mcpu=native`) on a free
`ubuntu-24.04-arm` runner and runs the full evidence suite in one click. Arm Performix is
wrapped behind the documented `apx` recipe flow (sourced from Arm's own MCP server), and an
opt-in agent closes the propose→measure loop.

## Challenges we ran into
- KleidiAI only accelerates **Q4_0/Q8_0** — the popular Q4_K_M default silently opts out of
  Arm acceleration. We turned that gotcha into a measured axis and documented it with sources.
- Quantized KV-cache **without flash attention can be slower than f16** — our KV experiment
  pins FA on for both sides so the comparison is methodologically sound.
- Honest measurement is hard: we caught and fixed peak-RSS attribution bugs, sampler noise in
  the quality probe (now greedy), a cost metric that mismatched the prefill headline, and a
  prompt-cache server harness that could deadlock on a full stdout pipe.

## Accomplishments we're proud of
- A judge can go from zero to a full Arm before/after report with **one workflow click** — no
  hardware, and the report appears in the run summary itself.
- Every claim is either measured on the same instance with variance shown, or clearly labeled
  synthetic until the real run lands. The report *enforces* this with an automatic DEMO banner.
- The stage-by-stage `versions/` build (v1→v7, each independently runnable and tested) doubles
  as learning-ready content for anyone assembling a benchmark rig.

## What we learned
Prefill is compute-bound and rewards Arm's matrix instructions heavily; generation is
bandwidth-bound and rewards cache/quant choices; and the single biggest TTFT lever for agent
serving isn't a kernel at all — it's **prefix caching**, which we measure server-side.

## What's next
Land Performix hotspot attribution on a real box run, extend the headline runs to the verified
1.5B model, and upstream the three-build CI recipe as a template other llama.cpp-on-Arm
projects can adopt.

## Setup Instructions (build / run / validate on Arm64)

**Any machine — real inference in three commands (no compiler; run in a fresh venv or conda
env):**
```bash
pip install -e ".[report,dev]"
firstflight setup-engine     # downloads the prebuilt llama.cpp for YOUR platform
firstflight smoke            # real model download + one real generation; `pytest` = 94 tests
```

**Path A — Arm64 in the cloud at zero cost (the judge path):** the repo's `arm-bench`
workflow runs on a free GitHub-hosted **`ubuntu-24.04-arm`** runner (Azure Cobalt 100, Arm
Neoverse N2). GitHub → Actions → **arm-bench** → *Run workflow* builds llama.cpp three ways
(baseline / `-DGGML_CPU_KLEIDIAI=ON` / `-mcpu=native`), runs the headline experiments + the
measured-TTFT prompt-cache demo + the concurrency sweep, and renders the before/after report
**directly into the run summary**. A green run is linked below.

**Path B — Arm VM (bigger models):** on Ubuntu 24.04 aarch64 (AWS Graviton or another
Arm-based cloud instance):
```bash
bash scripts/setup_arm_vm.sh && . .venv/bin/activate && make bench && make report
```

**Validate (not just run):** the report's `kleidiai` column proves the KleidiAI kernels were
**active** (grepped from the model-load log, never assumed); the quality column shows the
40-item exact-match probe held (n/N before → after); every result JSON records host, build,
threads, and THP mode so any number can be traced to its exact configuration.

## Built with
`python` · `llama.cpp` · `kleidiai` · `arm-performix` · `github-actions` (ubuntu-24.04-arm) ·
`matplotlib` · `aws-graviton`

## Links
- Repo: ⟨public GitHub URL⟩ (MIT)
- One-click Arm run — serves as the rules' required "functioning demo / test build" access:
  ⟨link to a green arm-bench workflow run⟩
- Video: ⟨link⟩ (script: `docs/DEMO_SCRIPT.md`)

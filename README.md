# Arm FirstFlight — Inference Optimization Lab

> Reproducible benchmark **+** optimization harness for CPU LLM inference on **Arm Neoverse**
> cloud servers — focused on the metric agentic/RAG apps actually feel: **time-to-first-token
> (TTFT) / prefill latency on long contexts**.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<!-- After pushing, replace OWNER/REPO and uncomment:
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![arm-bench](https://github.com/OWNER/REPO/actions/workflows/arm-bench.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/arm-bench.yml)
-->

`firstflight` profiles a workload (with **Arm Performix**), applies **Arm-specific
optimizations** (KleidiAI int4 kernels, thread pinning/affinity, quantization schemes,
KV-cache settings), and **auto-generates a one-page before/after report** with tokens/sec,
TTFT, peak memory, a small quality-delta eval, and **cost per million tokens**.

Built for the **Arm Create: AI Optimization Challenge (Cloud AI track)**.

> **Status: feature-complete.** `setup-engine` (auto-fetch prebuilt llama.cpp for ANY platform)
> → `smoke` → `bench` (prefill/TTFT sweep) → `ttft` (measured prompt-cache TTFT) →
> `throughput` (concurrency) → `profile` (Arm Performix hooks) → `experiment` (8 before/after
> experiments + quality eval) → `report` (standalone HTML). **One click in CI** runs the whole
> evidence suite on a free Arm runner and renders the report straight into the run summary. Off-Arm, Arm-only tools degrade
> gracefully and say what is skipped. A sample report lives in [`bench/reports/`](bench/reports/)
> — illustrative **synthetic** data until a real Arm run lands.

---

## 1. Project Overview

> **Headline (illustrative example — synthetic perf data, real instance price):** **1.54× faster
> prefill at a 32k-token context** — TTFT **63.0s → 41.0s**, prompt-token cost **\$0.170 →
> \$0.111 per million** (generation: \$1.61 → \$1.04, at the real c8g.2xlarge \$0.319/hr), with
> **quality held (32/40 → 32/40)** — by switching to a **KleidiAI Q4_0** build, with KleidiAI
> **proven active** via load-log detection. _Real numbers: run the `arm-bench` workflow's
> **kleidiai-before-after** job (free Arm runner, report lands in the run summary) or `make
> bench && make report` on an Arm box. Regenerate this example with `make report-demo`._

![Prefill TTFT vs context length — illustrative synthetic example](bench/reports/headline-prefill-ttft.png)

**What it is.** A small, rigorous, reproducible harness that measures where Arm CPU LLM
inference actually hurts — **prefill / TTFT as context length grows** — and proves how much
Arm-specific optimization recovers, in numbers a judge can re-run from CI.

## The optimization — baseline → changes → evidence

**Baseline:** stock llama.cpp CPU build, k-quant weights (Q4_K_M), default threading, on an
Arm Neoverse cloud instance. This is what most people deploy — and its long-context prefill is
the part users feel as time-to-first-token. It mirrors the Cloud AI track's first-listed
Learning Path, ["Deploy an LLM chatbot with llama.cpp using KleidiAI on Arm
servers"](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/)
(Ubuntu 24.04, Graviton, GGUF, `-mcpu=native`) — notably, that official path never passes an
explicit KleidiAI cmake flag or verifies the kernels engaged. We pin the tag, enable
`-DGGML_CPU_KLEIDIAI=ON` explicitly, **prove activation from the load log**, and measure the
delta — closing that gap is the optimization story.

**What we changed (each one Arm-specific, each one measured):**

| # | Change | Mechanism on Arm | Where in this repo |
|---|---|---|---|
| 1 | **KleidiAI microkernels** — rebuild llama.cpp with `-DGGML_CPU_KLEIDIAI=ON` | Arm's KleidiAI kernels repack **Q4_0/Q8_0** weights at load and route matmuls through **dotprod / i8mm / SME2** paths on Neoverse | built + compared automatically in [`arm-bench.yml`](.github/workflows/arm-bench.yml) (`kleidiai-before-after`), [`scripts/setup_arm_vm.sh`](scripts/setup_arm_vm.sh), [`docker/Dockerfile.arm64`](docker/Dockerfile.arm64) |
| 2 | **Quantization scheme chosen for the silicon** — Q4_0 instead of Q4_K_M | KleidiAI accelerates **Q4_0, not k-quants** (verified in ggml's kleidiai source) — the "default" quant silently opts out of Arm acceleration | [`configs/experiments.yaml`](configs/experiments.yaml) `quant-sweep` + `kleidiai` |
| 3 | **Thread count + CPU pinning** | `llama-bench -C/--cpu-mask + --cpu-strict` affinity on Neoverse cores | `thread-sweep` / `pinning` experiments |
| 3b | **Quantized KV-cache** — q8_0 cache vs f16 (`-ctk/-ctv`, flash-attn pinned on) | halves KV-cache footprint and eases the memory-bandwidth pressure that dominates long-context prefill on Neoverse | `kv-cache` experiment |
| 3c | **Prefill micro-batch + flash attention** — `-ub` 256→2048 sweep, `-fa` on/off | bigger micro-batches feed the KleidiAI/i8mm GEMM kernels larger tiles (the dominant prefill lever) | `prefill-batch` / `flash-attn` experiments |
| 3d | **Prompt/prefix caching** — llama-server `cache_prompt` + `--cache-reuse` | for agentic/RAG serving with a shared system prefix, the warm turn skips almost the entire prefill — **measured** TTFT collapse, not derived | `firstflight ttft` (server's own `timings`) |
| 4 | **Performix-ready profiling hooks + opt-in agent loop** — the wrapper implements Arm's documented `apx code_hotspots` recipe flow and renders top hotspots into the report **when `apx` is present** (sample report shows clearly-labeled demo hotspots until a box run lands); the `autotune` agent closes the propose→measure loop | attribution machinery is built and CI attempts it on every headline run (clean no-op without `apx`) | [`src/firstflight/profile/performix.py`](src/firstflight/profile/performix.py), [`src/firstflight/autotune/agent.py`](src/firstflight/autotune/agent.py) |

**The evidence (not just the pitch):**
- **Same instance, same model, only the optimization varies**; warm-ups and repeats with
  variance (5 for the full prefill sweep, 3 for the experiment suite, 2 for the every-push CI
  smoke), fixed seeds + greedy decoding in the generation/quality probes —
  [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).
- **KleidiAI is *proven* active**, not assumed — the harness greps the load log for the
  `CPU_KLEIDIAI` buffer marker and prints yes/no in the report's `kleidiai` column.
- **Quality guardrail per config** — an exact-match probe shows the speedup didn't buy speed
  with accuracy.
- **$/M-token from real prices** (AWS pricing feed, dated in [`configs/instances.yaml`](configs/instances.yaml)).
- **One click reproduces it**: the `kleidiai-before-after` CI job builds llama.cpp three ways,
  runs the five headline experiments (KleidiAI before/after, build-flags, quant sweep, KV-cache,
  micro-batch) plus the measured-TTFT and concurrency sweeps on a free Arm runner, and writes
  the report into the run summary. (Thread/pinning/flash-attn sweeps run anywhere via
  `firstflight experiment --name …`.)

**Reusable beyond this repo:** the pip-installable harness itself; the three-build
(baseline / KleidiAI / -mcpu=native) before/after **CI recipe**; the **Performix `apx` wrapper** (the CLI flow is otherwise
under-documented — sourced from Arm's MCP server); the **KleidiAI quant-support gotcha**
(Q4_0-only) written down with sources; the measurement methodology; and the stage-by-stage
[`versions/`](versions/) build for anyone learning to assemble a rig like this.

**Why it should win.**
- **Technological Implementation:** a real, non-trivial Arm optimization axis (KleidiAI Q4_0
  kernels, thread pinning, quant schemes) with *measured* deltas and variance, profiled by
  Arm Performix — and negative results (k-quants skip KleidiAI) documented, not hidden.
  Performix is the organizers' explicitly recommended measurement tool for Neoverse/cloud
  submissions ("use Arm Performix to measure and validate the impact of your optimizations"
  — [challenge Getting Started
  update](https://arm-ai-optimization-challenge.devpost.com/updates)).
- **WOW:** a genuinely nice auto-generated one-page HTML report, led by the headline number.
- **Potential Impact:** TTFT on long contexts is the cost/UX bottleneck for agentic & RAG
  workloads on cloud CPUs — exactly where Graviton/Ampere economics matter; every artifact
  above is reusable on any llama.cpp-on-Arm deployment.
- **Developer Experience:** `pip install -e . && firstflight setup-engine && make smoke` on any
  machine; `make bench && make report` for the full story; reproducible from CI with one click.

## 2. Functionality / Output

| Command | What it does |
| --- | --- |
| `firstflight setup-engine` | Auto-downloads the **prebuilt llama.cpp** release for THIS platform (Win/Linux/mac, x64/arm64) into `./engine` — real inference everywhere, no compiler. |
| `firstflight info` | Environment report: arch, Arm detection, llama.cpp binary, config summary. |
| `firstflight smoke` | Downloads the tiny Qwen2.5-0.5B GGUF and runs llama.cpp once — proves the pipeline on any machine (skips cleanly with a message if no binary). |
| `firstflight download` | Downloads model GGUFs from `configs/models.yaml` into `./models` only — pre-cache before offline runs (other commands also fetch on demand). |
| `firstflight bench` | Prefill/TTFT + generation throughput + peak memory across context lengths (`configs/workloads.yaml`) via `llama-bench` → `bench/results/*.json`. `--dry-run` prints the command. |
| `firstflight run` | A single (context-length, gen) point for quick iteration. |
| `firstflight ttft` | **Measured** TTFT via `llama-server`'s own `timings` + the prompt-cache demo: same long prefix, cold vs warm turn → prefill-time collapse with `--cache-reuse`. |
| `firstflight throughput` | **Concurrency axis**: aggregate tok/s at 1/2/4/8 parallel requests via `llama-batched-bench` — the serving-throughput story behind the agentic/RAG pitch. |
| `firstflight profile` | Runs Arm Performix (`apx` recipe `code_hotspots`) on a prefill run and surfaces the top hotspot functions into the report; no-ops cleanly off Arm. |
| `firstflight experiment` | The **optimization axis**: benchmarks a set of configs from `configs/experiments.yaml` (quant scheme / threads / CPU pinning / KleidiAI build) **holding model + instance fixed**, runs a quality probe per config, **detects whether KleidiAI is actually active** (load-log proof), and renders the before/after report — proving the delta *and* that accuracy held. |
| `firstflight report` | Renders the before/after **markdown + standalone HTML** report (headline, charts, $/M-token, quality) from `bench/results/*.json` → `bench/reports/`. `--demo` previews the layout with synthetic data. |
| `firstflight autotune --enable` | _(stretch, opt-in)_ Agent-in-the-loop optimizer: proposes a config → benchmarks → loops until no improvement. Default heuristic-grid proposer (no API key); `--llm` uses Claude (the `[agent]` extra). |

**Final output:** the optimized config **+** the auto-generated report
(`bench/reports/*.html` and `*.md`), plus structured results in `bench/results/*.json`.

## 3. Setup Instructions

### Any machine — REAL inference in three commands (no compiler needed)
```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
# or conda: conda create -n firstflight python=3.12 -y && conda activate firstflight
pip install -e ".[report,dev]"
firstflight setup-engine     # auto-downloads the prebuilt llama.cpp for THIS platform
make smoke                   # real model download + real generation, on Win/Linux/mac, x64/arm64
```
Without an engine (`setup-engine` not run, no `LLAMA_CPP_BIN`), everything still **skips
cleanly with a clear message** — nothing crashes.

### Path A — Free Arm execution via GitHub Actions (the judge path)
Push to a public GitHub repo and dispatch the **`arm-bench`** workflow
(`.github/workflows/arm-bench.yml`) on a free GitHub-hosted **`ubuntu-24.04-arm`** runner.
These runners are the subject of [Arm's own Learning
Path](https://learn.arm.com/learning-paths/cross-platform/github-arm-runners/) (native Arm64
execution, no emulation); the free public-repo runners are **Azure Cobalt 100** (Arm Neoverse
N2, 4 vCPU, Armv9-A + SVE2) — the same hyperscaler cloud silicon the Cloud AI track names.
The hackathon provides no cloud credits or hosted environment, so this path exists to let
anyone — including judges — reproduce the numbers at zero cost:

- **`smoke-arm`** (every push to `main` touching `src/`/`configs/`, and on every dispatch):
  real inference + baseline prefill sweep + report.
- **`kleidiai-before-after`** (one click): builds llama.cpp **three ways** (baseline,
  `-DGGML_CPU_KLEIDIAI=ON`, `-mcpu=native`) and runs the full evidence suite on the same
  Q4_0 model: KleidiAI before/after with **active-detection** + quality guardrail, the
  Q8_0/Q4_K_M/Q4_0 quant sweep, the KV-cache and ubatch experiments, the build-flags
  comparison, the **measured-TTFT prompt-cache demo**, and the **concurrency sweep**.

Both render the report **directly into the workflow run summary** (no artifact download needed
to see the numbers) and upload the standalone HTML report + JSON results as an artifact.

### Path B — Remote Arm VM (for bigger models)
On a fresh Ubuntu 24.04 **aarch64** instance — AWS Graviton `t4g`/`c7g`/`c8g` (primary;
matches Arm's Learning Paths), Google Axion `c4a` or Azure Cobalt (also named by the track),
or Oracle Ampere A1 (budget/Always-Free option):
```bash
bash scripts/setup_arm_vm.sh           # build llama.cpp (+KleidiAI) + deps; prints the Performix install-guide pointer (manual step)
. .venv/bin/activate                   # picks up firstflight + LLAMA_CPP_BIN
make bench && make report              # run + render
```
Or drive it from your laptop over SSH:
```bash
bash scripts/run_remote.sh --host user@<arm-ip> --key ~/.ssh/id_ed25519
```

### Reproducibility
The whole story is **one command** on an Arm box: `make bench && make report`. Measurement
protocol (warm-ups, fixed seeds, repeats, variance, same-instance before/after) is documented
in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Repo layout
```
configs/        models.yaml · instances.yaml · workloads.yaml · experiments.yaml
src/firstflight/ cli · runner · engines/ · profile/ · bench/ · eval/ · cost · report/ · autotune/
bench/          results/ (JSON) · reports/ (generated)
docker/         Dockerfile.arm64 (reproducible aarch64 env)
scripts/        setup_arm_vm.sh · run_remote.sh
docs/           EXPLAINER.md (concepts) · METHODOLOGY.md · RUNBOOK.md · DEMO_SCRIPT.md · DEVPOST.md (submission draft) · CONFIRM_ON_ARM.md
versions/       v1..v7 — the build sliced into cumulative, independently runnable stages
.github/workflows/ ci.yml (lint+test) · arm-bench.yml (free Arm benchmarks)
```
(`engine/` and `models/` are created at runtime by `setup-engine`/`download` and stay untracked.)

## License
[MIT](LICENSE). Open source, as required by the hackathon.

_Official challenge channels: [Arm Developer Program](https://developer.arm.com) ·
[Arm Learning Paths](https://learn.arm.com) ·
[Arm Developer Ecosystem GitHub](https://github.com/ArmDeveloperEcosystem) ·
the Arm Developer Program Discord._

---
_Some Arm-box-only specifics (exact Performix `apx` subcommands & output parsing, KleidiAI
flags for the pinned llama.cpp tag, instance hourly prices) are marked `TODO(confirm)` in the
code and verified on the live Arm box rather than invented. The consolidated list — and what
was already verified — is in [`docs/CONFIRM_ON_ARM.md`](docs/CONFIRM_ON_ARM.md)._

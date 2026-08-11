# Arm FirstFlight — Inference Optimization Lab

> ### Your KleidiAI speedup is measured against the wrong baseline.
>
> A stock llama.cpp build already has Arm acceleration switched on — `GGML_NATIVE` and
> `GGML_CPU_REPACK` both default to ON. So the standard "KleidiAI on vs off" test credits
> KleidiAI with work ggml's own aarch64 kernels were already doing. And on Q4_K_M, the quant
> most people actually download, KleidiAI's kernels never engage at all.
>
> **FirstFlight builds the real floor, splits the speedup by mechanism, proves the kernels ran
> — and refuses to report a win it can't defend.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
<!-- After pushing, replace OWNER/REPO and uncomment:
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![arm-bench](https://github.com/OWNER/REPO/actions/workflows/arm-bench.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/arm-bench.yml)
-->

Built for the **Arm Create: AI Optimization Challenge (Cloud AI track)**.

---

## 1. Project Overview

### The problem with the standard benchmark

Everyone measures KleidiAI the same way: build llama.cpp normally, build it again with
`-DGGML_CPU_KLEIDIAI=ON`, compare. That test is **not measuring what it claims to measure**,
for two reasons that are checkable in five minutes:

1. **The "before" build is already Arm-optimized.** In llama.cpp's own `ggml/CMakeLists.txt`,
   `GGML_NATIVE` defaults **ON** (native targeting, the `-mcpu=native`-equivalent) and
   `GGML_CPU_REPACK` defaults **ON** (ggml's own aarch64 Q4_0 repack GEMM — the same trick
   KleidiAI does). So the comparison is one Arm optimization against another, and whatever
   number falls out gets credited entirely to KleidiAI.
2. **On the most-downloaded quant, the kernels never run.** KleidiAI has kernels for Q4_0 and
   Q8_0 only. Point it at a Q4_K_M GGUF — the default almost everyone downloads — and it
   silently falls back. Upstream now logs exactly that
   ([PR #25701](https://github.com/ggml-org/llama.cpp/pull/25701), merged 2026-07-21).

### What this harness does instead

It builds the **true unaccelerated floor** (`GGML_NATIVE=OFF`, `GGML_CPU_ARM_ARCH=armv8-a`,
`GGML_CPU_REPACK=OFF`) and measures a **three-rung ladder** against it, so every speedup names
its own mechanism:

```
generic armv8-a  →  + native targeting & ggml repack  →  + KleidiAI kernels
```

Then it does three things a benchmark normally won't:

- **Proves the kernels engaged** — reads the weight-buffer line and ISA flags out of the real
  model-load log, and names the tier that ran (I8MM / DOTPROD / SVE2 / NEON). No marker, no claim.
- **Runs a negative control** — the KleidiAI build against Q4_K_M, where its kernels *cannot*
  engage. The probe must report INACTIVE. If it ever doesn't, the detection is broken and every
  other KleidiAI number here is void.
- **Refuses to claim wins inside its own noise.** The same build is measured twice under two
  labels to establish a floor; any delta that doesn't clear it is reported as
  **"within noise — not claimed"** instead of as a multiplier.

Measured on Arm Neoverse, around the metric agentic and RAG apps actually feel:
**time-to-first-token on long contexts**.

> **Numbers:** this README ships with **no headline figure**. The one-click
> [`arm-bench`](.github/workflows/arm-bench.yml) job produces the real ladder, noise floor and
> negative control on a free `ubuntu-24.04-arm` runner and renders the report into the run
> summary. A synthetic sample report lives in [`bench/reports/`](bench/reports/) to show the
> layout — its charts carry a **SYNTHETIC DEMO DATA** watermark burned into the pixels, because
> a project that refuses unproven claims cannot lead with illustrative data.

**What it is.** A small, rigorous, reproducible harness that measures where Arm CPU LLM
inference actually hurts — **prefill / TTFT as context length grows** — and proves how much
Arm-specific optimization recovers, in numbers a judge can re-run from CI.

```mermaid
flowchart LR
    A["setup-engine\nprebuilt llama.cpp"] --> B["bench\nprefill/TTFT sweep"]
    A --> C["ttft\nmeasured, prompt cache"]
    A --> D["throughput\n1-8 parallel"]
    subgraph CI["free Arm CI — ubuntu-24.04-arm (3 builds: generic / repack / KleidiAI)"]
        B --> E["experiment\nladder + quality + ppl"]
        E --> F["report\nHTML + MD, in run summary"]
        C --> F
        D --> F
    end
    G["profile\nArm Performix apx"] -.-> F
```

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
| 1 | **KleidiAI microkernels** — rebuild llama.cpp with `-DGGML_CPU_KLEIDIAI=ON` | Arm's KleidiAI kernels repack **Q4_0/Q8_0** weights at load and route matmuls through **dotprod / i8mm / SVE2** paths on Neoverse (KleidiAI also ships SME2 kernels, but no shipping Neoverse server core implements SME as of 2026-08 — it's client silicon only) | built + compared automatically in [`arm-bench.yml`](.github/workflows/arm-bench.yml) (`kleidiai-before-after`), [`scripts/setup_arm_vm.sh`](scripts/setup_arm_vm.sh), [`docker/Dockerfile.arm64`](docker/Dockerfile.arm64) |
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
- **One click reproduces it**: the `kleidiai-before-after` CI job builds llama.cpp three ways
  (generic armv8-a floor / native+repack default / KleidiAI), runs the attribution ladder in
  **3 interleaved rounds** plus a **same-build noise-floor control**, the build-flags, quant,
  KV-cache and micro-batch experiments, and the measured-TTFT and concurrency sweeps on a
  free Arm runner — and writes the report into the run summary. (Thread/pinning/flash-attn
  sweeps run anywhere via `firstflight experiment --name …`.)
- **Attribution, not vibes**: a plain llama.cpp Release build already carries native targeting
  **and** ggml's own aarch64 Q4_0 repack kernels (`GGML_NATIVE`/`GGML_CPU_REPACK` default ON —
  verified in b9873's CMakeLists). So the ladder measures each mechanism on its own rung:
  generic → repack → KleidiAI, with the same-build spread published next to it.

## Why this exists — what the alternatives can't tell you

| Where people look today | What it gives you | What it can't say |
|---|---|---|
| `llama-bench` | An honest tokens/sec number | *Which* mechanism produced it. Native targeting, ggml's repack and KleidiAI all land in one figure |
| The [AWS Graviton llama.cpp guide](https://github.com/aws/aws-graviton-getting-started) | `-mcpu=native`, and it works | Never mentions KleidiAI at all, so the kernel question is never asked |
| [Arm's own KleidiAI Learning Path](https://learn.arm.com/learning-paths/servers-and-cloud-computing/llama-cpu/) | A working chatbot on Graviton | Never passes the KleidiAI cmake flag, and never A/Bs against a build without it |
| Vendor blog posts ("up to 4× on Graviton") | A headline multiplier | Build config usually undisclosed — silicon, flags, repack or kernels? Unknowable |
| General LLM benchmark suites | Cross-engine comparisons | Backend-agnostic by design: no Arm build-variant axis, no proof a kernel engaged |

This harness answers the question those leave open: **which mechanism, how much, and prove it.**
Each rung of the ladder is a separate build, each speedup names its cause, activation is read
from the load log rather than assumed, and a delta inside the measured noise floor is reported
as *no win* rather than as a number.

**Reusable beyond this repo:** the pip-installable harness itself (adoption recipes:
[`docs/ADOPT.md`](docs/ADOPT.md)); a **drop-in Arm CI template**
([`docs/arm-bench-template.yml`](docs/arm-bench-template.yml)) that runs the three-build ladder
in any repo; the **Performix `apx` wrapper** (sourced from Arm's MCP server); the
**KleidiAI quant-support constraint** (Q4_0/Q8_0 only) written down with sources; the
measurement methodology; and the **commit history itself** — the repo was committed stage by
stage (foundation → benchmark → report → profiling → experiments → autotuner → integration →
hardening), so stepping through the commits replays how a rig like this gets assembled.

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
- **Developer Experience:** `pip install -e . && firstflight setup-engine && firstflight smoke`
  on any machine; `make bench && make report` for the full story; reproducible from CI.

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
| `firstflight perplexity` | Perplexity over a fixed corpus via `llama-perplexity` — the finer quality instrument (the 40-item probe can't resolve ~1% shifts). Also runs automatically per experiment config. |
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
firstflight smoke            # real model download + real generation, Win/Linux/mac, x64/arm64
```
(`make smoke` etc. work on Linux/macOS/Git Bash; plain `firstflight` commands work everywhere.)
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
- **`kleidiai-before-after`** (one dispatch): builds llama.cpp **three ways** — a **generic
  armv8-a floor** (`GGML_NATIVE=OFF`, `GGML_CPU_REPACK=OFF`), the **native+repack default**,
  and the **KleidiAI** build — and runs the full evidence suite on the same Q4_0 model: the
  attribution ladder (3 interleaved rounds, up to 16k context) with **active-detection** +
  quality guardrail + perplexity, the **same-build noise-floor control**, the
  Q8_0/Q4_K_M/Q4_0 quant sweep, the KV-cache and ubatch experiments, the build-flags
  comparison, the **measured-TTFT prompt-cache demo**, and the **concurrency sweep**.

Both render the report **directly into the workflow run summary** (no artifact download needed
to see the numbers) and upload the standalone HTML report + JSON results as an artifact.

> **For judges / anyone without write access:** `Run workflow` needs write permission, so
> fork the repo first, enable workflows on the fork when GitHub asks, then dispatch
> **arm-bench** there — same free Arm runners, same report in the run summary.

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
docs/           EXPLAINER.md (concepts) · METHODOLOGY.md · RUNBOOK.md · ADOPT.md (+ CI template) · DEMO_SCRIPT.md · DEVPOST.md (submission draft) · CONFIRM_ON_ARM.md
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

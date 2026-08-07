# Demo Script (~2:45)

A tight walkthrough for the submission video. Target: **~2:45** (slack before the 3:00 cap).
Times are cues. Record the terminal beats in advance so playback is instant.

**Rules constraints (from the official rules):** host the video publicly on YouTube, Vimeo,
or Youku; keep it under 3:00 (judges "are not required to watch beyond three minutes");
include footage of the project actually running on the Arm64 environment it targets (a
terminal on the Arm runner/VM and the Actions run summary count); no third-party trademarks
or copyrighted music.

---

**[0:00–0:20] Hook — the problem.**
> "Agentic and RAG apps live or die on *time-to-first-token*. On long contexts, that's
> dominated by **prefill** — and a huge amount of cloud LLM inference now runs on **Arm**
> CPUs: Graviton, Axion, Cobalt. So: how fast can we make prefill on Arm, and what does it
> cost?"

**[0:20–0:40] What it is.**
> "Arm FirstFlight is a reproducible benchmark + optimization harness. It measures prefill/TTFT
> across seven Arm-specific optimization axes — KleidiAI kernels, quant choice, pinning,
> KV-cache, micro-batch, build flags, prompt caching — and auto-generates a one-page
> before/after report with a quality guardrail and cost per million tokens."

**[0:40–1:00] Dead-simple repro (screen capture — 3 commands).**
> ```bash
> pip install -e ".[report,dev]"
> firstflight setup-engine    # prebuilt llama.cpp for THIS platform — no compiler
> make smoke                  # REAL inference on any machine
> ```
> "Three commands, real inference, any laptop. On an Arm box, `make bench && make report`
> is the whole story."

**[1:00–1:15] Measured TTFT beat.**
> Run `firstflight ttft` (pre-recorded). Point at the cold vs warm table:
> "Same 2,000-token context, second turn — the server's own measured prefill time collapses,
> because the prompt cache re-uses the shared prefix. Measured, not derived."

**[1:15–1:30] Concurrency beat.**
> Run `firstflight throughput` (pre-recorded). Point at the parallel 1→8 curve:
> "And because agent workloads are concurrent, here's aggregate throughput at one to eight
> parallel requests."

**[1:30–2:10] The report — the WOW.**
> Open the generated HTML report. Lead with the **headline**: before/after **prefill TTFT** at
> long context. Walk the prefill-scaling chart, the ± variance, the **quality held** column,
> the **$/M prompt tokens** cost, the KleidiAI **proven-active** column, and the measured-TTFT
> + concurrency sections.

**[2:10–2:30] One click on real Arm.**
> Show the **GitHub Actions run**: "one click builds llama.cpp three ways — baseline, KleidiAI,
> `-mcpu=native` — runs the headline experiments on a free Arm runner, and the report lands
> right in the run summary." Scroll the summary.

**[2:30–2:45] Close.**
> "Open source, MIT, every number reproducible. That's Arm FirstFlight."

---

### Shot list
- Terminal: `firstflight setup-engine`, `make smoke`, `firstflight ttft`, `firstflight throughput`.
- Browser: the standalone HTML report (headline → charts → measured TTFT → concurrency).
- CI tab: the `arm-bench` workflow green, run summary scrolled — *free, reproducible Arm run*.

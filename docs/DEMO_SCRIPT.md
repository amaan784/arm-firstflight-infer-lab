# Demo Script (~2:40)

The video makes one argument rather than touring features: the standard KleidiAI benchmark is
measured against the wrong baseline, and this harness catches it (including catching itself).

Record the terminal beats in advance (asciinema + agg, or VHS) so playback is instant and
typo-free. Beats 4 and 5 are the point of the video; everything else is setup.

**Rules constraints (from the official rules):** host publicly on YouTube, Vimeo or Youku; keep
under 3:00 (judges are not required to watch beyond three minutes); include footage of the
project running on the Arm64 environment it targets (terminal on the Arm runner and the Actions
run summary both count); no third-party trademarks or copyrighted music.

> Script every number against the real downloaded results, never a live run. Nothing in this
> video should be measured on camera.

---

**[0:00–0:20] The claim everybody makes.**
> "If you search how to speed up llama.cpp on Arm, you get one answer: rebuild it with KleidiAI
> and measure before and after. So that's what I did, and I got a number. Then I checked what
> that 'before' build was."

**[0:20–0:45] The problem, on screen.**
> Show `ggml/CMakeLists.txt` with the two default lines visible.
> "A stock llama.cpp Release build already has `GGML_NATIVE` on (native targeting) and
> `GGML_CPU_REPACK` on, which is ggml's own Arm kernel doing the same repacking trick KleidiAI
> does. The standard test compares Arm-optimized against Arm-optimized, then credits the whole
> difference to KleidiAI."

**[0:45–1:20] The fix: build the real floor.**
> Show the three build steps in `arm-bench.yml`, then the ladder running.
> "So FirstFlight builds the floor that nobody builds: native off, repack off, plain armv8-a.
> Then three rungs on the same runner, same model: generic, plus ggml's repack, plus KleidiAI.
> Now every step has a name."
> Cut to the report's ladder section with the real per-rung numbers.

**[1:20–1:50] Proof the kernels ran.**
> Show the kernel-evidence block in the report.
> "It doesn't take the build flag's word for it. This comes out of the model-load log: which
> weight buffer loaded, and which instruction tier ran. i8mm here, on Neoverse N2."

**[1:50–2:20] The harness catching a lie.**
> Show the negative-control experiment and the report row.
> "Here's the same KleidiAI build pointed at a Q4_K_M model, the quant most people download,
> and one KleidiAI has no kernels for. It gets faster. And the harness refuses to credit
> KleidiAI, because the probe says the kernels never engaged."
> Then the noise gate, on screen:
> "Same thing here. This delta didn't clear the measured noise floor, so instead of a
> multiplier it prints 'within noise' and doesn't claim the win. A benchmark that can't say
> *no* isn't measuring anything."

**[2:20–2:40] What you get, and close.**
> Show the run summary in GitHub Actions.
> "One click, a free Arm runner, zero dollars: the full report lands in the run summary.
> Fork the template, point it at your model, get your own defensible number.
> Open source, MIT. That's Arm FirstFlight."

---

## Shot list

| # | Beat | Source |
|---|---|---|
| 1 | The standard benchmark claim | slide or terminal |
| 2 | `GGML_NATIVE` / `GGML_CPU_REPACK` defaults | `ggml/CMakeLists.txt` |
| 3 | Three builds + the ladder | `.github/workflows/arm-bench.yml`, report ladder section |
| 4 | Kernel evidence (buffer + ISA tier) | report "Kernel evidence" section |
| 5 | Q4_K_M negative control: KleidiAI INACTIVE | report row, `kleidiai-null-control` |
| 6 | Noise gate: "within noise, not claimed" | report headline |
| 7 | Run summary on `ubuntu-24.04-arm` | GitHub Actions run page |

## Not in the video

`ttft`, `throughput`, `profile`, `autotune`, `perplexity` and the quant/KV/micro-batch sweeps
are all real and documented, but none of them belong here. Showing them would turn the video
into a feature tour. The video makes one claim and proves it.

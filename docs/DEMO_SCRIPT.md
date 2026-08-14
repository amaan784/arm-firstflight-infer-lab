# Demo video script

Target 2:45, hard cap 3:00. Read it straight through. The bracketed lines are screen cues,
not spoken.

Have these open before you record:

1. `README.md` on GitHub, scrolled to the top
2. The newest Q4_0 report: `bench/reports/report-20260814-215555.html`
   (filenames are timestamped; if you re-render, take the newest `bench/reports/*.html`)
3. The green runs: 31656321896 (Q4_0) and 31784946201 (Q8_0), both under
   https://github.com/amaan784/arm-firstflight-infer-lab/actions

---

**[README, top of the page]**

This is Arm FirstFlight.

Every KleidiAI benchmark works the same way. Build llama.cpp normally, build it again with
KleidiAI on, compare, publish the number.

The problem is that first build.

In llama.cpp's own CMake file, two flags default to on: native targeting, and ggml's own Arm
repack kernels. So your "before" build is already Arm-optimized. You're comparing one Arm
optimization against another, and handing KleidiAI credit for both.

**[scroll to the ladder diagram]**

So we built the baseline nobody builds. Native off. Plain armv8-a. Repack off.

Then three rungs. Same model, same machine, same run. Generic, repack, KleidiAI. Each has to
earn its own number.

We also ran the identical build twice, to see how much the machine wobbles on its own.
That's our noise floor. Zero point three percent.

**[scroll to the Q4_0 ladder table]**

Here's Q4_0.

Repack takes prefill from twenty-six tokens a second to ninety-five. Three point six times
faster.

Then KleidiAI, on top of that. One point zero zero. Nothing.

**[scroll to the kernel evidence table]**

Look at the load log. Three different weight buffers. The floor build can't even reach
the i8mm tier.

So KleidiAI loaded. It just had nothing left to take, because repack got there first.

The standard benchmark would have reported that entire three point six as a KleidiAI win.

**[scroll to the Q8_0 table]**

Now, ggml's repack is built for Q4_0. So we predicted KleidiAI would have room at Q8_0.
We wrote that down first, then ran it.

One point two three at two thousand tokens. One point one three at four thousand. One point
one five on generation.

And at Q8_0 the KleidiAI build reports repack off, KleidiAI on. A different code path, not
more of the same one.

So the usual benchmark is wrong in both directions. It gives KleidiAI credit it didn't earn
at Q4_0, and never tests the quant where it does.

**[scroll to the perplexity column]**

One more thing.

At Q4_0, repack and KleidiAI gave identical perplexity. Same number, six digits.

At Q8_0, KleidiAI shifts it by one point four percent.

So that twenty-three percent speedup isn't free. A speed-only benchmark ships that without
noticing. Ours caught it.

**[the green Actions run, then the Run workflow dialog]**

All of this runs on a free GitHub Arm runner. One click, no hardware. It builds llama.cpp
three ways and writes the report into the run summary.

**[open a running or completed job's live log and scroll the benchmark step for ~10 seconds -
this is the required "project functioning on Arm" footage, so let it breathe]**

That's it executing on `ubuntu-24.04-arm`.

**[the report headline]**

That's Arm FirstFlight. It builds the real baseline, splits the win by mechanism, and refuses
to claim anything inside its own noise.

Even when the honest answer is that nothing happened.

---

## Numbers to get right

| claim | value |
|---|---|
| repack vs generic, Q4_0 @ 1k | 26.4 -> 94.8 tok/s, 3.60x |
| KleidiAI vs repack, Q4_0 | 1.00-1.01x at every context |
| noise floor | 0.3% |
| KleidiAI vs repack, Q8_0 @ 2k | 1.23x |
| KleidiAI vs repack, Q8_0 @ 4k | 1.13x |
| KleidiAI vs repack, Q8_0 generation | 1.15x (39.2 -> 45.2 tok/s) |
| perplexity, Q4_0 repack vs kleidiai | 37.4181 both |
| perplexity, Q8_0 repack vs kleidiai | 31.6267 -> 32.0774 (+1.4%) |

## Do not say

- Don't call the Q4_0 result a failure. It's a measured null with the kernels proven active.
- Don't quote a dollar figure. Both runs are on a free runner priced at $0/hr.
- Don't claim a Q8_0 noise floor. That run skipped it. If you need to qualify the 1.23x, say
  "far outside the nought point three percent floor we measured at Q4_0".
- Don't say 32k context. Q4_0 covers 1k to 8k, Q8_0 covers 2k to 8k.

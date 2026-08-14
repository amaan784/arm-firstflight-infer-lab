# Demo video script

Target 2:45, hard cap 3:00. Read it straight through. The bracketed lines are screen cues,
not spoken.

Have these open before you record:

1. `README.md` on GitHub, scrolled to the top
2. `bench/reports/report-20260814-193627.html` (the Q4_0 report)
3. The green run: https://github.com/amaan784/arm-firstflight-infer-lab/actions/runs/31784946201

---

**[README, top of the page]**

This is Arm FirstFlight.

I want to show you something about KleidiAI benchmarks that I think is wrong.

Here's how everyone measures it. You build llama.cpp normally. You build it again with
KleidiAI switched on. You compare the two, and you publish the number.

The problem is that first build.

In llama.cpp's own CMake file, two flags default to on. Native targeting, and ggml's own
Arm repack kernels. So your "before" build is already Arm-optimized. You're comparing one
Arm optimization against another, and handing KleidiAI the credit for both.

**[scroll to the ladder diagram]**

So we built the baseline nobody builds. Native off. Plain armv8-a. Repack off.

Then three rungs. Same model, same machine, same run. Generic, then repack, then KleidiAI.
Each one has to earn its own number.

We also ran the identical build twice under two different labels, just to see how much the
machine wobbles on its own. That's our noise floor. Nought point three percent.

**[scroll to the Q4_0 ladder table]**

Here's Q4_0.

Repack takes prefill from twenty-six tokens a second to ninety-five. Three point six times
faster.

Then KleidiAI, on top of that. One point zero zero. Nothing.

And that is not a broken test.

**[scroll to the kernel evidence table]**

Look at the load log. Three different weight buffers, three different kernel tiers.
KleidiAI loaded. It just had nothing left to take, because repack got there first.

The standard benchmark would have reported that entire three point six as a KleidiAI win.

**[scroll to the Q8_0 table]**

Now, ggml's repack is built for Q4_0. So we predicted KleidiAI would have room at Q8_0
instead. We wrote that down first, then we ran it.

One point two three at two thousand tokens. One point one three at four thousand. One point
one five on generation.

And at Q8_0, the KleidiAI build reports repack off, KleidiAI on. That's a different code
path, not more of the same one.

So the usual benchmark is wrong in both directions. It gives KleidiAI credit it didn't earn
at Q4_0, and it never tests the quant where it actually does.

**[scroll to the perplexity column]**

One more thing, and this is the part I care about most.

At Q4_0, repack and KleidiAI gave identical perplexity. Same number, six digits.

At Q8_0, KleidiAI shifts it by one point four percent.

So that twenty-three percent speedup isn't free. A benchmark that only measures speed would
ship that without ever noticing. Ours caught it.

**[the green Actions run, then the Run workflow dialog]**

All of this runs on a free GitHub Arm runner. One click, no hardware. It builds llama.cpp
three ways, runs the ladder, and writes the report straight into the run summary.

**[the report headline]**

That's Arm FirstFlight. It builds the real baseline, splits the win by mechanism, proves the
kernels actually ran, and refuses to claim anything inside its own noise.

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
- Don't say 32k context. The runs cover 1k to 8k.

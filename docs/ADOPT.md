# Adopt this in your own project

Four copy-paste recipes. Everything here works from a plain checkout; nothing needs this
repo's models or history.

## 1. Benchmark YOUR model

Add a stanza to `configs/models.yaml` (any GGUF with a direct download URL works):

```yaml
  my-model:
    hf_repo: your-org/Your-Model-GGUF
    description: "Your model."
    variants:
      q4_0:                      # the KleidiAI-accelerated quant — benchmark this one on Arm
        file: your-model-q4_0.gguf
        url: "https://huggingface.co/your-org/Your-Model-GGUF/resolve/main/your-model-q4_0.gguf"
        sha256: ""
```

Then: `firstflight bench --model my-model --variant q4_0` (pick a workload with
`--workload`; `rag-context` is the fast one, `prefill-scaling` the full curve).

## 2. Benchmark YOUR instance

Add it to `configs/instances.yaml` so $/M-token costs are computed from a real price:

```yaml
  my-instance:
    arch: arm64
    cpu: "Arm Neoverse-XX (provider name)"
    vcpus: 16
    usd_per_hour: 0.000          # your region's on-demand price — date it in a comment
    notes: "verified YYYY-MM-DD"
```

Then pass `--instance my-instance` to `firstflight report` / `experiment`.

## 3. Copy the Arm CI recipe into your repo

[`docs/arm-bench-template.yml`](arm-bench-template.yml) is a trimmed, self-contained
workflow: free `ubuntu-24.04-arm` runner, the three-build attribution ladder (generic
armv8-a floor / native+repack default / KleidiAI), cached builds keyed on the llama.cpp
tag, and your benchmark step where the placeholder is. Drop it into
`.github/workflows/`, replace the `RUN YOUR BENCHMARK HERE` step, push to a public repo,
dispatch. The three-build pattern is the important part: a plain Release build is
not a baseline (`GGML_NATIVE` and `GGML_CPU_REPACK` default ON), so build the generic
floor explicitly or your before/after attributes someone else's optimization to your change.

## 4. Reuse the measurement rules

The short version of [`METHODOLOGY.md`](METHODOLOGY.md), portable to any harness:

- Same instance, same model, one variable per comparison; pin `-fa` so flash-attn's auto
  heuristic can't differ between arms.
- Interleave rounds (A,B,A,B) instead of blocking (A,A,B,B); report the median and the
  between-round spread.
- Run a **noise floor**: the same config twice under two labels. If your measured win is
  inside that spread, don't claim it.
- Prove kernel activation from the load log (`CPU_KLEIDIAI` buffer line), never from the
  build flag.
- Pair every speedup with a quality guardrail (exact-match probe + perplexity on a fixed
  corpus).
- Price tokens with a dated, real hourly price, or `0.0` and say so.

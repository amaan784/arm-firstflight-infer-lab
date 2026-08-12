# Confirm-on-Arm checklist

Some specifics change over time or are only knowable on the live Arm box. They are marked
`TODO(confirm)` in the code with a doc link. This is the consolidated list.

## Already verified live

Checked against authoritative sources (dates noted) and baked in. Re-confirm if you pin
a different tag/date:

| Item | Value | Source / date |
| --- | --- | --- |
| Smoke model repo | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (lowercase filenames; q4_0/q4_k_m/q8_0 all present) | huggingface.co, 2026-06-26 + 2026-07-07 |
| llama.cpp repo | `github.com/ggml-org/llama.cpp` (was `ggerganov`) | GitHub, 2026-06-26 |
| llama-bench flags | `-p` prefill, `-n` gen, `-o json`, `-r`, `-t`, `-C/--cpu-mask`, `--cpu-strict`, `--no-warmup` | tools/llama-bench/README.md, 2026-06-26 |
| llama-bench JSON schema | top-level array; `avg_ts`/`stddev_ts`/`n_prompt`/`n_gen`/… | llama-bench.cpp source, 2026-06-26 |
| llama-completion flags | `-t`, `-n`, `-s`, `-no-cnv`, `--no-display-prompt`, `-v` | real b9873 binary `--help`, 2026-08-12 |
| llama-bench KV-cache flags | `-ctk/--cache-type-k`, `-ctv/--cache-type-v` (default f16) | real b9873 binary `--help`, 2026-07-07 |
| CPU build | `cmake -B build && cmake --build build --config Release` | docs/build.md, 2026-06-26 |
| KleidiAI flag | `-DGGML_CPU_KLEIDIAI=ON` | docs/build.md, 2026-06-26 |
| KleidiAI quant support | Q4_0 / Q8_0 only (not Q4_K_M) | ggml-cpu/kleidiai/kleidiai.cpp, 2026-06-26 |
| KleidiAI active marker | `load_tensors: CPU_KLEIDIAI model buffer size = …` (auto-detected by the harness) | docs/build.md, 2026-06-26 |
| Pinned release tag | `b9873` (2026-07-04); all per-platform assets exist incl. `win-cpu-x64.zip` (binaries at zip root) | GitHub Releases, 2026-07-05 |
| GitHub Arm runner | `ubuntu-24.04-arm` (GA, free for public repos) | github.blog 2025-08-07 |
| Performix CLI | `apx` recipe flow (`recipe run code_hotspots` → `run render` → `render query`) | github.com/arm/mcp, 2026-06-26 |
| Instance prices | c8g.2xlarge $0.319/hr, c7g.2xlarge $0.29/hr, t4g.small $0.0168/hr (us-east-1); OCI A1 $0.01/OCPU-hr, Always-Free now ~2 OCPU/12GB | AWS pricing feed + Oracle price-list API, 2026-07-05 |

## Still needs the live Arm box / your input

1. **Performix `apx` run_id/session_id JSON keys + output shape**: `src/firstflight/profile/performix.py`.
   The commands come from Arm's own MCP server; the exact JSON keys and whether
   `recipe run --json` embeds the hotspot table are `TODO(confirm)` against the CLI Reference
   Guide (Arm doc [111566](https://developer.arm.com/documentation/111566)).

2. **Performix install steps**: `scripts/setup_arm_vm.sh`, `docker/Dockerfile.arm64`.
   Follow <https://learn.arm.com/install-guides/performix/>.

3. **Prices for YOUR region/date**: `configs/instances.yaml` holds verified us-east-1 values
   (2026-07-05); re-check for the exact instance/region you benchmark on before submission.

4. ~~Repo URL~~ (done): `pyproject.toml` `[project.urls]` points at
   `amaan784/arm-firstflight-infer-lab`.

5. **Replace the synthetic sample**: run the `arm-bench` workflow (the
   `kleidiai-before-after` job) once and commit its real report over the demo sample.

6. **Pinning CPU mask**: `configs/experiments.yaml` `pinning` uses `cpu_mask: "0x0f"`
   (cores 0–3); set it for the core count of the box you pin on.

7. **bartowski Q4_0 URL**: `configs/models.yaml` marks the bartowski
   `Q4_0.gguf` resolve URL `TODO(confirm)`. Optional: the `kleidiai` experiment already uses
   the official Qwen repo's verified q4_0, so this entry is a spare, not a blocker.

8. **lm-eval task names / endpoint** (only if you use the optional `[eval]` extra):
   `src/firstflight/eval/quality.py:build_lm_eval_cmd` marks the exact task names and
   server endpoint `TODO(confirm)` on the box; the built-in probe needs nothing.

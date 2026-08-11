# Runbook — how to run, test, and ship Arm FirstFlight

The operator's guide, start to finish: local run on any machine → review builds → free Arm CI
→ optional Arm VM → submission. (The README has the project story; this is the hands-on list.)

---

## A. Run everything locally (~10 min, any machine)

From the repo root:

```powershell
# 1. Create + activate an environment — venv:
python -m venv .venv
.\.venv\Scripts\activate          # PowerShell (Linux/mac: . .venv/bin/activate)
#    ...or conda, if you prefer (identical from step 2 on):
# conda create -n firstflight python=3.12 -y
# conda activate firstflight

# 2. Install with report + dev extras
pip install -e ".[report,dev]"

# 3. Unit tests + lint (expect 100+ passed; the same two ruff checks CI runs)
pytest
ruff check .
ruff format --check .

# 4. Get a real llama.cpp (prebuilt for YOUR platform, no compiler)
firstflight setup-engine

# 5. See what's detected
firstflight info

# 6. REAL inference smoke test (first run downloads a ~470 MB model)
firstflight smoke

# 7. Quick real micro-benchmark (one point, fast)
firstflight run --prompt-len 512 --gen 16

# 8. Full prefill/TTFT sweep (longer — several minutes on a laptop)
firstflight bench

# 8b. MEASURED TTFT + prompt-cache demo (starts a local llama-server, ~a minute)
firstflight ttft

# 8c. Concurrency sweep: aggregate tok/s at 1/2/4/8 parallel requests
firstflight throughput

# 9. Render the report and open it
firstflight report            # from your real results in bench/results/
firstflight report --demo     # or the synthetic layout preview
```

Open the newest `bench/reports/report-*.html` in a browser — it's fully standalone.

**PowerShell notes:** if `activate` is blocked by execution policy, either run
`Set-ExecutionPolicy -Scope Process Bypass` first, or skip activation and prefix commands with
`.\.venv\Scripts\` (e.g. `.\.venv\Scripts\firstflight.exe smoke`). Quote paths — this folder
name contains spaces.

## B. Review the build stage-by-stage

Each `versions/vN` is cumulative and independently runnable (v1 foundation → v7 = full project):

```powershell
cd versions\v3          # any stage
python -m venv .venv ; .\.venv\Scripts\activate
# or conda (one env per stage): conda create -n ff-v3 python=3.12 -y ; conda activate ff-v3
pip install -e ".[report,dev]"    # ".[dev]" suffices for v1/v2
pytest
firstflight info
```

Each `vN/README.md` says exactly what that stage adds.

## C. Free Arm execution via GitHub (the real numbers)

1. **Create the repo** (must be **public** — Arm runners are free only for public repos):
   ```powershell
   git init
   git add -A
   git commit -m "Arm FirstFlight — Arm CPU LLM prefill/TTFT optimization harness"
   gh repo create arm-firstflight-infer-lab --public --source . --push
   ```
   (or create it on github.com and `git remote add origin ... ; git push -u origin main`)

2. **Repo URL** in `pyproject.toml` `[project.urls]` — already set (`amaan784/arm-firstflight-infer-lab`).

3. **CI runs automatically on push:** `CI` (lint + tests + demo report, every push) and the
   `arm-bench` **smoke-arm** job (real inference + baseline sweep on an Arm runner; report
   appears in the run's Summary tab). Note smoke-arm is path-filtered — it runs only when
   `src/`, `configs/`, or the workflow file change (a docs-only commit skips it).

4. **The headline run:** GitHub → Actions → **arm-bench** → *Run workflow*.
   This dispatches **kleidiai-before-after**: builds llama.cpp three ways (generic armv8-a
   floor / native+repack default / KleidiAI), runs the attribution ladder (3 interleaved
   rounds) + noise-floor control + quality guardrail + perplexity + KleidiAI-active detection
   + the quant sweep, and renders the report **into the run summary**. The standalone HTML
   report + JSON results are attached as the `arm-headline-*` artifact.
   *(No write access to the repo? Fork it, enable workflows on the fork when GitHub asks,
   and dispatch there — `Run workflow` needs write permission.)*

5. **Promote the real numbers:** download that artifact, copy its report + results over
   `bench/reports/` / `bench/results/` (replacing the synthetic sample), update the README
   headline with the real before/after figures, commit.

## D. Optional — remote Arm VM (bigger models, Performix)

Cheapest options (prices verified 2026-07-05, us-east-1): AWS `t4g.small` free-trial
(750 h/mo through 2026), `c8g.2xlarge` ≈ $0.319/hr for the headline box, or Oracle A1
Always-Free (~2 OCPUs/12 GB). On a fresh Ubuntu 24.04 arm64 instance:

```bash
git clone <your-repo> && cd <your-repo>
bash scripts/setup_arm_vm.sh          # builds llama.cpp (+KleidiAI), installs deps
. .venv/bin/activate                  # picks up firstflight + LLAMA_CPP_BIN
make bench && make report             # the whole story
firstflight experiment --name kleidiai   # with LLAMA_BASELINE_BIN / LLAMA_KLEIDIAI_BIN set
```

Or drive it from a laptop (WSL/Git Bash on Windows — needs `rsync`/`ssh`):
```bash
bash scripts/run_remote.sh --host ubuntu@<arm-ip> --key ~/.ssh/id_ed25519 --setup
```

**Performix:** install `apx` on the box (learn.arm.com/install-guides/performix), then
`firstflight profile`. Performix is free; the Linux arm64 package comes from
`artifacts.tools.arm.com/arm-performix/app/latest/linux/arm64/` (a CLI-only build exists —
right for headless VMs), ships the `apx` CLI, and requires accepting the license. Verify the
install with `apx version`. Full support needs an Arm64 Linux target on Amazon Linux 2023 or
Ubuntu 22.04/24.04 — the VM in this section (README's "Path B") qualifies; GitHub-hosted
runners are ephemeral, so Performix profiling belongs here rather than in CI. First run may need the `TODO(confirm)` items in
`docs/CONFIRM_ON_ARM.md` §1 checked against `apx` output.

**OS tuning (AWS's own Graviton recommendations, all captured as run evidence):**
- **Transparent hugepages:** `echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled`
  (or run `ENABLE_THP=1 bash scripts/setup_arm_vm.sh`). The harness records the active THP
  mode in every result JSON and the report footer.
- **Model residency:** add `--mlock` to `firstflight ttft` for steady demo TTFT (no post-idle
  page-fault stalls); raise `ulimit -l` if the lock fails.
- **Build targeting:** the VM script already builds with `-mcpu=native` (the official
  AWS/Arm recipe); cross-compilers use `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv9-a+i8mm+dotprod`.

## E. Submission checklist

**Accounts (both are listed entry steps in the Official Rules — neither provides compute):**
- [ ] Devpost account + **Join Hackathon** clicked (this is what makes you able to submit)
- [ ] Free **Arm Developer Program** account — <https://developer.arm.com/arm-developer-program>
      (Rules §4 "How To Enter" walks through it: name/company/email/job title → verify email →
      complete profile. ~2 minutes. It grants docs/labs/support, **not** cloud instances —
      there is no Arm-provided cloud environment or credits for this challenge.)

**Project:**
- [ ] Real Arm numbers in the README headline (from the CI artifact or the VM run)
- [ ] Real report committed in `bench/reports/` (replacing the synthetic, banner-labeled sample)
- [ ] `configs/instances.yaml` price confirmed for your actual instance/region
- [ ] Repo URL in `pyproject.toml`; repo public; MIT license visible in the About sidebar
- [ ] `arm-bench` workflow green and linked in the submission (judges can re-run it)
- [ ] Demo video < 3 min following `docs/DEMO_SCRIPT.md`
- [ ] The Devpost text description itself contains the Setup Instructions (build/run/validate)
      and embedded before/after results as images — the rules let judges "judge based solely on
      the text description, images, and video" (ready-to-paste section in `docs/DEVPOST.md`)
- [ ] Repo stays public, free of charge, and the `arm-bench` workflow stays runnable through
      the end of the Judging Period (**Aug 17 – Sep 4, 2026 4:00pm PT**) — don't archive,
      rename, or make it private after submitting
- [ ] Name the concrete Arm64 silicon + instance you benchmarked on in the write-up (e.g.
      "GitHub-hosted `ubuntu-24.04-arm` = Azure Cobalt 100 / Neoverse N2, 4 vCPU", or your
      Graviton instance type) — the rules ask entrants to document their own test environment
- [ ] Open the Devpost **"Enter a Submission"** form early and check its actual fields — it is
      login-gated, so nobody can verify from outside whether it asks for anything extra
- [ ] Remaining `docs/CONFIRM_ON_ARM.md` items resolved or consciously accepted

## Troubleshooting

| Symptom | Fix |
|---|---|
| ``SKIP no llama.cpp `llama-…` binary found`` (any of cli/bench/server/batched) | Run `firstflight setup-engine` (or set `LLAMA_CPP_BIN`) |
| `report rendering needs the [report] extra` | `pip install -e ".[report]"` |
| `setup-engine failed: release asset not found` | The pinned tag rotated — pass `--tag b####` from the llama.cpp Releases page |
| Model download slow/interrupted | Re-run; downloads are atomic (`.part` then rename), cached in `models/` |
| PowerShell won't run `activate` | `Set-ExecutionPolicy -Scope Process Bypass`, or call `.\.venv\Scripts\<tool>.exe` directly |
| Tests can't find modules | You're outside the venv — reactivate or use the venv's `python -m pytest` |
| Old llama.cpp build hangs on generation | Fixed in the harness (stdin closed); if driving llama-cli manually, redirect stdin from null |

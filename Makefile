# Arm FirstFlight: inference optimization lab.
# One-command story on an Arm box:  make bench && make report
#
# Targets invoke the CLI as `python -m firstflight` (same thing as the `firstflight` console
# script). Override with `make PYTHON=python3.11 ...` or `make FF=firstflight ...`.

PYTHON ?= python
FF     ?= $(PYTHON) -m firstflight

.DEFAULT_GOAL := help

.PHONY: help setup setup-engine smoke ttft throughput info download bench profile experiment report report-demo autotune test lint format clean

help:  ## Show this help
	@echo "Arm FirstFlight — targets:"
	@echo "  setup     Install the package (editable) with report+dev extras"
	@echo "  setup-engine  Download the prebuilt llama.cpp for THIS platform into ./engine"
	@echo "  smoke     Download tiny model + run llama.cpp once (skips cleanly off-Arm/no binary)"
	@echo "  info      Print environment + config summary"
	@echo "  download  Download the default smoke model only"
	@echo "  bench     Run the prefill/TTFT benchmark sweep"
	@echo "  ttft      MEASURED TTFT + prompt-cache demo (llama-server timings)"
	@echo "  throughput  Concurrency sweep: tok/s at 1/2/4/8 parallel requests"
	@echo "  profile   Profile with Arm Performix (apx)                (no-op off Arm)"
	@echo "  experiment  Before/after axis: quant/threads/pinning/KleidiAI + quality eval"
	@echo "  report    Render the before/after markdown + HTML report from bench/results"
	@echo "  report-demo  Render a synthetic DEMO report (no Arm box needed)"
	@echo "  autotune  Agent-in-the-loop optimizer (stretch; needs --enable)"
	@echo "  test      Run unit tests (pytest)"
	@echo "  lint      Lint with ruff"
	@echo "  format    Auto-format with ruff"
	@echo "  clean     Remove caches and local scratch"

setup:  ## Editable install with report + dev extras
	$(PYTHON) -m pip install -e ".[report,dev]"

setup-engine:  ## Download the prebuilt llama.cpp for this platform (no compiler needed)
	$(FF) setup-engine

smoke:  ## End-to-end smoke test (any machine)
	$(FF) smoke

info:  ## Environment + config summary
	$(FF) info

download:  ## Download the default smoke model only
	$(FF) download

bench:  ## Prefill/TTFT benchmark sweep
	$(FF) bench

ttft:  ## Measured TTFT + prompt-cache demo (llama-server timings)
	$(FF) ttft

throughput:  ## Concurrency sweep via llama-batched-bench
	$(FF) throughput

profile:  ## Arm Performix profile
	$(FF) profile

experiment:  ## Before/after optimization experiment + quality eval
	$(FF) experiment

report:  ## Render the before/after report from bench/results
	$(FF) report

report-demo:  ## Render a synthetic DEMO report (no Arm box needed)
	$(FF) report --demo

autotune:  ## Agentic autotuner (running the target IS the opt-in)
	$(FF) autotune --enable

test:  ## Unit tests
	$(PYTHON) -m pytest

lint:  ## Lint with ruff (same two checks CI runs)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:  ## Auto-format with ruff
	$(PYTHON) -m ruff format .

clean:  ## Remove caches and local scratch
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+['.pytest_cache','.ruff_cache','build','dist']]"

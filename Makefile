# Arm FirstFlight — Inference Optimization Lab
# One-command story on an Arm box:  make bench && make report
#
# Targets call the `firstflight` console script. If it is not on PATH, they fall back to
# `python -m firstflight`. Override the interpreter with `make PYTHON=python3.11 ...`.

PYTHON ?= python
FF     ?= $(PYTHON) -m firstflight

.DEFAULT_GOAL := help

.PHONY: help setup setup-engine smoke info download bench test lint format clean

help:  ## Show this help
	@echo "Arm FirstFlight — targets:"
	@echo "  setup     Install the package (editable) with dev extras"
	@echo "  setup-engine  Download the prebuilt llama.cpp for THIS platform into ./engine"
	@echo "  smoke     Download tiny model + run llama.cpp once (skips cleanly off-Arm/no binary)"
	@echo "  info      Print environment + config summary"
	@echo "  download  Download the default smoke model only"
	@echo "  bench     Run the prefill/TTFT benchmark sweep            (Phase 1)"
	@echo "  test      Run unit tests (pytest)"
	@echo "  lint      Lint with ruff"
	@echo "  format    Auto-format with ruff"
	@echo "  clean     Remove caches and local scratch"

setup:  ## Editable install with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

setup-engine:  ## Download the prebuilt llama.cpp for this platform (no compiler needed)
	$(FF) setup-engine

smoke:  ## End-to-end smoke test (any machine)
	$(FF) smoke

info:  ## Environment + config summary
	$(FF) info

download:  ## Download the default smoke model only
	$(FF) download

bench:  ## Prefill/TTFT benchmark sweep (Phase 1)
	$(FF) bench

test:  ## Unit tests
	$(PYTHON) -m pytest

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check .

format:  ## Auto-format with ruff
	$(PYTHON) -m ruff format .

clean:  ## Remove caches and local scratch
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in glob.glob('**/__pycache__',recursive=True)+['.pytest_cache','.ruff_cache','build','dist']]"

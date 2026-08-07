#!/usr/bin/env bash
# One-shot setup on a fresh Ubuntu 24.04 aarch64 instance (AWS Graviton / Oracle Ampere A1).
# Builds llama.cpp (+KleidiAI), installs Arm Performix, and installs firstflight.
#
# Usage:   bash scripts/setup_arm_vm.sh
# Tunables (env):  LLAMA_TAG=b9873  WITH_KLEIDIAI=1  INSTALL_PERFORMIX=1
set -euo pipefail

LLAMA_TAG="${LLAMA_TAG:-b9873}"            # verified 2026-07-05; tags float — bump deliberately
WITH_KLEIDIAI="${WITH_KLEIDIAI:-1}"
INSTALL_PERFORMIX="${INSTALL_PERFORMIX:-1}"
WORKDIR="${WORKDIR:-$HOME/firstflight-bench}"
# Resolve the repo this script lives in BEFORE any cd, so the install step below
# finds it no matter where the clone sits (FIRSTFLIGHT_REPO overrides).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> arch: $(uname -m)  (expecting aarch64)"
[ "$(uname -m)" = "aarch64" ] || echo "WARNING: not aarch64 — Arm optimizations won't apply."

echo "==> apt build deps"
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-venv python3-pip \
  libcurl4-openssl-dev rsync time

# Optional OS tuning (AWS Graviton perf runbook): transparent hugepages cut TLB pressure for
# the multi-GB weight working set. Set ENABLE_THP=1 to apply; the harness records the actual
# THP mode in every result either way.
if [ "${ENABLE_THP:-0}" = "1" ]; then
  echo "==> enabling transparent hugepages (madvise)"
  echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null || true
fi

mkdir -p "$WORKDIR" && cd "$WORKDIR"

echo "==> build llama.cpp @ ${LLAMA_TAG} (KleidiAI=${WITH_KLEIDIAI})"
if [ ! -d llama.cpp ]; then
  git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
# -mcpu=native is the official AWS/Arm build recipe (unlocks SVE/i8mm MMLA paths on
# Graviton3/4). Cross-compiling instead? Use -DGGML_NATIVE=OFF
# -DGGML_CPU_ARM_ARCH=armv9-a+i8mm+dotprod (GGML_SVE was removed upstream).
CMAKE_FLAGS=(-DCMAKE_BUILD_TYPE=Release
             -DCMAKE_C_FLAGS="-mcpu=native" -DCMAKE_CXX_FLAGS="-mcpu=native")
if [ "$WITH_KLEIDIAI" = "1" ]; then
  # KleidiAI accelerates Q4_0 / Q8_0 weights; flag verified 2026-06-26.
  CMAKE_FLAGS+=(-DGGML_CPU_KLEIDIAI=ON)
fi
cmake -B build "${CMAKE_FLAGS[@]}"
cmake --build build --config Release -j"$(nproc)"
echo "    built: $(ls build/bin/llama-bench build/bin/llama-cli 2>/dev/null || true)"
cd "$WORKDIR"

if [ "$INSTALL_PERFORMIX" = "1" ]; then
  echo "==> install Arm Performix (apx)"
  # TODO(confirm): follow the official install guide for the exact steps/package:
  #   https://learn.arm.com/install-guides/performix/
  #   CLI Reference (apx): https://developer.arm.com/documentation/111566
  # Left intentionally unimplemented rather than inventing commands. After install,
  # ensure `apx` is on PATH (or at /opt/Arm Performix/assets/apx/apx).
  echo "    SKIPPED — see learn.arm.com/install-guides/performix (marked TODO(confirm))."
fi

echo "==> python env + firstflight"
cd "${FIRSTFLIGHT_REPO:-$REPO_ROOT}"
if [ -f pyproject.toml ]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  pip install --upgrade pip
  pip install -e ".[report,hub]"
  echo "export LLAMA_CPP_BIN=$WORKDIR/llama.cpp/build" >> .venv/bin/activate
  echo "==> done. Activate: . .venv/bin/activate   then:  make bench && make report"
else
  echo "NOTE: run this from the repo, or set FIRSTFLIGHT_REPO to its path, to install firstflight."
  echo "      llama.cpp build is at: $WORKDIR/llama.cpp/build  (set LLAMA_CPP_BIN to it)"
fi

#!/usr/bin/env bash
# Rsync this repo to a remote Arm VM and run the full benchmark over SSH, then pull results
# back. For models bigger than CI allows.
#
# Usage:
#   bash scripts/run_remote.sh --host user@1.2.3.4 [--key ~/.ssh/id_ed25519] \
#        [--dir ~/firstflight] [--setup]
#
#   --setup   also run scripts/setup_arm_vm.sh on the remote first (one-time).
set -euo pipefail

HOST=""
KEY=""
REMOTE_DIR="~/arm-firstflight-infer-lab"
DO_SETUP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --key)  KEY="$2";  shift 2 ;;
    --dir)  REMOTE_DIR="$2"; shift 2 ;;
    --setup) DO_SETUP=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "$HOST" ] || { echo "ERROR: --host user@ip is required" >&2; exit 2; }

SSH_OPTS=()
RSYNC_SSH="ssh"
if [ -n "$KEY" ]; then
  SSH_OPTS=(-i "$KEY")
  RSYNC_SSH="ssh -i $KEY"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> rsync repo -> $HOST:$REMOTE_DIR"
# Exclude local heavy/scratch dirs; remote rebuilds its own venv + models.
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude 'models' \
  --exclude '__pycache__' --exclude '*.gguf' \
  -e "$RSYNC_SSH" \
  "$REPO_ROOT"/ "$HOST:$REMOTE_DIR"/

if [ "$DO_SETUP" = "1" ]; then
  echo "==> remote setup (build llama.cpp + KleidiAI, install deps)"
  ssh "${SSH_OPTS[@]}" "$HOST" "cd $REMOTE_DIR && FIRSTFLIGHT_REPO=$REMOTE_DIR bash scripts/setup_arm_vm.sh"
fi

echo "==> remote benchmark + report"
# shellcheck disable=SC2029
ssh "${SSH_OPTS[@]}" "$HOST" "cd $REMOTE_DIR && . .venv/bin/activate 2>/dev/null; make bench && make report"

echo "==> pull results + reports back"
mkdir -p "$REPO_ROOT/bench"
rsync -az -e "$RSYNC_SSH" "$HOST:$REMOTE_DIR/bench/results/" "$REPO_ROOT/bench/results/" || true
rsync -az -e "$RSYNC_SSH" "$HOST:$REMOTE_DIR/bench/reports/" "$REPO_ROOT/bench/reports/" || true
echo "==> done. See bench/reports/ for the generated report."

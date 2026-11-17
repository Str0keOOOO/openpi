#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/scripts/env.sh"
mode="${1:-help}"
if (( $# )); then shift; fi
cd "$repo"
case "$mode" in
  train)
    mkdir -p "$WANDB_DIR"
    exec "$ROBOCASA365_ROOT/envs/policy/bin/python" scripts/train.py "$@" ;;
  serve) exec "$ROBOCASA365_ROOT/envs/policy/bin/python" examples/robocasa_unified/seeded_server.py "$@" ;;
  evaluate) exec "$ROBOCASA365_ROOT/envs/sim/bin/python" scripts/evaluate.py "$@" ;;
  prepare) exec "$ROBOCASA365_ROOT/envs/policy/bin/python" scripts/prepare_data.py "$@" ;;
  download) exec "$ROBOCASA365_ROOT/envs/policy/bin/python" scripts/download_checkpoint.py "$@" ;;
  check) exec "$ROBOCASA365_ROOT/envs/policy/bin/python" scripts/check_workspace.py "$@" ;;
  check-sim) exec "$ROBOCASA365_ROOT/envs/sim/bin/python" scripts/check_sim.py "$@" ;;
  *) echo 'Usage: scripts/run.sh {prepare|train|serve|evaluate|download|check|check-sim} [arguments]'; exit 0 ;;
esac

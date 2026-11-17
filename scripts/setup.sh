#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/scripts/env.sh"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROBOCASA365_ROOT/envs/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ROBOCASA365_ROOT/envs/.python}"
cd "$repo"
mode="${1:-all}"
if [[ "$mode" != policy && "$mode" != sim && "$mode" != all ]]; then
  echo 'Usage: scripts/setup.sh [policy|sim|all]' >&2; exit 2
fi
for name in robocasa robosuite; do
  if [[ ! -f "third_party/$name/setup.py" ]]; then
    git submodule update --init "third_party/$name"
  fi
done
if [[ "$mode" == policy || "$mode" == all ]]; then
  UV_PROJECT_ENVIRONMENT="$ROBOCASA365_ROOT/envs/policy" uv sync --frozen --python 3.11 --group dev
fi
if [[ "$mode" == sim || "$mode" == all ]]; then
  UV_PROJECT_ENVIRONMENT="$ROBOCASA365_ROOT/envs/sim" uv sync --frozen --python 3.11 --only-group sim
fi

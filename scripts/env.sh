#!/usr/bin/env bash
# Source this file before invoking a project script directly.
_openpi_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export ROBOCASA365_ROOT="${ROBOCASA365_ROOT:-$(dirname -- "${_openpi_repo}")}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ROBOCASA365_ROOT}/model}"
export PYTHONPATH="${_openpi_repo}/src:${_openpi_repo}:${_openpi_repo}/packages/openpi-client/src:${_openpi_repo}/third_party/robocasa:${_openpi_repo}/third_party/robosuite${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DIR="${WANDB_DIR:-${ROBOCASA365_ROOT}/output}"
export WANDB_MODE="${WANDB_MODE:-offline}"
unset _openpi_repo

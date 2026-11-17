#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../../.." && pwd)"
PY="$ROOT/envs/benchmark/bin/python"
: "${GPU:?Set GPU to a selected physical GPU index}"
: "${OUTPUT:?Set OUTPUT to a new directory under /data4/xuchenfei/robocasa365/output}"
PORT="${PORT:-18281}"
CONFIG="${CONFIG:-$HERE/benchmark.json}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$GPU"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.28
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mode="${1:-plan}"
if [ "$#" -gt 0 ]; then shift; fi
mkdir -p "$OUTPUT/logs"
case "$mode" in
  serve)
    # A new process and a dedicated port; no existing service is stopped or reused.
    readarray -t model < <("$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); print(c["policy_config"]); print(c["checkpoint"])' "$CONFIG")
    cd "$ROOT/openpi"
    exec "$PY" -u scripts/serve_policy.py --port="$PORT" policy:checkpoint \
      --policy.config="${model[0]}" --policy.dir="${model[1]}" \
      >> "$OUTPUT/logs/policy.log" 2>&1
    ;;
  plan|prepare|sample|evaluate)
    exec "$PY" -u "$HERE/main.py" --mode "$mode" --config "$CONFIG" --output "$OUTPUT" --port "$PORT" "$@" \
      >> "$OUTPUT/logs/$mode.log" 2>&1
    ;;
  *) echo "Expected plan, prepare, sample, serve or evaluate" >&2; exit 2 ;;
esac

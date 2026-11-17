"""Run prepare -> sample -> independent server -> evaluate; owns only its children."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", required=True)
    p.add_argument("--port", type=int, default=18281)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = Path(__file__).resolve().parent
    root = source.parents[3]
    python = root / "envs/benchmark/bin/python"
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    snapshot = out / "code"
    snapshot.mkdir(exist_ok=True)
    for name in ("main.py", "benchmark.json", "launch.py", "README.md"):
        if not (snapshot / name).exists():
            shutil.copy2(source / name, snapshot / name)
    config = json.loads((snapshot / "benchmark.json").read_text())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               MUJOCO_EGL_DEVICE_ID=args.gpu,
               __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
               XLA_PYTHON_CLIENT_MEM_FRACTION="0.28", XLA_PYTHON_CLIENT_PREALLOCATE="true",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               TMPDIR=str(out / "tmp"), OPENPI_DATA_HOME=os.environ.get("OPENPI_DATA_HOME", str(root / "model")),
               JAX_COMPILATION_CACHE_DIR=str(out / "cache/jax"))
    for key in ("TMPDIR", "OPENPI_DATA_HOME", "JAX_COMPILATION_CACHE_DIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    tokenizer = root / "model/big_vision/paligemma_tokenizer.model"
    if tokenizer.exists():
        dest = Path(env["OPENPI_DATA_HOME"]) / "big_vision/paligemma_tokenizer.model"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tokenizer, dest)
    started = time.perf_counter()
    manifest = dict(started_at=datetime.datetime.now().astimezone().isoformat(),
                    gpu=args.gpu, port=args.port, output=str(out), status="preflight")

    def status(phase, **fields):
        manifest.update(status=phase, wall_seconds=time.perf_counter() - started, **fields)
        temp = out / "run_manifest.json.tmp"
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(out / "run_manifest.json")
        print(phase, fields, flush=True)

    server = None
    try:
        free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                   "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        if free < 17500:
            raise RuntimeError(f"GPU {args.gpu} has only {free} MiB free; no processes changed")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        manifest["gpu_free_before_mib"] = free
        command = [str(python), "-u", str(snapshot / "main.py"), "--config", str(snapshot / "benchmark.json"),
                   "--output", str(out), "--port", str(args.port)]

        def run_phase(mode, extra=()):
            status(mode, current_task=extra[-1] if extra else None)
            with (out / "logs" / f"{mode}.log").open("a") as log:
                child = subprocess.Popen(command + ["--mode", mode, *extra], env=env, stdout=log, stderr=log)
                manifest["worker_pid"] = child.pid
                while child.poll() is None:
                    status(mode, worker_pid=child.pid)
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f"{mode} failed with exit code {child.returncode}; see logs/{mode}.log")

        for task in config["tasks"]:
            if not (out / "anchors" / task / "anchor.json").exists():
                run_phase("prepare", ["--task", task])
        run_phase("sample")
        status("loading_policy")
        with (out / "logs/policy.log").open("a") as log:
            server = subprocess.Popen([str(python), "-u", "scripts/serve_policy.py", f"--port={args.port}",
                        "policy:checkpoint", f'--policy.config={config["policy_config"]}',
                        f'--policy.dir={config["checkpoint"]}'], cwd=root / "third_party/openpi", env=env, stdout=log, stderr=log)
        deadline = time.monotonic() + 1800
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            if server.poll() is not None:
                raise RuntimeError("Independent policy service exited; see logs/policy.log")
            try:
                with opener.open(f"http://127.0.0.1:{args.port}/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("Policy startup timed out")
            status("loading_policy", server_pid=server.pid)
            time.sleep(5)
        run_phase("evaluate")
        status("complete", finished_at=datetime.datetime.now().astimezone().isoformat())
    except BaseException as exc:
        status("failed", error=repr(exc))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()

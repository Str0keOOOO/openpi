#!/usr/bin/env python3
"""Download inference-only RoboCasa365 PI0.5 checkpoint files.

The public 75k checkpoint's train_state is intentionally omitted. Downloads are
resumable and verified against the sizes returned by the Hugging Face tree API.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote

from openpi.shared.paths import workspace_root

ROOT = workspace_root()
DEST = ROOT / "model/pi05_pretrain_human300/multitask_learning/75000"
LOG = ROOT / "output/setup_logs/pi05_75000_params_download.log"
REPO = "robocasa/robocasa365_checkpoints"
REVISION = "c484448aba1a9b60a04c9b0ca117241518ea69f3"
REMOTE_PREFIX = "pi05_pretrain_human300/multitask_learning/75000"
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")

PARAM_FILES = {
    "params/_METADATA": 23544,
    "params/_sharding": 12797,
    "params/array_metadatas/process_0": 9162,
    "params/d/35c0b41e13ddb0a578456230f51a3416": 2138,
    "params/manifest.ocdbt": 117,
    "params/ocdbt.process_0/d/16a296efc9531f413b90c15199e43385": 538931200,
    "params/ocdbt.process_0/d/28b537f74e118c2c1c44f72e8b0e00b7": 2240151552,
    "params/ocdbt.process_0/d/471d6e9b709ec1a579f83c8b4914e1ed": 30384128,
    "params/ocdbt.process_0/d/5cf1127d4e3fcd6e80a66ebba54dc32d": 2799407104,
    "params/ocdbt.process_0/d/6292e633c11b8f6931cac6ab824f3279": 35061760,
    "params/ocdbt.process_0/d/65f82551b4d64f31f6bb1cdcf17945e1": 70041600,
    "params/ocdbt.process_0/d/7f76f4974bbb6946ff46b5dbcd4d44f8": 2529587200,
    "params/ocdbt.process_0/d/b627c21812ded6c524507187a8526a3e": 1956962304,
    "params/ocdbt.process_0/d/b808ab91bab1b5805b381e09b7fc0a85": 5367,
    "params/ocdbt.process_0/d/d71893b153190783845eb3aabec57214": 1098,
    "params/ocdbt.process_0/d/d76149c5c280cb9da59fd2f948d9cdda": 213,
    "params/ocdbt.process_0/d/fb6f9dae6d90f1b488a3d4ebf79278ed": 2240278528,
    "params/ocdbt.process_0/manifest.ocdbt": 486,
    "assets/norm_stats.json": 3208,
    "_CHECKPOINT_METADATA": 426,
}


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def fetch(item: tuple[str, int], destination_root: Path, endpoint: str) -> tuple[str, bool, str]:
    relative, expected = item
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected:
        return relative, True, "already complete"

    partial = destination.with_name(destination.name + ".part")
    url = f"{endpoint}/{REPO}/resolve/{REVISION}/{quote(REMOTE_PREFIX + '/' + relative, safe='/')}"
    command = [
        "curl",
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "8",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "0",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return relative, False, result.stderr.strip()[-800:]
    actual = partial.stat().st_size if partial.exists() else -1
    if actual != expected:
        return relative, False, f"size mismatch: expected {expected}, got {actual}"
    partial.replace(destination)
    return relative, True, f"{actual} bytes"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEST)
    parser.add_argument("--endpoint", default=ENDPOINT)
    args = parser.parse_args()
    destination = args.output.resolve()
    endpoint = args.endpoint.rstrip("/")
    destination.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    total = sum(PARAM_FILES.values())
    log(f"START endpoint={endpoint} revision={REVISION} files={len(PARAM_FILES)} bytes={total}")
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(fetch, item, destination, endpoint): item[0] for item in PARAM_FILES.items()}
        for future in concurrent.futures.as_completed(futures):
            relative = futures[future]
            try:
                name, ok, detail = future.result()
            except Exception as error:  # Keep other files progressing.
                name, ok, detail = relative, False, repr(error)
            log(f"{'OK' if ok else 'ERROR'} {name}: {detail}")
            if not ok:
                failures.append((name, detail))
    if failures:
        log(f"INCOMPLETE {len(failures)} files; rerun this script to resume")
        return 1
    log(f"COMPLETE checkpoint={destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

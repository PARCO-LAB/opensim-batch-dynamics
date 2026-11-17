#!/usr/bin/env python3
"""Run manifest groups locally with a fixed, low CPU concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Run output folder for logs/status (defaults to phase_a)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    manifest = args.manifest.resolve()
    study_root = args.study_root.resolve()
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    groups = list(dict.fromkeys((row["variant"], row["subject"]) for row in rows))
    if not groups:
        raise RuntimeError(f"No groups in {manifest}")

    run_dir = (args.run_dir or (study_root / "phase_a")).resolve()
    logs = run_dir / "local_logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "local_group_runs.csv"
    script = Path(__file__).with_name("run_id_group.py").resolve()
    env = os.environ.copy()
    opensim_bin = "/home/emartini/miniconda3/envs/opensim-torque/bin"
    env["PATH"] = opensim_bin + os.pathsep + env.get("PATH", "")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    try:
        os.nice(10)
    except OSError:
        pass

    def run_one(index: int) -> tuple[int, str, str, int, float, str]:
        variant, subject = groups[index]
        log_path = logs / f"{index:02d}_{variant}_{subject}.log"
        started = time.monotonic()
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, str(script), "--manifest", str(manifest), "--group-index", str(index)],
                cwd=script.parents[3],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return index, variant, subject, result.returncode, time.monotonic() - started, str(log_path)

    fields = ("group_index", "variant", "subject", "return_code", "elapsed_seconds", "log")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", newline="", encoding="utf-8") as status_stream:
        writer = csv.DictWriter(status_stream, fieldnames=fields)
        if status_stream.tell() == 0:
            writer.writeheader()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, i) for i in range(len(groups))]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                index, variant, subject, code, elapsed, log_path = future.result()
                writer.writerow({
                    "group_index": index,
                    "variant": variant,
                    "subject": subject,
                    "return_code": code,
                    "elapsed_seconds": f"{elapsed:.1f}",
                    "log": log_path,
                })
                status_stream.flush()
                print(f"[{completed}/{len(groups)}] {variant}/{subject} rc={code} elapsed={elapsed/3600:.2f}h", flush=True)

    failures = [row for row in csv.DictReader(status_path.open(newline="")) if int(row["return_code"]) != 0]
    print(f"groups={len(groups)} workers={args.workers} failures_in_status_file={len(failures)} status={status_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

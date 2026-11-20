#!/usr/bin/env python3
"""Watch an AMASS run and submit one 32 GB retry for each newly seen OOM."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_oom_tasks(results_dir: Path) -> list[int]:
    tasks: set[int] = set()
    for path in results_dir.glob("task_*.json"):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("status") == "failed" and result.get("return_code") == -9:
                tasks.add(int(result["task_index"]))
        except (OSError, ValueError, KeyError, TypeError):
            # A worker may be writing this file; the next polling pass will retry it.
            continue
    return sorted(tasks)


def _contiguous_groups(pairs: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    for pair in sorted(pairs):
        if not groups or pair[0] != groups[-1][-1][0] + 1:
            groups.append([pair])
        else:
            groups[-1].append(pair)
    return groups


def _run_once(args: argparse.Namespace, slurm_dir: Path, state: dict[str, object]) -> dict[str, object]:
    map_path = slurm_dir / "oom_retry_indices.txt"
    map_rows = [int(line) for line in map_path.read_text().splitlines() if line.strip()]
    index_by_task = {task: index for index, task in enumerate(map_rows)}

    submitted = {int(task) for task in state.get("submitted_task_indices", [])}
    if not state.get("initialized"):
        # Existing map entries were already submitted manually before this watcher.
        submitted.update(map_rows)
        state["initialized"] = True

    oom_tasks = _read_oom_tasks(slurm_dir / "results")
    new_tasks = [task for task in oom_tasks if task not in index_by_task]
    if new_tasks and not args.dry_run:
        start = len(map_rows)
        with map_path.open("a", encoding="utf-8") as mapping:
            for task in new_tasks:
                mapping.write(f"{task}\n")
            mapping.flush()
        for offset, task in enumerate(new_tasks):
            index_by_task[task] = start + offset
        map_rows.extend(new_tasks)

    retry_tasks = [task for task in oom_tasks if task not in submitted]
    pairs: list[tuple[int, int]] = []
    next_index = len(map_rows)
    for task in retry_tasks:
        index = index_by_task.get(task)
        if index is None:
            index = next_index
            next_index += 1
        pairs.append((index, task))

    submitted_now: list[int] = []
    for group in _contiguous_groups(pairs):
        array_start = group[0][0]
        array_end = group[-1][0]
        tasks = [task for _, task in group]
        comment = f"amass_oom_auto_{array_start}_{array_end}"
        if args.dry_run:
            print(
                f"DRY_RUN tasks={tasks} array={array_start}-{array_end} "
                f"memory={args.memory} partition={args.partition}",
                flush=True,
            )
            continue

        command = [
            "sbatch",
            "--parsable",
            f"--partition={args.partition}",
            f"--mem={args.memory}",
            f"--array={array_start}-{array_end}%{min(len(group), args.array_throttle)}",
            f"--comment={comment}",
            str(args.retry_script),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode == 0 and completed.stdout.strip():
            submitted.update(tasks)
            submitted_now.extend(tasks)
            print(
                f"SUBMITTED tasks={tasks} job={completed.stdout.strip()} "
                f"array={array_start}-{array_end}",
                flush=True,
            )
        else:
            print(
                f"SUBMIT_FAILED tasks={tasks} code={completed.returncode} "
                f"detail={(completed.stderr or completed.stdout).strip()}",
                flush=True,
            )

    state["submitted_task_indices"] = sorted(submitted)
    state["last_scan_utc"] = datetime.now(timezone.utc).isoformat()
    if not args.dry_run:
        _atomic_json(slurm_dir / "oom_retry_watcher_state.json", state)
    print(
        f"SCAN oom={len(oom_tasks)} mapped={len(index_by_task)} "
        f"unmapped={len(new_tasks)} submitted_now={len(submitted_now)}",
        flush=True,
    )
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--retry-script", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--partition", default="gpuRTX")
    parser.add_argument("--memory", default="32G")
    parser.add_argument("--array-throttle", type=int, default=4)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.interval_seconds <= 0 or args.array_throttle <= 0:
        parser.error("interval and array throttle must be positive")
    root = args.output_root.resolve()
    slurm_dir = root / "slurm"
    if not (slurm_dir / "results").is_dir():
        parser.error(f"No results directory under {slurm_dir}")
    if not (slurm_dir / "oom_retry_indices.txt").is_file():
        parser.error(f"Missing retry map under {slurm_dir}")
    if not args.retry_script.is_file():
        parser.error(f"Missing retry script: {args.retry_script}")

    lock_path = slurm_dir / "oom_retry_watcher.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another OOM watcher already holds the lock.", file=sys.stderr)
            return 2

        state_path = slurm_dir / "oom_retry_watcher_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        except (OSError, ValueError):
            state = {}

        while True:
            state = _run_once(args, slurm_dir, state)
            if args.once:
                return 0
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

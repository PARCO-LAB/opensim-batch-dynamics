#!/usr/bin/env python3
"""Feed missing AMASS array tasks into Slurm as controller capacity opens."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _outstanding_job_count() -> int:
    # MaxJobCount is controller-wide, so count all users rather than only this user.
    result = subprocess.run(
        ["squeue", "-h", "-r", "-t", "PD,R", "-o", "%i"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"squeue failed ({result.returncode}): {result.stderr.strip()}")
    return sum(bool(line.strip()) for line in result.stdout.splitlines())


def _chunk_index(script: str) -> int:
    match = re.search(r"run_batch_(\d+)", Path(script).name)
    return int(match.group(1)) if match else 0


def _task_count_for_chunk(script: str, total_tasks: int, chunk_size: int) -> int:
    return max(0, min(chunk_size, total_tasks - _chunk_index(script) * chunk_size))


def _covered_tasks(
    script: str, count: int, results: list[object]
) -> list[bool]:
    covered = [False] * count
    resolved_script = str(Path(script).resolve())
    for item in results:
        if not isinstance(item, dict):
            continue
        if str(Path(str(item.get("script_path", ""))).resolve()) != resolved_script:
            continue
        try:
            returncode = int(item.get("returncode", 1))
        except (TypeError, ValueError):
            continue
        if returncode != 0:
            continue

        array_start = item.get("array_start")
        array_end = item.get("array_end")
        if array_start is None or array_end is None:
            return [True] * count
        start = max(0, int(array_start))
        end = min(count - 1, int(array_end))
        for index in range(start, end + 1):
            covered[index] = True
    return covered


def _first_missing_range(covered: list[bool]) -> tuple[int, int] | None:
    try:
        start = covered.index(False)
    except ValueError:
        return None
    end = start
    while end + 1 < len(covered) and not covered[end + 1]:
        end += 1
    return start, end


def _submit(
    script: str,
    chunk_index: int,
    start: int,
    end: int,
    throttle: int,
    partition: str,
) -> tuple[int, str, str, str | None]:
    comment = f"amass_chunk_{chunk_index}_{start}_{end}"
    result = subprocess.run(
        [
            "sbatch",
            "--parsable",
            f"--comment={comment}",
            f"--partition={partition}",
            f"--array={start}-{end}%{throttle}",
            script,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    job_id_match = re.search(r"Submitted batch job\s+(\d+)", result.stdout)
    job_id = result.stdout.strip().split(";", 1)[0] if result.returncode == 0 else None
    if job_id_match:
        job_id = job_id_match.group(1)
    return result.returncode, result.stdout.strip(), result.stderr.strip(), job_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--max-job-count", type=int, default=10000)
    parser.add_argument("--safety-slots", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--min-array-size", type=int, default=32)
    parser.add_argument("--array-throttle", type=int, default=72)
    parser.add_argument("--partition", default="gpuRTX")
    args = parser.parse_args(argv)

    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    while "sbatch_submission_results" not in plan:
        print(
            "Initial array-submission pass is still running; waiting for its result list.",
            flush=True,
        )
        time.sleep(args.poll_seconds)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

    scripts = [str(Path(path).resolve()) for path in plan.get("sbatch_scripts", [])]
    total_tasks = int(plan.get("scheduled_tasks", 0))
    results = plan.get("sbatch_submission_results", [])
    if not scripts or total_tasks <= 0 or not isinstance(results, list):
        raise SystemExit(f"Invalid submission plan: {plan_path}")

    print(
        f"Capacity feeder: {total_tasks} tasks across {len(scripts)} arrays; "
        f"partition={args.partition}, throttle={args.array_throttle}, "
        f"controller cap={args.max_job_count}, safety={args.safety_slots}.",
        flush=True,
    )

    for script in scripts:
        chunk_index = _chunk_index(script)
        count = _task_count_for_chunk(script, total_tasks, args.chunk_size)
        if count <= 0:
            continue

        while True:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            result_items = plan.get("sbatch_submission_results", [])
            if not isinstance(result_items, list):
                raise SystemExit("submit_plan.json has invalid sbatch_submission_results")
            covered = _covered_tasks(script, count, result_items)
            missing = _first_missing_range(covered)
            if missing is None:
                break

            start, end = missing
            outstanding = _outstanding_job_count()
            available = args.max_job_count - outstanding - args.safety_slots
            minimum_batch = min(args.min_array_size, end - start + 1)
            if available < minimum_batch:
                print(
                    f"Controller queue has {outstanding} pending/running tasks; "
                    f"{available} slots available, need {minimum_batch} before "
                    f"chunk {chunk_index}.",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                continue

            submit_end = min(end, start + available - 1, start + args.chunk_size - 1)
            code, stdout, stderr, job_id = _submit(
                script,
                chunk_index,
                start,
                submit_end,
                args.array_throttle,
                args.partition,
            )
            if code != 0 or not job_id:
                print(
                    f"sbatch failed for chunk {chunk_index} tasks {start}-{submit_end} "
                    f"({code}): {stderr or stdout}; retrying in {args.poll_seconds:g}s.",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                continue

            record: dict[str, object] = {
                "script_path": script,
                "sbatch_command": [
                    "sbatch",
                    "--parsable",
                    f"--partition={args.partition}",
                    f"--array={start}-{submit_end}%{args.array_throttle}",
                    script,
                ],
                "task_index_offset": chunk_index * args.chunk_size,
                "array_start": start,
                "array_end": submit_end,
                "retry_count": 0,
                "returncode": 0,
                "stdout": stdout,
                "stderr": stderr,
                "job_id": job_id,
                "submitted_by_capacity_feeder": True,
            }
            result_items.append(record)
            plan["sbatch_submission_results"] = result_items
            _atomic_write_json(plan_path, plan)
            print(
                f"Submitted chunk {chunk_index} tasks {start}-{submit_end} "
                f"as job {job_id}; queue={outstanding}, arrays={len(scripts)}.",
                flush=True,
            )
            time.sleep(2.0)

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    all_covered = True
    result_items = plan.get("sbatch_submission_results", [])
    for script in scripts:
        count = _task_count_for_chunk(script, total_tasks, args.chunk_size)
        if _first_missing_range(_covered_tasks(script, count, result_items)) is not None:
            all_covered = False
            break
    plan["submitted"] = all_covered
    _atomic_write_json(plan_path, plan)
    print(
        "All scheduled array tasks have been submitted."
        if all_covered
        else "Feeder stopped with array tasks still uncovered.",
        flush=True,
    )
    return 0 if all_covered else 1


if __name__ == "__main__":
    sys.exit(main())

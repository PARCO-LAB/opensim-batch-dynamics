#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from batch_common import (
    BatchTask,
    DYNAMICS_PIPELINE_VERSION,
    build_trial_name,
    discover_amass_input_files,
    excluded_trial_reason,
    is_nonempty_file,
    is_quality_pass,
    is_valid_groundlink_csv,
    resolve_pipeline_script,
    resolve_submit_path,
    write_json,
)


@dataclass(frozen=True)
class BatchTaskResult:
    task: BatchTask
    status: str
    return_code: int
    duration_s: float
    error: str | None


@contextlib.contextmanager
def _batch_lock(output_root: Path) -> Iterator[None]:
    """Prevent two batches from writing the same output tree concurrently."""
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".batch.lock").open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another batch is already using {output_root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full AMASS->OpenSim->CSV pipeline in parallel over all .npz/.npy files "
            "found inside an input folder."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root folder containing AMASS .npz or Motion-X .npy files (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/bsm_batch",
        help="Output root for generated CSVs and logs (default: outputs/bsm_batch).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of parallel workers (default: max(1, cpu_count/2)).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of input files to process.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used for child pipeline runs (default: current interpreter).",
    )
    parser.add_argument(
        "--pipeline-script",
        default=None,
        help=(
            "Path to run_amass_to_bsm_csv.py (default: scripts/run_amass_to_bsm_csv.py "
            "resolved from repository root)."
        ),
    )
    parser.add_argument(
        "--skip-existing-csv",
        dest="skip_existing",
        action="store_true",
        help="Skip files whose final CSV already exists and is non-empty (default: enabled).",
    )
    parser.add_argument(
        "--no-skip-existing-csv",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing CSVs.",
    )
    parser.add_argument(
        "--rerun-walking",
        action="store_true",
        help="Recompute walk trials even when --skip-existing-csv is enabled.",
    )
    parser.add_argument(
        "--require-intermediate-artifacts",
        action="store_true",
        help=(
            "When skipping existing CSVs, require a retained OpenSim results directory; "
            "otherwise rerun the trial."
        ),
    )
    # Backward-compatible aliases.
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument(
        "--cleanup-intermediate",
        dest="cleanup_intermediate",
        action="store_true",
        help="Pass --cleanup-intermediate to the single-file pipeline (default: enabled).",
    )
    parser.add_argument(
        "--no-cleanup-intermediate",
        dest="cleanup_intermediate",
        action="store_false",
        help="Keep intermediate files for each run.",
    )
    parser.set_defaults(cleanup_intermediate=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned jobs without running them.",
    )

    # Forwarded options for the single-file pipeline.
    parser.add_argument("--smplx-model-dir", default="model/smpl")
    parser.add_argument("--bsm-model", default="model/bsm/bsm.osim")
    parser.add_argument("--addbio-root", default=None)
    parser.add_argument(
        "--engine-work-root",
        default="/tmp/groundlink-engine-work",
        help="Local scratch root for AddBiomechanics/OpenSim work (default: /tmp/groundlink-engine-work).",
    )
    parser.add_argument(
        "--engine-timeout-s",
        type=float,
        default=7200.0,
        help="Maximum AddBiomechanics engine runtime in seconds (default: 7200; 0 disables).",
    )
    parser.add_argument("--sex", choices=["neutral", "male", "female"], default=None)
    parser.add_argument("--subject-mass-kg", type=float, default=None)
    parser.add_argument("--subject-height-m", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-inverse-dynamics", action="store_true")
    parser.add_argument(
        "--id-filter-mode",
        default="auto",
        choices=["auto", "walking", "dynamic", "none"],
    )
    parser.add_argument("--id-cutoff-hz", type=float, default=None)
    parser.add_argument(
        "--id-grf-mode",
        default="estimated",
        choices=["estimated", "none"],
    )
    parser.add_argument(
        "--id-contact-bodies",
        default="calcn_l,toes_l,calcn_r,toes_r",
        help="Contact bodies for GRF estimation (default: calcn/toes on both sides).",
    )
    parser.add_argument("--id-friction-coeff", type=float, default=0.8)
    parser.add_argument("--id-contact-height-threshold-m", type=float, default=0.04)
    parser.add_argument("--id-contact-speed-threshold-mps", type=float, default=0.5)
    parser.add_argument(
        "--walking-contact-height-threshold-m",
        type=float,
        default=None,
        help="Override contact height threshold for walk trials only.",
    )
    parser.add_argument(
        "--walking-contact-speed-threshold-mps",
        type=float,
        default=None,
        help="Override contact speed threshold for walk trials only.",
    )
    return parser.parse_args()

def _build_tasks(args: argparse.Namespace, input_root: Path, output_root: Path) -> list[BatchTask]:
    files = discover_amass_input_files(input_root)
    if args.limit is not None:
        files = files[: max(0, args.limit)]

    tasks: list[BatchTask] = []
    for input_path in files:
        relative_path = input_path.relative_to(input_root)
        output_csv_path = (output_root / relative_path).with_suffix(".csv")
        log_path = (output_root / "logs" / relative_path).with_suffix(".log")
        trial_name = build_trial_name(relative_path)
        tasks.append(
            BatchTask(
                input_path=input_path,
                relative_path=relative_path,
                output_csv_path=output_csv_path,
                log_path=log_path,
                trial_name=trial_name,
            )
        )
    return tasks


def _build_single_run_cmd(
    args: argparse.Namespace,
    pipeline_script: Path,
    task: BatchTask,
    output_root: Path,
    child_trial_name: str | None = None,
) -> list[str]:
    trial_name = child_trial_name or task.trial_name
    cmd = [
        args.python_exe,
        str(pipeline_script),
        "--input",
        str(task.input_path),
        "--trial",
        trial_name,
        "--output-dir",
        str(output_root),
        "--final-csv-path",
        str(task.output_csv_path),
        "--smplx-model-dir",
        args.smplx_model_dir,
        "--bsm-model",
        args.bsm_model,
        "--device",
        args.device,
        "--id-filter-mode",
        args.id_filter_mode,
        "--id-grf-mode",
        args.id_grf_mode,
        "--id-contact-bodies",
        args.id_contact_bodies,
        "--id-friction-coeff",
        str(args.id_friction_coeff),
        "--id-contact-height-threshold-m",
        str(args.id_contact_height_threshold_m),
        "--id-contact-speed-threshold-mps",
        str(args.id_contact_speed_threshold_mps),
    ]
    if "_walk_" in task.relative_path.stem.lower():
        if args.walking_contact_height_threshold_m is not None:
            cmd[-4:-2] = [
                "--id-contact-height-threshold-m",
                str(args.walking_contact_height_threshold_m),
            ]
        if args.walking_contact_speed_threshold_mps is not None:
            cmd[-2:] = [
                "--id-contact-speed-threshold-mps",
                str(args.walking_contact_speed_threshold_mps),
            ]
    if args.addbio_root:
        cmd.extend(["--addbio-root", args.addbio_root])
    engine_work_root = getattr(args, "engine_work_root", None)
    if engine_work_root:
        cmd.extend(["--engine-work-root", engine_work_root])
    engine_timeout_s = getattr(args, "engine_timeout_s", 7200.0)
    cmd.extend(["--engine-timeout-s", str(engine_timeout_s)])
    if args.sex:
        cmd.extend(["--sex", args.sex])
    if args.subject_mass_kg is not None:
        cmd.extend(["--subject-mass-kg", str(args.subject_mass_kg)])
    if args.subject_height_m is not None:
        cmd.extend(["--subject-height-m", str(args.subject_height_m)])
    if args.id_cutoff_hz is not None:
        cmd.extend(["--id-cutoff-hz", str(args.id_cutoff_hz)])
    if args.skip_inverse_dynamics:
        cmd.append("--skip-inverse-dynamics")
    if args.cleanup_intermediate:
        cmd.append("--cleanup-intermediate")
    return cmd


def _run_single_task(
    args: argparse.Namespace,
    pipeline_script: Path,
    output_root: Path,
    task: BatchTask,
) -> BatchTaskResult:
    exclusion = excluded_trial_reason(task.relative_path)
    if exclusion:
        # Do not leave an old successful sidecar next to a trial that is now
        # intentionally excluded from the foot-contact dynamics model.  The
        # marker is useful to resume/audit tooling and makes stale CSVs
        # unambiguously ineligible without spending hours regenerating them.
        write_json(
            task.output_csv_path.with_suffix(task.output_csv_path.suffix + ".quality.json"),
            {
                "status": "excluded",
                "reason": exclusion,
                "trial": task.trial_name,
                "dynamics_pipeline_version": DYNAMICS_PIPELINE_VERSION,
            },
        )
        return BatchTaskResult(
            task=task,
            status="excluded",
            return_code=0,
            duration_s=0.0,
            error=exclusion,
        )
    is_walking = "_walk_" in task.relative_path.stem.lower()
    has_quality = is_quality_pass(task.output_csv_path)
    artifact_pattern = f"{task.output_csv_path.stem}_*/results"
    has_artifacts = any(path.is_dir() for path in output_root.glob(artifact_pattern))
    can_skip = has_quality and (not args.require_intermediate_artifacts or has_artifacts)
    if args.skip_existing and not (args.rerun_walking and is_walking) and can_skip:
        return BatchTaskResult(
            task=task,
            status="skipped",
            return_code=0,
            duration_s=0.0,
            error=None,
        )

    task.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()

    with task.log_path.open("w", encoding="utf-8") as log_file:
        if args.cleanup_intermediate:
            # Cleanup mode can safely use local scratch because no downstream
            # report needs the OpenSim artifacts after the CSV is published.
            with tempfile.TemporaryDirectory(prefix="groundlink-batch-", dir="/tmp") as scratch:
                cmd = _build_single_run_cmd(args, pipeline_script, task, Path(scratch))
                log_file.write("COMMAND:\n")
                log_file.write(" ".join(cmd) + "\n\n")
                log_file.flush()
                proc = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
        else:
            # E4 needs IK, models, ID sto and contact-wrench artifacts after
            # the batch exits.  Keep them on the output NAS under a unique
            # trial directory, avoiding stale-directory races between reruns.
            unique_trial_name = (
                f"{task.trial_name}_{os.getpid()}_{threading.get_ident()}_{time.time_ns()}"
            )
            cmd = _build_single_run_cmd(
                args,
                pipeline_script,
                task,
                output_root,
                child_trial_name=unique_trial_name,
            )
            log_file.write("COMMAND:\n")
            log_file.write(" ".join(cmd) + "\n\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

    duration = time.time() - start
    if proc.returncode != 0:
        return BatchTaskResult(
            task=task,
            status="failed",
            return_code=proc.returncode,
            duration_s=duration,
            error=f"Pipeline failed (exit code {proc.returncode}). See log: {task.log_path}",
        )
    if not is_nonempty_file(task.output_csv_path):
        return BatchTaskResult(
            task=task,
            status="failed",
            return_code=proc.returncode,
            duration_s=duration,
            error=f"Missing expected output CSV: {task.output_csv_path}",
        )
    return BatchTaskResult(
        task=task,
        status="ok",
        return_code=proc.returncode,
        duration_s=duration,
        error=None,
    )


def _run_parallel(
    args: argparse.Namespace,
    pipeline_script: Path,
    output_root: Path,
    tasks: list[BatchTask],
) -> list[BatchTaskResult]:
    results: list[BatchTaskResult] = []
    total = len(tasks)
    if total == 0:
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_task = {
            executor.submit(_run_single_task, args, pipeline_script, output_root, task): task
            for task in tasks
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            completed += 1
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as exc:
                result = BatchTaskResult(
                    task=task,
                    status="failed",
                    return_code=1,
                    duration_s=0.0,
                    error=f"Batch worker error: {type(exc).__name__}: {exc}",
                )
            results.append(result)
            label = f"[{completed}/{total}]"
            rel = result.task.relative_path.as_posix()
            if result.status == "ok":
                print(
                    f"{label} OK {rel} -> {result.task.output_csv_path} "
                    f"({result.duration_s:.1f}s)"
                )
            elif result.status == "skipped":
                print(f"{label} SKIP {rel} (existing CSV)")
            elif result.status == "excluded":
                print(f"{label} EXCLUDE {rel} ({result.error})")
            else:
                print(
                    f"{label} FAIL {rel} ({result.duration_s:.1f}s) "
                    f"- {result.error}"
                )
    results.sort(key=lambda item: item.task.relative_path.as_posix())
    return results


def _write_summary(
    output_root: Path,
    args: argparse.Namespace,
    input_root: Path,
    results: list[BatchTaskResult],
) -> Path:
    summary_path = output_root / "batch_summary.json"
    payload = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "workers": args.workers,
        "engine_work_root": getattr(args, "engine_work_root", None),
        "total_tasks": len(results),
        "ok": sum(1 for r in results if r.status == "ok"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "skipped": sum(1 for r in results if r.status == "skipped"),
        "excluded": sum(1 for r in results if r.status == "excluded"),
        "results": [
            {
                "relative_path": r.task.relative_path.as_posix(),
                "status": r.status,
                "duration_s": r.duration_s,
                "return_code": r.return_code,
                "output_csv_path": str(r.task.output_csv_path),
                "log_path": str(r.task.log_path),
                "error": r.error,
            }
            for r in results
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return summary_path


def main() -> int:
    args = parse_args()

    if str(args.input_root).startswith("smb://"):
        raise ValueError(
            "SMB URLs are not directly readable by this script. Mount the share first "
            "(for example under /Volumes or a local mount path), then pass that local path "
            "to --input-root."
        )

    input_root = resolve_submit_path(args.input_root, prefer_storage_home=True)
    output_root = resolve_submit_path(args.output_dir, prefer_storage_home=True)
    pipeline_script = resolve_pipeline_script(args.pipeline_script)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root must be a directory: {input_root}")
    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    tasks = _build_tasks(args, input_root=input_root, output_root=output_root)
    print(f"Discovered {len(tasks)} .npz/.npy files under: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Workers: {args.workers}")
    if args.skip_existing:
        existing = sum(1 for task in tasks if is_valid_groundlink_csv(task.output_csv_path))
        print(f"Skip existing CSVs: enabled ({existing} already present)")
    else:
        print("Skip existing CSVs: disabled")

    if args.dry_run:
        for idx, task in enumerate(tasks, start=1):
            print(
                f"[DRY {idx}] {task.relative_path.as_posix()} -> "
                f"{task.output_csv_path} (trial={task.trial_name})"
            )
        return 0

    start = time.time()
    with _batch_lock(output_root):
        results = _run_parallel(args, pipeline_script, output_root, tasks)
        summary_path = _write_summary(output_root, args, input_root, results)
    elapsed = time.time() - start

    ok = sum(1 for r in results if r.status == "ok")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    excluded = sum(1 for r in results if r.status == "excluded")
    print(
        f"Batch completed in {elapsed:.1f}s - ok: {ok}, failed: {failed}, "
        f"skipped: {skipped}, excluded: {excluded}."
    )
    print(f"Summary: {summary_path}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())

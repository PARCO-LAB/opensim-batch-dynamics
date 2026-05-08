#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchTask:
    input_path: Path
    relative_path: Path
    output_npz_path: Path
    log_path: Path


def _resolve_submit_path(raw: str | Path) -> Path:
    path = Path(os.path.expandvars(str(raw))).expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        (Path.cwd() / path).resolve(),
        (repo_root / path).resolve(),
        (Path.home() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_converter_script(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "scripts" / "convert_humanml3d_joints_to_smplx_npz.py").resolve()


def _scan_root(input_root: Path, joints_subdir: str | None) -> Path:
    if joints_subdir:
        candidate = input_root / joints_subdir
        if candidate.is_dir():
            return candidate
    return input_root


def _discover_joint_files(scan_root: Path, limit: int | None) -> list[Path]:
    files = sorted(path for path in scan_root.rglob("*.npy") if path.is_file())
    if limit is not None:
        files = files[: max(0, int(limit))]
    return files


def _is_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _build_tasks(scan_root: Path, output_root: Path, limit: int | None) -> list[BatchTask]:
    tasks: list[BatchTask] = []
    for input_path in _discover_joint_files(scan_root, limit):
        relative_path = input_path.relative_to(scan_root)
        output_npz_path = (output_root / relative_path).with_suffix(".npz")
        log_path = (output_root / "logs" / relative_path).with_suffix(".log")
        tasks.append(
            BatchTask(
                input_path=input_path,
                relative_path=relative_path,
                output_npz_path=output_npz_path,
                log_path=log_path,
            )
        )
    return tasks


def _worker_python(args: argparse.Namespace) -> str:
    if args.slurm_python_exe:
        return args.slurm_python_exe
    if args.slurm_setup_cmd and args.python_exe == sys.executable:
        return "python"
    return args.python_exe


def _build_converter_cmd(args: argparse.Namespace, converter_script: Path, task: BatchTask) -> list[str]:
    python_exe = args.python_exe
    if args.slurm_setup_cmd and args.python_exe == sys.executable:
        python_exe = "python"
    cmd = [
        python_exe,
        str(converter_script),
        "--input",
        str(task.input_path),
        "--output-dir",
        str(task.output_npz_path.parent),
        "--smplx-model-dir",
        args.smplx_model_dir,
        "--gender",
        args.gender,
        "--device",
        args.device,
        "--frame-rate",
        str(args.frame_rate),
        "--target-frame",
        args.target_frame,
        "--num-betas",
        str(args.num_betas),
        "--shape-iters",
        str(args.shape_iters),
        "--shape-lr",
        str(args.shape_lr),
        "--shape-beta-prior-weight",
        str(args.shape_beta_prior_weight),
        "--pose-iters",
        str(args.pose_iters),
        "--pose-lr",
        str(args.pose_lr),
        "--pose-prior-weight",
        str(args.pose_prior_weight),
        "--hand-prior-weight",
        str(args.hand_prior_weight),
        "--temporal-smooth-weight",
        str(args.temporal_smooth_weight),
        "--floor-prior-weight",
        str(args.floor_prior_weight),
    ]
    if args.force:
        cmd.append("--force")
    return cmd


def _write_manifest(path: Path, tasks: list[BatchTask], commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (task, command) in enumerate(zip(tasks, commands)):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "relative_path": task.relative_path.as_posix(),
                        "input_path": str(task.input_path),
                        "output_npz_path": str(task.output_npz_path),
                        "log_path": str(task.log_path),
                        "command": command,
                    }
                )
                + "\n"
            )


def _read_manifest_record(manifest_path: Path, task_index: int) -> dict[str, object]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == task_index:
                return json.loads(line)
    raise IndexError(f"Task index {task_index} out of range: {manifest_path}")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_sbatch_script(args: argparse.Namespace, output_root: Path, manifest_path: Path, task_count: int) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    slurm_root = output_root / "slurm"
    log_dir = slurm_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    script_path = slurm_root / "run_humanml3d_joints_to_npz.sbatch"

    array_spec = f"0-{task_count - 1}"
    if args.slurm_array_parallelism is not None:
        array_spec = f"{array_spec}%{args.slurm_array_parallelism}"

    worker_cmd = [
        _worker_python(args),
        str(Path(__file__).resolve()),
        "worker",
        "--manifest",
        str(manifest_path),
    ]
    if args.skip_existing:
        worker_cmd.append("--skip-existing")
    else:
        worker_cmd.append("--no-skip-existing")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.slurm_job_name}",
        f"#SBATCH --time={args.slurm_time}",
        f"#SBATCH --cpus-per-task={args.slurm_cpus_per_task}",
        f"#SBATCH --mem={args.slurm_mem}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --output={log_dir}/%x_%A_%a.out",
        f"#SBATCH --error={log_dir}/%x_%A_%a.err",
    ]
    if args.slurm_partition:
        lines.append(f"#SBATCH --partition={args.slurm_partition}")
    if args.slurm_account:
        lines.append(f"#SBATCH --account={args.slurm_account}")
    if args.slurm_nodelist:
        lines.append(f"#SBATCH -w {args.slurm_nodelist}")
    if args.slurm_gres:
        lines.append(f"#SBATCH --gres={args.slurm_gres}")

    lines.extend(["", "set -euo pipefail", f"cd {shlex.quote(str(repo_root))}"])
    lines.extend(args.slurm_setup_cmd)
    lines.extend(
        [
            "unset PYTHONPATH",
            "export PYTHONNOUSERSITE=1",
            "export KMP_DUPLICATE_LIB_OK=TRUE",
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
            "export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
            "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
            shlex.join(worker_cmd),
        ]
    )
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _submit(args: argparse.Namespace) -> int:
    input_root = _resolve_submit_path(args.input_root)
    output_root = _resolve_submit_path(args.output_dir)
    scan_root = _scan_root(input_root, args.joints_subdir)
    converter_script = _resolve_converter_script(args.converter_script)
    if not scan_root.is_dir():
        raise NotADirectoryError(f"HumanML3D joints root not found or not a directory: {scan_root}")
    if not converter_script.exists():
        raise FileNotFoundError(f"Converter script not found: {converter_script}")

    tasks = _build_tasks(scan_root, output_root, args.limit)
    skipped = 0
    runnable: list[BatchTask] = []
    for task in tasks:
        if args.skip_existing and _is_ready(task.output_npz_path):
            skipped += 1
        else:
            runnable.append(task)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "slurm" / "manifest.jsonl"
    commands = [_build_converter_cmd(args, converter_script, task) for task in runnable]
    _write_manifest(manifest_path, runnable, commands)

    plan = {
        "mode": "submit",
        "input_root": str(input_root),
        "scan_root": str(scan_root),
        "output_root": str(output_root),
        "converter_script": str(converter_script),
        "total_discovered": len(tasks),
        "skipped_existing": skipped,
        "scheduled_tasks": len(runnable),
        "manifest_path": str(manifest_path),
        "submitted": False,
    }

    print(f"Scan root: {scan_root}")
    print(f"Found .npy: {len(tasks)}")
    print(f"Skip existing npz: {skipped}")
    print(f"Scheduled: {len(runnable)}")
    print(f"Manifest: {manifest_path}")
    if not runnable:
        _write_json(output_root / "slurm" / "submit_plan.json", plan)
        print("Nothing to submit.")
        return 0

    sbatch_path = _write_sbatch_script(args, output_root, manifest_path, len(runnable))
    sbatch_cmd = ["sbatch", str(sbatch_path)]
    plan["sbatch_script"] = str(sbatch_path)
    plan["sbatch_command"] = sbatch_cmd
    _write_json(output_root / "slurm" / "submit_plan.json", plan)
    print(f"SBATCH: {sbatch_path}")
    print(f"Command: {shlex.join(sbatch_cmd)}")

    if args.dry_run:
        for task in runnable[:5]:
            print(f"[DRY] {task.relative_path.as_posix()} -> {task.output_npz_path}")
        return 0
    if not args.submit:
        print("Add --submit to launch.")
        return 0

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=False)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip())
    plan["submitted"] = result.returncode == 0
    plan["sbatch_stdout"] = result.stdout.strip()
    plan["sbatch_stderr"] = result.stderr.strip()
    _write_json(output_root / "slurm" / "submit_plan.json", plan)
    return result.returncode


def _worker(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    raw_index = os.environ.get("SLURM_ARRAY_TASK_ID", args.task_index)
    if raw_index is None:
        raise ValueError("Need SLURM_ARRAY_TASK_ID or --task-index")
    task_index = int(raw_index) + int(args.task_index_offset)
    record = _read_manifest_record(manifest_path, task_index)

    output_npz = Path(str(record["output_npz_path"])).resolve()
    log_path = Path(str(record["log_path"])).resolve()
    result_path = manifest_path.parent / "results" / f"task_{task_index:06d}.json"

    if args.skip_existing and _is_ready(output_npz):
        payload = {
            "task_index": task_index,
            "status": "skipped_existing",
            "relative_path": record["relative_path"],
            "output_npz_path": str(output_npz),
        }
        _write_json(result_path, payload)
        print(json.dumps(payload))
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    command = [str(item) for item in record["command"]]
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    duration_s = time.time() - start
    log_path.write_text(
        "COMMAND:\n"
        + shlex.join(command)
        + "\n\nSTDOUT:\n"
        + proc.stdout
        + "\n\nSTDERR:\n"
        + proc.stderr,
        encoding="utf-8",
    )
    status = "ok" if proc.returncode == 0 and _is_ready(output_npz) else "failed"
    payload = {
        "task_index": task_index,
        "status": status,
        "returncode": proc.returncode,
        "duration_s": duration_s,
        "relative_path": record["relative_path"],
        "input_path": record["input_path"],
        "output_npz_path": str(output_npz),
        "log_path": str(log_path),
    }
    _write_json(result_path, payload)
    print(json.dumps(payload))
    return 0 if status == "ok" else 1


def _add_common_converter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", required=True, help="HumanML3D root or its joints/ directory.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--joints-subdir", default="joints", help="Auto-use this child directory when it exists.")
    parser.add_argument("--converter-script", default=None)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Forward --force to converter.")

    parser.add_argument("--smplx-model-dir", default="model/smpl")
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-rate", type=float, default=20.0)
    parser.add_argument("--target-frame", choices=["z-up", "y-up"], default="z-up")
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--shape-iters", type=int, default=800)
    parser.add_argument("--shape-lr", type=float, default=0.04)
    parser.add_argument("--shape-beta-prior-weight", type=float, default=0.01)
    parser.add_argument("--pose-iters", type=int, default=1600)
    parser.add_argument("--pose-lr", type=float, default=0.018)
    parser.add_argument("--pose-prior-weight", type=float, default=0.0005)
    parser.add_argument("--hand-prior-weight", type=float, default=0.001)
    parser.add_argument("--temporal-smooth-weight", type=float, default=0.01)
    parser.add_argument("--floor-prior-weight", type=float, default=0.5)


def _add_submit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--slurm-job-name", default="humanml3d_smplx")
    parser.add_argument("--slurm-partition", default=None)
    parser.add_argument("--slurm-account", default=None)
    parser.add_argument("--slurm-nodelist", "--slurm-node", dest="slurm_nodelist", default=None)
    parser.add_argument("--slurm-time", default="08:00:00")
    parser.add_argument("--slurm-cpus-per-task", type=int, default=4)
    parser.add_argument("--slurm-mem", default="16G")
    parser.add_argument("--slurm-gres", default=None, help="Optional SLURM --gres, e.g. gpu:1.")
    parser.add_argument("--slurm-array-parallelism", type=int, default=None)
    parser.add_argument("--slurm-python-exe", default=None)
    parser.add_argument("--slurm-setup-cmd", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and optionally submit SLURM array jobs for HumanML3D joints/*.npy -> SMPL-X .npz conversion."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit")
    _add_common_converter_args(submit)
    _add_submit_args(submit)
    submit.set_defaults(func=_submit)

    worker = sub.add_parser("worker")
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--task-index", default=None)
    worker.add_argument("--task-index-offset", type=int, default=0)
    worker.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    worker.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    worker.set_defaults(skip_existing=True)
    worker.set_defaults(func=_worker)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

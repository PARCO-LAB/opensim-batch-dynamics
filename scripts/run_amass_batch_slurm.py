#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HPC_INPUT_ROOT = "storage/emartini/AMASS/AMASS"
DEFAULT_HPC_OUTPUT_DIR = "storage/emartini/AMASS_torque"


@dataclass(frozen=True)
class BatchTask:
    input_path: Path
    relative_path: Path
    output_csv_path: Path
    log_path: Path
    trial_name: str


@dataclass(frozen=True)
class SbatchChunk:
    script_path: Path
    sbatch_cmd: list[str]
    task_index_offset: int
    task_count: int


def _sanitize_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return cleaned or "trial"


def _build_trial_name(relative_npz_path: Path) -> str:
    base = relative_npz_path.with_suffix("").as_posix()
    prefix = _sanitize_component(base.replace("/", "__"))
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    trial_name = f"{prefix}_{digest}"
    if len(trial_name) > 140:
        trial_name = trial_name[:131] + "_" + digest
    return trial_name


def _resolve_pipeline_script(path_from_arg: str | None) -> Path:
    if path_from_arg:
        return Path(path_from_arg).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "scripts" / "run_amass_to_bsm_csv.py").resolve()


def _resolve_submit_path(path_from_arg: str) -> Path:
    raw_path = Path(os.path.expandvars(path_from_arg)).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    repo_root = Path(__file__).resolve().parents[1]
    home_candidate = (Path.home() / raw_path).resolve()
    cwd_candidate = (Path.cwd() / raw_path).resolve()
    repo_candidate = (repo_root / raw_path).resolve()

    if raw_path.parts and raw_path.parts[0] == "storage":
        candidates = [home_candidate, cwd_candidate, repo_candidate]
    else:
        candidates = [cwd_candidate, repo_candidate, home_candidate]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _discover_npz_files(input_root: Path) -> list[Path]:
    excluded_names = {
        "shape.npz",
        "neutral_stagei.npz",
        "female_stagei.npz",
        "male_stagei.npz",
    }
    files = [
        path
        for path in input_root.rglob("*.npz")
        if path.is_file() and path.name.lower() not in excluded_names
    ]
    files.sort()
    return files


def _is_existing_csv_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _build_tasks(input_root: Path, output_root: Path, limit: int | None) -> list[BatchTask]:
    files = _discover_npz_files(input_root)
    if limit is not None:
        files = files[: max(0, limit)]

    tasks: list[BatchTask] = []
    for input_path in files:
        relative_path = input_path.relative_to(input_root)
        output_csv_path = (output_root / relative_path).with_suffix(".csv")
        log_path = (output_root / "logs" / relative_path).with_suffix(".log")
        trial_name = _build_trial_name(relative_path)
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
) -> list[str]:
    # When SLURM setup commands activate an environment, prefer its `python`
    # for child pipeline runs unless user explicitly overrides `--python-exe`.
    python_exe = args.python_exe
    if args.slurm_setup_cmd and args.python_exe == sys.executable:
        python_exe = "python"

    cmd = [
        python_exe,
        str(pipeline_script),
        "--input",
        str(task.input_path),
        "--trial",
        task.trial_name,
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
    if args.addbio_root:
        cmd.extend(["--addbio-root", args.addbio_root])
    if args.sex:
        cmd.extend(["--sex", args.sex])
    if args.id_cutoff_hz is not None:
        cmd.extend(["--id-cutoff-hz", str(args.id_cutoff_hz)])
    if args.skip_inverse_dynamics:
        cmd.append("--skip-inverse-dynamics")
    if args.cleanup_intermediate:
        cmd.append("--cleanup-intermediate")
    return cmd


def _write_manifest(output_root: Path, tasks: list[BatchTask], commands: list[list[str]]) -> Path:
    slurm_root = output_root / "slurm"
    slurm_root.mkdir(parents=True, exist_ok=True)
    manifest_path = slurm_root / "manifest.jsonl"
    if len(tasks) != len(commands):
        raise ValueError(
            f"Task/command count mismatch while writing manifest: {len(tasks)} tasks vs "
            f"{len(commands)} commands"
        )
    with manifest_path.open("w", encoding="utf-8") as handle:
        for index, (task, command) in enumerate(zip(tasks, commands)):
            payload = {
                "index": index,
                "relative_path": task.relative_path.as_posix(),
                "input_path": str(task.input_path),
                "output_csv_path": str(task.output_csv_path),
                "log_path": str(task.log_path),
                "trial_name": task.trial_name,
                "command": command,
            }
            handle.write(json.dumps(payload) + "\n")
    return manifest_path


def _write_sbatch_script(
    args: argparse.Namespace,
    output_root: Path,
    manifest_path: Path,
    task_count: int,
    task_index_offset: int = 0,
    chunk_index: int = 0,
) -> SbatchChunk:
    repo_root = Path(__file__).resolve().parents[1]
    slurm_root = output_root / "slurm"
    slurm_root.mkdir(parents=True, exist_ok=True)

    log_dir = (
        Path(args.slurm_log_dir).resolve()
        if args.slurm_log_dir
        else (slurm_root / "logs").resolve()
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    script_name = "run_batch.sbatch" if chunk_index == 0 else f"run_batch_{chunk_index:04d}.sbatch"
    script_path = slurm_root / script_name
    array_spec = f"0-{task_count - 1}"
    if args.slurm_array_parallelism is not None:
        if args.slurm_array_parallelism < 1:
            raise ValueError("--slurm-array-parallelism must be >= 1")
        array_spec = f"{array_spec}%{args.slurm_array_parallelism}"

    # If user provides setup commands (e.g. conda activate), prefer `python`
    # from the activated environment unless explicitly overridden.
    if args.slurm_python_exe:
        slurm_python_exe = args.slurm_python_exe
    elif args.slurm_setup_cmd:
        slurm_python_exe = "python"
    else:
        slurm_python_exe = args.python_exe
    worker_cmd = [
        slurm_python_exe,
        str(Path(__file__).resolve()),
        "worker",
        "--manifest",
        str(manifest_path),
        "--task-index-offset",
        str(task_index_offset),
    ]
    if args.skip_existing:
        worker_cmd.append("--skip-existing-csv")
    else:
        worker_cmd.append("--no-skip-existing-csv")

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
    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"cd {shlex.quote(str(repo_root))}",
        ]
    )
    for setup_cmd in args.slurm_setup_cmd:
        lines.append(setup_cmd)
    lines.extend(
        [
            # Avoid importing incompatible packages from ~/.local on compute nodes.
            "unset PYTHONPATH",
            "export PYTHONNOUSERSITE=1",
            # If a Conda env is active, ensure `python` resolves from it.
            'if [[ -n "${CONDA_PREFIX:-}" ]]; then',
            '  PY_BIN="$(command -v python || true)"',
            '  case "$PY_BIN" in',
            '    "$CONDA_PREFIX"/*) ;;',
            '    *)',
            '      echo "ERROR: python does not come from active CONDA_PREFIX=$CONDA_PREFIX";',
            '      echo "Resolved python: $PY_BIN";',
            "      exit 2;",
            "      ;;",
            "  esac",
            "fi",
        ]
    )
    lines.extend(
        [
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
            "export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
            "export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
            shlex.join(worker_cmd),
        ]
    )

    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    sbatch_cmd = ["sbatch", str(script_path)]
    return SbatchChunk(
        script_path=script_path,
        sbatch_cmd=sbatch_cmd,
        task_index_offset=task_index_offset,
        task_count=task_count,
    )


def _build_sbatch_chunks(
    args: argparse.Namespace,
    output_root: Path,
    manifest_path: Path,
    task_count: int,
) -> list[SbatchChunk]:
    max_array_size = args.slurm_max_array_size
    if max_array_size is not None and max_array_size < 1:
        raise ValueError("--slurm-max-array-size must be >= 1")

    chunk_size = task_count
    if max_array_size is not None:
        chunk_size = min(task_count, max_array_size)

    chunks: list[SbatchChunk] = []
    task_index_offset = 0
    chunk_index = 0
    while task_index_offset < task_count:
        current_task_count = min(chunk_size, task_count - task_index_offset)
        chunks.append(
            _write_sbatch_script(
                args=args,
                output_root=output_root,
                manifest_path=manifest_path,
                task_count=current_task_count,
                task_index_offset=task_index_offset,
                chunk_index=chunk_index,
            )
        )
        task_index_offset += current_task_count
        chunk_index += 1
    return chunks


def _read_manifest_record(manifest_path: Path, task_index: int) -> dict[str, object]:
    if task_index < 0:
        raise ValueError("Task index must be >= 0")
    with manifest_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == task_index:
                return json.loads(line)
    raise IndexError(
        f"Task index {task_index} is out of range for manifest: {manifest_path}"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _should_retry_sbatch(stderr_text: str) -> bool:
    msg = (stderr_text or "").lower()
    return (
        "temporarily unable to accept job" in msg
        or "resource temporarily unavailable" in msg
        or "socket timed out" in msg
        or "connection timed out" in msg
    )


def _submit_chunk_with_retry(
    chunk: SbatchChunk,
    retries: int,
    initial_sleep_s: float,
) -> tuple[subprocess.CompletedProcess[str], int]:
    """
    Submit one sbatch chunk with retry/backoff on transient scheduler errors.

    Returns the final CompletedProcess and the number of retries performed.
    """
    attempts = 0
    sleep_s = max(0.1, float(initial_sleep_s))
    while True:
        result = subprocess.run(
            chunk.sbatch_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result, attempts
        if attempts >= retries:
            return result, attempts
        if not _should_retry_sbatch(result.stderr):
            return result, attempts
        attempts += 1
        print(
            f"Transient sbatch failure for chunk offset={chunk.task_index_offset} "
            f"(attempt {attempts}/{retries}). Retrying in {sleep_s:.1f}s..."
        )
        time.sleep(sleep_s)
        sleep_s = min(sleep_s * 2.0, 60.0)


def _load_previous_result_statuses(output_root: Path) -> dict[str, str]:
    """
    Load previous SLURM worker results, keyed by relative_path.

    Newer result files (by mtime) overwrite older statuses for the same path.
    """
    results_dir = output_root / "slurm" / "results"
    if not results_dir.exists():
        return {}
    files = sorted(results_dir.glob("task_*.json"), key=lambda p: p.stat().st_mtime)
    statuses: dict[str, str] = {}
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rel = str(payload.get("relative_path", "")).strip()
            status = str(payload.get("status", "")).strip().lower()
            if rel and status:
                statuses[rel] = status
        except Exception:
            # Ignore malformed historical files.
            continue
    return statuses


def _run_submit(args: argparse.Namespace) -> int:
    if str(args.input_root).startswith("smb://"):
        raise ValueError(
            "SMB URLs are not directly readable by this script. Mount the share first, then "
            "pass the mounted local path to --input-root."
        )

    input_root = _resolve_submit_path(args.input_root)
    output_root = _resolve_submit_path(args.output_dir)
    pipeline_script = _resolve_pipeline_script(args.pipeline_script)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root must be a directory: {input_root}")
    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")
    if args.slurm_cpus_per_task < 1:
        raise ValueError("--slurm-cpus-per-task must be >= 1")

    tasks = _build_tasks(input_root=input_root, output_root=output_root, limit=args.limit)
    previous_status = _load_previous_result_statuses(output_root=output_root)
    skipped_existing = 0
    skipped_previous_ok = 0
    runnable: list[BatchTask] = []
    for task in tasks:
        if args.skip_existing and _is_existing_csv_ready(task.output_csv_path):
            skipped_existing += 1
            continue
        if args.skip_previously_ok and previous_status.get(task.relative_path.as_posix()) == "ok":
            skipped_previous_ok += 1
            continue
        runnable.append(task)

    total_runnable = len(runnable)
    if args.max_submit_tasks is not None:
        if args.max_submit_tasks < 1:
            raise ValueError("--max-submit-tasks must be >= 1")
        runnable = runnable[: args.max_submit_tasks]

    print(f"Discovered {len(tasks)} .npz files under: {input_root}")
    print(f"Skip existing CSVs: {'enabled' if args.skip_existing else 'disabled'}")
    print(f"Already present CSVs skipped at submit-time: {skipped_existing}")
    print(
        "Previously successful tasks skipped at submit-time: "
        f"{skipped_previous_ok} (from {output_root / 'slurm' / 'results'})"
    )
    if args.max_submit_tasks is not None:
        print(f"Max tasks per submit: {args.max_submit_tasks}")
    print(f"Total pending before cap: {total_runnable}")
    print(f"Runnable SLURM tasks: {len(runnable)}")

    slurm_root = output_root / "slurm"
    plan_path = slurm_root / "submit_plan.json"

    if not runnable:
        _write_json(
            plan_path,
            {
                "input_root": str(input_root),
                "output_root": str(output_root),
                "pipeline_script": str(pipeline_script),
                "total_discovered": len(tasks),
                "skipped_existing": skipped_existing,
                "skipped_previous_ok": skipped_previous_ok,
                "pending_before_cap": total_runnable,
                "scheduled_tasks": 0,
                "sbatch_script": None,
                "sbatch_command": None,
                "submitted": False,
            },
        )
        print("Nothing to submit: all discovered outputs are already present.")
        print(f"Plan summary: {plan_path}")
        return 0

    commands = [
        _build_single_run_cmd(args, pipeline_script=pipeline_script, task=task, output_root=output_root)
        for task in runnable
    ]
    manifest_path = _write_manifest(output_root=output_root, tasks=runnable, commands=commands)
    sbatch_chunks = _build_sbatch_chunks(
        args=args,
        output_root=output_root,
        manifest_path=manifest_path,
        task_count=len(runnable),
    )
    sbatch_paths = [str(chunk.script_path) for chunk in sbatch_chunks]
    sbatch_cmds = [chunk.sbatch_cmd for chunk in sbatch_chunks]

    _write_json(
        plan_path,
        {
            "input_root": str(input_root),
            "output_root": str(output_root),
            "pipeline_script": str(pipeline_script),
            "total_discovered": len(tasks),
            "skipped_existing": skipped_existing,
            "skipped_previous_ok": skipped_previous_ok,
            "pending_before_cap": total_runnable,
            "scheduled_tasks": len(runnable),
            "manifest_path": str(manifest_path),
            "sbatch_script": sbatch_paths[0],
            "sbatch_command": sbatch_cmds[0],
            "sbatch_scripts": sbatch_paths,
            "sbatch_commands": sbatch_cmds,
            "sbatch_chunk_count": len(sbatch_chunks),
            "submitted": False,
        },
    )

    print(f"Manifest: {manifest_path}")
    if len(sbatch_chunks) == 1:
        print(f"SBATCH script: {sbatch_chunks[0].script_path}")
        print(f"SBATCH command: {shlex.join(sbatch_chunks[0].sbatch_cmd)}")
    else:
        print(f"SBATCH scripts: {len(sbatch_chunks)} chunks")
        for chunk in sbatch_chunks[:5]:
            print(
                f"  chunk offset={chunk.task_index_offset} count={chunk.task_count}: "
                f"{shlex.join(chunk.sbatch_cmd)}"
            )
        if len(sbatch_chunks) > 5:
            print(f"  ... {len(sbatch_chunks) - 5} more chunks")

    if args.dry_run:
        preview = min(5, len(runnable))
        for idx, task in enumerate(runnable[:preview], start=1):
            print(
                f"[DRY {idx}] {task.relative_path.as_posix()} -> {task.output_csv_path} "
                f"(trial={task.trial_name})"
            )
        print(f"Plan summary: {plan_path}")
        return 0

    if not args.submit:
        print("Submission not requested. Run the SBATCH command above when ready.")
        print(f"Plan summary: {plan_path}")
        return 0

    submitted_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    submission_results: list[dict[str, object]] = []
    all_ok = True
    combined_stdout: list[str] = []
    combined_stderr: list[str] = []
    for chunk in sbatch_chunks:
        result, retries_used = _submit_chunk_with_retry(
            chunk=chunk,
            retries=args.sbatch_retries,
            initial_sleep_s=args.sbatch_retry_sleep_s,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
            combined_stdout.append(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
            combined_stderr.append(result.stderr.strip())
        submission_results.append(
            {
                "script_path": str(chunk.script_path),
                "sbatch_command": chunk.sbatch_cmd,
                "task_index_offset": chunk.task_index_offset,
                "task_count": chunk.task_count,
                "retry_count": retries_used,
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        )
        if result.returncode != 0:
            all_ok = False
        if args.sbatch_submit_interval_s > 0:
            time.sleep(args.sbatch_submit_interval_s)

    submitted_payload["submitted"] = all_ok
    submitted_payload["sbatch_stdout"] = "\n".join(combined_stdout)
    submitted_payload["sbatch_stderr"] = "\n".join(combined_stderr)
    submitted_payload["sbatch_submission_results"] = submission_results
    _write_json(plan_path, submitted_payload)
    print(f"Plan summary: {plan_path}")
    return 0 if all_ok else 1


def _run_worker(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if args.task_index is not None:
        task_index = args.task_index
    else:
        raw_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw_idx is None:
            raise ValueError(
                "Task index not provided. Use --task-index or run under a SLURM array "
                "job with SLURM_ARRAY_TASK_ID."
            )
        task_index = int(raw_idx)
    task_index += int(args.task_index_offset)

    record = _read_manifest_record(manifest_path, task_index=task_index)
    command = [str(token) for token in record["command"]]  # type: ignore[index]
    relative_path = str(record["relative_path"])
    output_csv_path = Path(str(record["output_csv_path"])).resolve()
    log_path = Path(str(record["log_path"])).resolve()

    result_dir = manifest_path.parent / "results"
    result_path = result_dir / f"task_{task_index:06d}.json"
    started_at = time.time()

    if args.skip_existing and _is_existing_csv_ready(output_csv_path):
        payload = {
            "task_index": task_index,
            "relative_path": relative_path,
            "status": "skipped",
            "return_code": 0,
            "duration_s": 0.0,
            "log_path": str(log_path),
            "output_csv_path": str(output_csv_path),
            "error": None,
        }
        _write_json(result_path, payload)
        print(f"[SLURM {task_index}] SKIP {relative_path} (existing CSV)")
        return 0

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(shlex.join(command) + "\n\n")
        log_file.flush()
        proc = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    duration_s = time.time() - started_at
    if proc.returncode != 0:
        payload = {
            "task_index": task_index,
            "relative_path": relative_path,
            "status": "failed",
            "return_code": proc.returncode,
            "duration_s": duration_s,
            "log_path": str(log_path),
            "output_csv_path": str(output_csv_path),
            "error": f"Pipeline failed (exit code {proc.returncode}).",
        }
        _write_json(result_path, payload)
        print(
            f"[SLURM {task_index}] FAIL {relative_path} ({duration_s:.1f}s) - "
            f"see log: {log_path}"
        )
        return 1

    if not _is_existing_csv_ready(output_csv_path):
        payload = {
            "task_index": task_index,
            "relative_path": relative_path,
            "status": "failed",
            "return_code": proc.returncode,
            "duration_s": duration_s,
            "log_path": str(log_path),
            "output_csv_path": str(output_csv_path),
            "error": f"Missing expected output CSV: {output_csv_path}",
        }
        _write_json(result_path, payload)
        print(
            f"[SLURM {task_index}] FAIL {relative_path} ({duration_s:.1f}s) - "
            f"missing output CSV"
        )
        return 1

    payload = {
        "task_index": task_index,
        "relative_path": relative_path,
        "status": "ok",
        "return_code": proc.returncode,
        "duration_s": duration_s,
        "log_path": str(log_path),
        "output_csv_path": str(output_csv_path),
        "error": None,
    }
    _write_json(result_path, payload)
    print(
        f"[SLURM {task_index}] OK {relative_path} -> {output_csv_path} "
        f"({duration_s:.1f}s)"
    )
    return 0


def _add_pipeline_forwarded_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--pipeline-script", default=None)
    parser.add_argument("--smplx-model-dir", default="model/smpl")
    parser.add_argument("--bsm-model", default="model/bsm/bsm.osim")
    parser.add_argument("--addbio-root", default=None)
    parser.add_argument("--sex", choices=["neutral", "male", "female"], default=None)
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
    parser.add_argument("--id-contact-bodies", default="all")
    parser.add_argument("--id-friction-coeff", type=float, default=0.8)
    parser.add_argument("--id-contact-height-threshold-m", type=float, default=0.06)
    parser.add_argument("--id-contact-speed-threshold-mps", type=float, default=0.6)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SLURM helper for AMASS->OpenSim->CSV batch execution. "
            "Use 'submit' to prepare/submit an array job, and 'worker' inside SLURM."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    submit = subparsers.add_parser(
        "submit",
        help="Build manifest + SBATCH script and optionally submit SLURM array jobs.",
    )
    submit.add_argument(
        "--input-root",
        default=DEFAULT_HPC_INPUT_ROOT,
        help=(
            "Root folder containing AMASS .npz files. Relative paths starting with "
            "'storage/' are resolved from $HOME (default: %(default)s)."
        ),
    )
    submit.add_argument(
        "--output-dir",
        default=DEFAULT_HPC_OUTPUT_DIR,
        help=(
            "Destination root for generated CSVs/logs. Relative paths starting with "
            "'storage/' are resolved from $HOME (default: %(default)s)."
        ),
    )
    submit.add_argument("--limit", type=int, default=None)
    submit.add_argument(
        "--max-submit-tasks",
        type=int,
        default=None,
        help=(
            "Optional cap on number of pending tasks to schedule in this submit call. "
            "Useful to avoid overloading the scheduler."
        ),
    )
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument(
        "--submit",
        action="store_true",
        help="If set, call sbatch automatically. Otherwise only generate files/instructions.",
    )
    _add_pipeline_forwarded_args(submit)
    submit.add_argument(
        "--skip-previously-ok",
        dest="skip_previously_ok",
        action="store_true",
        help=(
            "Skip files marked as successful in previous SLURM worker results "
            "(default: enabled)."
        ),
    )
    submit.add_argument(
        "--no-skip-previously-ok",
        dest="skip_previously_ok",
        action="store_false",
        help="Do not use previous SLURM worker results for submit-time filtering.",
    )
    submit.set_defaults(skip_previously_ok=True)
    submit.add_argument("--slurm-job-name", default="amass_bsm")
    submit.add_argument("--slurm-partition", default=None)
    submit.add_argument("--slurm-account", default=None)
    submit.add_argument(
        "--slurm-nodelist",
        default=None,
        help=(
            "Optional subset of nodes to use, passed through to SBATCH as '-w'. "
            "Example: 'node[001-002],blade[010-012]'."
        ),
    )
    submit.add_argument("--slurm-time", default="08:00:00")
    submit.add_argument("--slurm-cpus-per-task", type=int, default=4)
    submit.add_argument("--slurm-mem", default="16G")
    submit.add_argument(
        "--sbatch-retries",
        type=int,
        default=6,
        help="Retry count for transient sbatch submission failures (default: %(default)s).",
    )
    submit.add_argument(
        "--sbatch-retry-sleep-s",
        type=float,
        default=5.0,
        help="Initial retry sleep in seconds for transient sbatch failures (default: %(default)s).",
    )
    submit.add_argument(
        "--sbatch-submit-interval-s",
        type=float,
        default=1.0,
        help=(
            "Sleep interval between sbatch submissions (seconds) to avoid overloading "
            "the scheduler (default: %(default)s)."
        ),
    )
    submit.add_argument(
        "--slurm-max-array-size",
        type=int,
        default=1000,
        help=(
            "Maximum number of tasks per submitted SLURM array chunk. Large batches are "
            "split into multiple sbatch scripts automatically (default: %(default)s)."
        ),
    )
    submit.add_argument(
        "--slurm-array-parallelism",
        type=int,
        default=None,
        help="Optional max number of concurrent tasks in the SLURM array.",
    )
    submit.add_argument(
        "--slurm-log-dir",
        default=None,
        help="Optional path for SBATCH stdout/stderr logs (default: <output-dir>/slurm/logs).",
    )
    submit.add_argument(
        "--slurm-python-exe",
        default=None,
        help="Python executable on compute nodes (default: --python-exe).",
    )
    submit.add_argument(
        "--slurm-setup-cmd",
        action="append",
        default=[],
        help=(
            "Optional setup command executed inside the SBATCH script before launching each "
            "worker (can be passed multiple times)."
        ),
    )

    worker = subparsers.add_parser(
        "worker",
        help="Internal mode used by SLURM array jobs.",
    )
    worker.add_argument("--manifest", required=True)
    worker.add_argument("--task-index", type=int, default=None)
    worker.add_argument("--task-index-offset", type=int, default=0)
    worker.add_argument(
        "--skip-existing-csv",
        dest="skip_existing",
        action="store_true",
        help="Skip if output CSV exists (default: enabled).",
    )
    worker.add_argument(
        "--no-skip-existing-csv",
        dest="skip_existing",
        action="store_false",
        help="Force re-run even if CSV already exists.",
    )
    worker.set_defaults(skip_existing=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "submit":
        return _run_submit(args)
    if args.mode == "worker":
        return _run_worker(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import errno
import csv
import json
import math
import os
import re
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_INPUT_NAMES = {
    "shape.npz",
    "neutral_stagei.npz",
    "female_stagei.npz",
    "male_stagei.npz",
}

# Trials whose kinematics violate the foot-contact-only dynamics model.  Keep
# this list in the batch layer as well as the validator so a full rerun does
# not spend hours regenerating outputs that must be excluded from E4.
EXCLUDED_TRIALS = {
    ("s002", "jumpingjack_3"): "invalid_pelvis_kinematics",
    ("s002", "balletsmalljump_1"): "nonballistic_flight_kinematics",
}

# Bump this whenever an existing CSV must not be reused by --skip-existing.
# Version 7 additionally records the post-fallback contact activation used by
# E4 eligibility, so airborne/negative-normal frames cannot be reused as valid
# contact comparisons.
DYNAMICS_PIPELINE_VERSION = "opensim_root_force_target_zup_v7"


@dataclass(frozen=True)
class BatchTask:
    input_path: Path
    relative_path: Path
    output_csv_path: Path
    log_path: Path
    trial_name: str


def _sanitize_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return cleaned or "trial"


def build_trial_name(relative_npz_path: Path) -> str:
    base = relative_npz_path.with_suffix("").as_posix()
    prefix = _sanitize_component(base.replace("/", "__"))
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    trial_name = f"{prefix}_{digest}"
    if len(trial_name) > 140:
        trial_name = trial_name[:131] + "_" + digest
    return trial_name


def excluded_trial_reason(relative_path: Path) -> str | None:
    """Return the dataset exclusion reason for an AMASS relative path."""
    stem = relative_path.stem.replace("_stageii", "")
    parts = stem.split("_")
    if len(parts) < 3 or not parts[0].startswith("s"):
        return None
    subject = parts[0]
    trial = "_".join(parts[2:])
    if trial.endswith("_full"):
        trial = trial[:-5]
    return EXCLUDED_TRIALS.get((subject, trial))


def resolve_pipeline_script(path_from_arg: str | None) -> Path:
    if path_from_arg:
        return Path(path_from_arg).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / "scripts" / "run_amass_to_bsm_csv.py").resolve()


def resolve_submit_path(path_from_arg: str | Path, prefer_storage_home: bool = False) -> Path:
    raw_path = Path(os.path.expandvars(str(path_from_arg))).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    repo_root = Path(__file__).resolve().parents[1]
    home_candidate = (Path.home() / raw_path).resolve()
    cwd_candidate = (Path.cwd() / raw_path).resolve()
    repo_candidate = (repo_root / raw_path).resolve()

    if prefer_storage_home and raw_path.parts and raw_path.parts[0] == "storage":
        candidates = [home_candidate, cwd_candidate, repo_candidate]
    else:
        candidates = [cwd_candidate, repo_candidate, home_candidate]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def current_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def read_manifest_record(manifest_path: Path, task_index: int) -> dict[str, object]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == task_index:
                return json.loads(line)
    raise IndexError(f"Task index {task_index} out of range: {manifest_path}")


def is_nonempty_file(path: Path) -> bool:
    # ponytail: NAS stat calls can transiently return EAGAIN/ESTALE; retry here
    # so one filesystem hiccup does not abort an otherwise independent batch job.
    for attempt in range(4):
        try:
            info = path.stat()
            return path.is_file() and info.st_size > 0
        except FileNotFoundError:
            return False
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.ESTALE) or attempt == 3:
                return False
            time.sleep(0.25 * (attempt + 1))
    return False


def is_quality_pass(path: Path) -> bool:
    """Return whether an output passed with the current dynamics algorithm."""
    if not is_nonempty_file(path):
        return False
    sidecar = path.with_suffix(path.suffix + ".quality.json")
    try:
        report = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            report.get("status") == "ok"
            and report.get("dynamics_pipeline_version") == DYNAMICS_PIPELINE_VERSION
        )
    except (OSError, ValueError, TypeError):
        return False


def is_valid_groundlink_csv(path: Path) -> bool:
    """Return true only for a non-empty CSV with finite subject metadata."""
    if not is_nonempty_file(path):
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        return all(math.isfinite(float(row[name])) for name in ("subject_mass_kg", "subject_height_m"))
    except (StopIteration, OSError, KeyError, TypeError, ValueError):
        return False


def discover_amass_input_files(input_root: Path) -> list[Path]:
    files = [
        path
        for pattern in ("*.npz", "*.npy")
        for path in input_root.rglob(pattern)
        if path.is_file() and path.name.lower() not in EXCLUDED_INPUT_NAMES
    ]
    files.sort()
    return files

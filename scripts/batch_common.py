from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_INPUT_NAMES = {
    "shape.npz",
    "neutral_stagei.npz",
    "female_stagei.npz",
    "male_stagei.npz",
}


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


def read_manifest_record(manifest_path: Path, task_index: int) -> dict[str, object]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == task_index:
                return json.loads(line)
    raise IndexError(f"Task index {task_index} out of range: {manifest_path}")


def is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def discover_amass_input_files(input_root: Path) -> list[Path]:
    files = [
        path
        for pattern in ("*.npz", "*.npy")
        for path in input_root.rglob(pattern)
        if path.is_file() and path.name.lower() not in EXCLUDED_INPUT_NAMES
    ]
    files.sort()
    return files

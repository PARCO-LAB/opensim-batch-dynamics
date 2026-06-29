#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smplx.transfer_model import merge_output as transfer_merge_output
from smplx.transfer_model.transfer_model import prepare_fitting_assets, run_fitting
from smplx.transfer_model.utils import read_deformation_transfer


DEFAULT_INPUT_ROOT = Path("/Volumes/MAEVE/dataset/CARE-PD/Canonicalized_SMPL_pickles")
DEFAULT_OUTPUT_DIR = Path("/Volumes/MAEVE/dataset/CARE-PD/npz")
DEFAULT_FRAME_RATE_HZ = 30.0
DEFAULT_NUM_BETAS = 16
DEFAULT_SMPLX_REPO_ROOT = Path(__file__).resolve().parents[1] / "smplx"
DEFAULT_TRANSFER_DATA_ROOT = Path(__file__).resolve().parents[1] / "transfer_data"
DEFAULT_SMPL_MODEL_ROOT = Path(__file__).resolve().parents[1] / "model" / "smpl"
TRANSFER_MODEL_NUM_BETAS = 10
TRANSFER_MODEL_NUM_EXPRESSION = 10
_IN_MEMORY_MERGE_KEYS = (
    "transl",
    "global_orient",
    "body_pose",
    "betas",
    "left_hand_pose",
    "right_hand_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "expression",
)


_IN_MEMORY_RUNTIME_CACHE: dict[tuple[str, ...], InMemoryWorkerRuntime] = {}


@dataclass(frozen=True)
class ConversionConfig:
    input_root: Path
    output_root: Path
    repo_root: Path
    transfer_data_root: Path
    smpl_model_root: Path
    pipeline: str
    force: bool
    frame_rate_hz: float
    num_betas: int
    frame_batch_size: int
    solver_init: str
    solver_maxiters: int
    freeze_betas_after_first_frame: bool
    fast_warm_start: bool
    warm_solver_maxiters: int
    verbose: bool
    keep_workdirs: bool
    limit_threads: bool


@dataclass(frozen=True)
class TakeRecord:
    source_path: Path
    take_name: str
    output_path: Path
    poses: np.ndarray
    betas: np.ndarray
    gender: str
    frame_rate_hz: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class InMemoryWorkerRuntime:
    device: Any
    def_matrix: Any
    mask_ids: Any
    transfer_cfg: dict[str, Any]
    source_model: Any
    target_models: dict[str, Any]
    target_assets: dict[str, Any]
    source_faces: np.ndarray
    source_faces_tensor: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CARE-PD canonicalized SMPL pickles to AMASS-like SMPL-X .npz files "
            "using the official vchoutas/smplx transfer_model pipeline."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Input folder with CARE-PD canonicalized SMPL .pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder for AMASS-like SMPL-X .npz files.",
    )
    parser.add_argument(
        "--smplx-repo-root",
        type=Path,
        default=DEFAULT_SMPLX_REPO_ROOT,
        help="Local checkout of https://github.com/vchoutas/smplx.",
    )
    parser.add_argument(
        "--transfer-data-root",
        type=Path,
        default=DEFAULT_TRANSFER_DATA_ROOT,
        help="Folder containing smpl2smplx_deftrafo_setup.pkl and smplx_mask_ids.npy.",
    )
    parser.add_argument(
        "--smpl-model-root",
        type=Path,
        default=DEFAULT_SMPL_MODEL_ROOT,
        help="Folder containing SMPL/SMPL-X model files from this repo.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Scan input root recursively (default: enabled).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan top-level .pkl files.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing .npz files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help=(
            "Number of parallel workers. For classic/in-memory this counts source files; "
            "for cpu-take it counts takes. Default: max(1, cpu_count/2)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan only. No files written.")
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=DEFAULT_FRAME_RATE_HZ,
        help="Fallback frame rate when source pickle has no fps metadata.",
    )
    parser.add_argument(
        "--num-betas",
        type=int,
        default=DEFAULT_NUM_BETAS,
        help="Length of output betas vector.",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=8,
        help=(
            "Number of frames to optimize together inside the in-memory GPU path. "
            "Use 1 for closest classic behavior; larger values are usually much faster."
        ),
    )
    parser.add_argument(
        "--solver-init",
        choices=("zeros", "source"),
        default="zeros",
        help=(
            "Initial guess for the solver. 'source' warm-starts from the input SMPL "
            "pose and shape, which can reduce iterations substantially."
        ),
    )
    parser.add_argument(
        "--solver-maxiters",
        type=int,
        default=100,
        help="Maximum number of iterations passed to the transfer solver.",
    )
    parser.add_argument(
        "--fast-warm-start",
        action="store_true",
        help=(
            "Use a cheaper solver profile for frames after the first one in cpu-take mode. "
            "This skips the edge-init stage and switches to LBFGS for warm frames."
        ),
    )
    parser.add_argument(
        "--warm-solver-maxiters",
        type=int,
        default=4,
        help=(
            "Outer iteration budget for the warm-start solver profile. Only used with "
            "--fast-warm-start."
        ),
    )
    parser.add_argument(
        "--freeze-betas-after-first-frame",
        action="store_true",
        help=(
            "Estimate betas on the first frame of each take, then keep them fixed "
            "for the rest of the sequence."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information while converting.",
    )
    parser.add_argument(
        "--keep-workdirs",
        action="store_true",
        help="Keep temp workdirs for debugging.",
    )
    parser.add_argument(
        "--summary-name",
        default="conversion_summary.json",
        help="JSON summary filename written inside output root.",
    )
    parser.add_argument(
        "--pipeline",
        choices=("in-memory", "classic", "cpu-take"),
        default="in-memory",
        help=(
            "Conversion backend. 'in-memory' avoids intermediate OBJ/pkl files and "
            "reuses loaded models; 'classic' keeps the original transfer_model CLI flow; "
            "'cpu-take' runs one warm-started CPU worker per take."
        ),
    )
    return parser.parse_args()


def _sanitize_component(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return cleaned or "take"


def _discover_pickles(input_root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.pkl" if recursive else "*.pkl"
    files = [path for path in input_root.glob(pattern) if path.is_file()]
    files.sort()
    return files


def _unwrap(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return _unwrap(value.item())
    return value


def _to_gender(value: Any) -> str:
    if value is None:
        return "neutral"
    text = str(_unwrap(value)).strip().lower()
    if text in {"m", "male"}:
        return "male"
    if text in {"f", "female"}:
        return "female"
    return "neutral" if not text else text


def _to_array(value: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    arr = np.asarray(_unwrap(value))
    if arr.dtype.kind in {"i", "u", "f", "b"}:
        arr = arr.astype(dtype, copy=False)
    return arr


def _ensure_2d(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D {name}, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _pad_or_truncate(array: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    if arr.shape[1] == width:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] > width:
        return arr[:, :width].astype(np.float32, copy=False)
    out = np.zeros((arr.shape[0], width), dtype=np.float32)
    out[:, : arr.shape[1]] = arr
    return out


def _normalize_betas(value: Any, num_betas: int) -> np.ndarray:
    if value is None:
        return np.zeros((num_betas,), dtype=np.float32)
    arr = np.asarray(_unwrap(value), dtype=np.float32).reshape(-1)
    if arr.size >= num_betas:
        return arr[:num_betas].astype(np.float32, copy=False)
    out = np.zeros((num_betas,), dtype=np.float32)
    out[: arr.size] = arr
    return out


def _looks_like_take_mapping(obj: Any) -> bool:
    if not isinstance(obj, Mapping):
        return False
    keys = {str(key).lower() for key in obj.keys()}
    return any(
        key in keys
        for key in (
            "trans",
            "transl",
            "translation",
            "poses",
            "pose",
            "body_pose",
            "root_orient",
            "global_orient",
        )
    )


def _collect_take_records(
    obj: Any,
    source_path: Path,
    path_parts: tuple[str, ...] = (),
) -> list[tuple[str, Mapping[str, Any]]]:
    obj = _unwrap(obj)

    if _looks_like_take_mapping(obj):
        if path_parts:
            take_name = "__".join((source_path.stem, *(_sanitize_component(part) for part in path_parts)))
        else:
            take_name = source_path.stem
        return [(take_name, obj)]

    records: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            records.extend(_collect_take_records(value, source_path, path_parts + (str(key),)))
        return records

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            records.extend(_collect_take_records(value, source_path, path_parts + (f"{idx}",)))
        return records

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.ndim == 0:
            return _collect_take_records(obj.item(), source_path, path_parts)
        for idx, value in enumerate(obj.tolist()):
            records.extend(_collect_take_records(value, source_path, path_parts + (f"{idx}",)))
        return records

    return []


def _iter_take_records(obj: Any, source_path: Path) -> list[tuple[str, Mapping[str, Any]]]:
    records = _collect_take_records(obj, source_path)
    if records:
        return records
    raise ValueError(f"Unsupported pickle layout in {source_path}")


def _extract_take_record(
    record: Mapping[str, Any],
    source_path: Path,
    take_name: str,
    fallback_frame_rate: float,
    num_betas: int,
) -> tuple[np.ndarray, np.ndarray, str, float, dict[str, Any]]:
    keys = {str(key).lower(): key for key in record.keys()}

    def get(*names: str) -> Any:
        for name in names:
            if name in keys:
                return record[keys[name]]
        return None

    gender = _to_gender(get("gender", "sex"))
    frame_rate_value = get("mocap_frame_rate", "mocap_framerate", "fps", "frame_rate")
    try:
        frame_rate_hz = float(fallback_frame_rate if frame_rate_value is None else _unwrap(frame_rate_value))
    except Exception:
        frame_rate_hz = float(fallback_frame_rate)
    betas = _normalize_betas(get("betas", "shape", "beta"), num_betas)

    trans_raw = get("trans", "transl", "translation")
    poses_raw = get("poses", "pose")
    root_orient_raw = get("root_orient", "global_orient")
    body_pose_raw = get("body_pose")

    diagnostics: dict[str, Any] = {
        "source_path": str(source_path),
        "take_name": take_name,
        "source_keys": sorted(str(key) for key in record.keys()),
    }

    if trans_raw is not None:
        trans = _ensure_2d(_to_array(trans_raw), "trans")
    elif poses_raw is not None:
        poses_arr = _ensure_2d(_to_array(poses_raw), "poses")
        trans = np.zeros((poses_arr.shape[0], 3), dtype=np.float32)
    elif root_orient_raw is not None:
        root_arr = _ensure_2d(_to_array(root_orient_raw), "root_orient")
        trans = np.zeros((root_arr.shape[0], 3), dtype=np.float32)
    else:
        raise ValueError(f"Missing trans/poses in {source_path} take {take_name}")

    n_frames = int(trans.shape[0])

    if root_orient_raw is not None and body_pose_raw is not None:
        root_orient = _pad_or_truncate(_ensure_2d(_to_array(root_orient_raw), "root_orient"), 3)
        pose_body = _ensure_2d(_to_array(body_pose_raw), "body_pose")
        diagnostics["pose_layout"] = "split"
    elif poses_raw is not None:
        poses = _ensure_2d(_to_array(poses_raw), "poses")
        if poses.shape[0] != n_frames:
            if n_frames == 1:
                trans = np.repeat(trans, poses.shape[0], axis=0)
                n_frames = int(trans.shape[0])
            else:
                raise ValueError(
                    f"Frame mismatch in {source_path} take {take_name}: trans {trans.shape}, poses {poses.shape}"
                )
        if poses.shape[1] < 66:
            raise ValueError(f"Expected poses with at least 66 columns, got {poses.shape}")
        root_orient = poses[:, :3].astype(np.float32, copy=False)
        pose_body = poses[:, 3:66].astype(np.float32, copy=False)
        diagnostics["pose_layout"] = f"packed_{poses.shape[1]}"
        if poses.shape[1] > 66:
            diagnostics["ignored_pose_dims"] = int(poses.shape[1] - 66)
    else:
        raise ValueError(f"Missing poses/body_pose in {source_path} take {take_name}")

    if root_orient.shape[0] != n_frames:
        if root_orient.shape[0] == 1:
            root_orient = np.repeat(root_orient, n_frames, axis=0)
        else:
            raise ValueError(f"Frame mismatch for root_orient in {source_path} take {take_name}")
    if pose_body.shape[0] != n_frames:
        if pose_body.shape[0] == 1:
            pose_body = np.repeat(pose_body, n_frames, axis=0)
        else:
            raise ValueError(f"Frame mismatch for body_pose in {source_path} take {take_name}")

    poses = np.concatenate([root_orient, pose_body], axis=1).astype(np.float32, copy=False)
    diagnostics["n_frames"] = int(n_frames)
    diagnostics["source_frame_rate_hz"] = float(frame_rate_hz)
    return poses, betas, gender, frame_rate_hz, diagnostics


def _build_output_path(output_root: Path, source_root: Path, source_path: Path, take_name: str) -> Path:
    relative_parent = source_path.relative_to(source_root).parent
    filename = f"{_sanitize_component(take_name)}.npz"
    return (output_root / relative_parent / filename).resolve()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _verbose_print(config: ConversionConfig, message: str) -> None:
    if config.verbose:
        print(message, flush=True)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _progress_message(completed: int, total: int, elapsed_seconds: float) -> str:
    total = max(1, int(total))
    completed = min(max(0, int(completed)), total)
    pct = 100.0 * completed / total
    if completed <= 0:
        eta_text = "ETA --"
    elif completed >= total:
        eta_text = "ETA 0:00"
    else:
        eta_seconds = elapsed_seconds * (total - completed) / completed
        eta_text = f"ETA {_format_duration(eta_seconds)}"
    return f"{pct:5.1f}% | {eta_text} | elapsed {_format_duration(elapsed_seconds)}"


def _cached_betas_from_output(var_dict: Mapping[str, Any]) -> Any:
    betas = var_dict.get("betas")
    if betas is None:
        return None

    import torch

    betas_t = betas.detach()
    if betas_t.ndim == 1:
        return betas_t.reshape(1, -1)
    if betas_t.ndim >= 2:
        return betas_t.mean(dim=0, keepdim=True)
    if torch.is_tensor(betas_t):
        return betas_t.reshape(1, -1)
    return None


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _append_frame_outputs(
    frame_outputs: list[Mapping[str, Any]],
    var_dict: Mapping[str, Any],
    batch_size: int,
) -> None:
    import torch

    for frame_idx in range(batch_size):
        frame_output: dict[str, Any] = {}
        for key in _IN_MEMORY_MERGE_KEYS:
            value = var_dict.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                frame_output[key] = value[frame_idx : frame_idx + 1].detach().cpu()
            else:
                arr = np.asarray(value)
                frame_output[key] = torch.as_tensor(arr[frame_idx : frame_idx + 1]).cpu()
        frame_outputs.append(frame_output)


def _initial_values_from_previous_output(
    var_dict: Mapping[str, Any],
    device: Any,
    betas_value: Any | None = None,
) -> dict[str, Any]:
    import torch

    initial_values: dict[str, Any] = {}
    pose_keys = (
        "global_orient",
        "body_pose",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
    )

    for key in ("transl", "expression", *pose_keys):
        value = var_dict.get(key)
        if value is None:
            continue
        arr = _tensor_to_numpy(value)
        if key in pose_keys:
            arr = _rotmat_to_rotvec(arr)
        initial_values[key] = torch.as_tensor(arr, dtype=torch.float32, device=device)

    if betas_value is not None:
        initial_values["betas"] = betas_value
    else:
        betas = var_dict.get("betas")
        if betas is not None:
            initial_values["betas"] = torch.as_tensor(
                _tensor_to_numpy(betas)[:1], dtype=torch.float32, device=device
            )

    return initial_values


def _configure_worker_runtime(limit_threads: bool) -> None:
    if not limit_threads:
        return

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")

    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass


def _resolve_repo_root(path: Path) -> Path:
    repo_root = path.expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"SMPL-X repo root not found: {repo_root}")
    if not (repo_root / "transfer_model" / "__main__.py").exists():
        raise FileNotFoundError(
            f"Invalid SMPL-X repo root: {repo_root}. Missing transfer_model/__main__.py"
        )
    return repo_root


def _resolve_transfer_data_root(path: Path) -> Path:
    transfer_root = path.expanduser().resolve()
    if not transfer_root.exists():
        raise FileNotFoundError(
            f"Transfer-data root not found: {transfer_root}. "
            "Need smpl2smplx_deftrafo_setup.pkl and smplx_mask_ids.npy from SMPL-X correspondences."
        )
    required = [
        transfer_root / "smpl2smplx_deftrafo_setup.pkl",
        transfer_root / "smplx_mask_ids.npy",
    ]
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise FileNotFoundError("Missing transfer-data files:\n- " + "\n- ".join(missing))
    return transfer_root


def _resolve_smpl_model_root(path: Path) -> Path:
    model_root = path.expanduser().resolve()
    if not model_root.exists():
        raise FileNotFoundError(f"SMPL model root not found: {model_root}")
    if not (model_root / "SMPL_NEUTRAL.pkl").exists():
        raise FileNotFoundError(
            f"Expected SMPL model files in {model_root}. Missing SMPL_NEUTRAL.pkl."
        )
    return model_root


def _resolve_model_gender(model_family_root: Path, model_prefix: str, gender: str) -> str:
    requested = (gender or "neutral").strip().lower()
    if requested not in {"neutral", "male", "female"}:
        requested = "neutral"
    candidate = model_family_root / f"{model_prefix}_{requested.upper()}.{'pkl' if model_prefix == 'SMPL' else 'npz'}"
    neutral = model_family_root / f"{model_prefix}_NEUTRAL.{'pkl' if model_prefix == 'SMPL' else 'npz'}"
    if candidate.exists():
        return requested
    if neutral.exists():
        return "neutral"
    return requested


def _prepare_official_models_root(workdir: Path, smpl_model_root: Path) -> Path:
    models_root = workdir / "models"
    _ensure_dir(models_root)
    for subdir in ("smpl", "smplx"):
        target = models_root / subdir
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(smpl_model_root, target_is_directory=True)
        except OSError:
            shutil.copytree(smpl_model_root, target)
    return models_root


def _build_subprocess_env(limit_threads: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if limit_threads:
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env.setdefault(key, "1")
    return env


def _runtime_device(config: ConversionConfig):
    import torch

    if config.pipeline == "cpu-take":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _transfer_exp_cfg(
    solver_maxiters: int,
    *,
    optim_type: str = "trust-ncg",
    gtol: float = 1e-6,
    ftol: float = -1.0,
    lbfgs_max_iter: int | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "summary_steps": 100,
        "interactive": False,
        "optim": {
            "type": optim_type,
            "maxiters": int(solver_maxiters),
            "gtol": float(gtol),
            "ftol": float(ftol),
        },
        "edge_fitting": {
            "per_part": False,
            "reduction": "sum",
        },
        "vertex_fitting": {
            "reduction": "sum",
        },
    }
    if optim_type in {"lbfgs", "lbfgsls"}:
        cfg["optim"]["lbfgs"] = {
            "lr": 1.0,
            "max_iter": int(lbfgs_max_iter or max(1, int(solver_maxiters))),
            "line_search_fn": "strong_wolfe",
        }
    return cfg


def _model_file(root: Path, prefix: str, gender: str, ext: str) -> Path:
    requested = _resolve_model_gender(root, prefix, gender)
    return root / f"{prefix}_{requested.upper()}.{ext}"


def _runtime_cache_key(config: ConversionConfig) -> tuple[str, ...]:
    return (
        str(config.input_root),
        str(config.output_root),
        str(config.repo_root),
        str(config.transfer_data_root),
        str(config.smpl_model_root),
        str(config.pipeline),
        str(config.force),
        str(config.frame_rate_hz),
        str(config.num_betas),
        str(config.frame_batch_size),
        str(config.solver_init),
        str(config.solver_maxiters),
        str(config.freeze_betas_after_first_frame),
        str(config.verbose),
        str(config.keep_workdirs),
        str(config.limit_threads),
    )


def _build_in_memory_runtime(config: ConversionConfig) -> InMemoryWorkerRuntime:
    import torch
    import smplx
    from smplx import build_layer

    device = _runtime_device(config)
    if config.limit_threads:
        _configure_worker_runtime(True)

    _verbose_print(
        config,
        f"[runtime] initializing worker on {device} (limit_threads={config.limit_threads})",
    )
    _verbose_print(config, f"[runtime] loading transfer data from {config.transfer_data_root}")
    def_matrix = read_deformation_transfer(
        str(config.transfer_data_root / "smpl2smplx_deftrafo_setup.pkl"),
        device=device,
    )
    mask_ids = np.load(config.transfer_data_root / "smplx_mask_ids.npy")
    mask_ids = torch.as_tensor(mask_ids, dtype=torch.long, device=device)
    _verbose_print(
        config,
        f"[runtime] transfer matrix and mask ids ready (mask_ids={int(mask_ids.numel())})",
    )

    source_gender = _resolve_model_gender(config.smpl_model_root, "SMPL", "neutral")
    source_model_path = _model_file(config.smpl_model_root, "SMPL", source_gender, "pkl")
    _verbose_print(config, f"[runtime] loading source SMPL model from {source_model_path}")
    source_model = smplx.create(
        model_path=str(source_model_path),
        model_type="smpl",
        use_pca=False,
        batch_size=1,
        gender=source_gender,
        ext="pkl",
        num_betas=max(1, int(config.num_betas)),
    ).to(device=device)
    source_model.eval()
    source_faces = np.asarray(source_model.faces, dtype=np.int32)
    source_faces_tensor = torch.as_tensor(source_faces, dtype=torch.long, device=device)
    _verbose_print(
        config,
        f"[runtime] source SMPL model ready (num_betas={int(source_model.num_betas)}, faces={source_faces.shape[0]})",
    )

    runtime = InMemoryWorkerRuntime(
        device=device,
        def_matrix=def_matrix,
        mask_ids=mask_ids,
        transfer_cfg=_transfer_exp_cfg(config.solver_maxiters),
        source_model=source_model,
        target_models={},
        target_assets={},
        source_faces=source_faces,
        source_faces_tensor=source_faces_tensor,
    )
    return runtime


def _get_worker_runtime(config: ConversionConfig) -> InMemoryWorkerRuntime:
    key = _runtime_cache_key(config)
    runtime = _IN_MEMORY_RUNTIME_CACHE.get(key)
    if runtime is None:
        _verbose_print(config, "[runtime] cache miss, building in-memory worker state")
        runtime = _build_in_memory_runtime(config)
        _IN_MEMORY_RUNTIME_CACHE[key] = runtime
    return runtime


def _get_target_model_and_assets(
    runtime: InMemoryWorkerRuntime,
    config: ConversionConfig,
    gender: str,
):
    import torch
    from smplx import build_layer

    resolved_gender = _resolve_model_gender(config.smpl_model_root, "SMPLX", gender)
    model = runtime.target_models.get(resolved_gender)
    if model is None:
        model_path = _model_file(config.smpl_model_root, "SMPLX", resolved_gender, "npz")
        _verbose_print(
            config,
            f"[runtime] loading SMPL-X target model for gender={resolved_gender} from {model_path}",
        )
        model = build_layer(
            str(model_path),
            gender=resolved_gender,
            use_compressed=False,
            use_face_contour=True,
            num_betas=TRANSFER_MODEL_NUM_BETAS,
            num_expression_coeffs=TRANSFER_MODEL_NUM_EXPRESSION,
            batch_size=1,
            ext="npz",
        ).to(device=runtime.device)
        model.eval()
        runtime.target_models[resolved_gender] = model

    assets = runtime.target_assets.get(resolved_gender)
    if assets is None:
        _verbose_print(config, f"[runtime] preparing reusable fitting assets for gender={resolved_gender}")
        assets = prepare_fitting_assets(
            runtime.transfer_cfg,
            model,
            runtime.def_matrix,
            mask_ids=runtime.mask_ids,
        )
        runtime.target_assets[resolved_gender] = assets
    return model, assets, resolved_gender


def _write_motion_npz(path: Path, poses: np.ndarray, betas: np.ndarray, gender: str, frame_rate_hz: float) -> None:
    _ensure_dir(path.parent)
    np.savez_compressed(
        path,
        poses=poses.astype(np.float32, copy=False),
        betas=betas.astype(np.float32, copy=False),
        gender=np.array(gender),
        mocap_frame_rate=np.array(float(frame_rate_hz), dtype=np.float32),
    )


def _write_smpl_meshes(
    output_folder: Path,
    poses: np.ndarray,
    betas: np.ndarray,
    gender: str,
    model_root: Path,
) -> None:
    import smplx
    import torch

    _ensure_dir(output_folder)
    source_gender = _resolve_model_gender(model_root / "smpl", "SMPL", gender)
    model = smplx.create(
        model_path=str(model_root),
        model_type="smpl",
        gender=source_gender,
        ext="pkl",
        use_pca=False,
        batch_size=int(poses.shape[0]),
        num_betas=int(max(1, betas.size)),
    )
    model = model.to(device=torch.device("cpu"))

    poses = np.asarray(poses, dtype=np.float32)
    if poses.shape[1] < 72:
        poses = np.pad(poses, ((0, 0), (0, 72 - poses.shape[1])), mode="constant")
    elif poses.shape[1] > 72:
        poses = poses[:, :72]

    betas_vec = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas_vec.size == 0:
        betas_vec = np.zeros((model.num_betas,), dtype=np.float32)
    if betas_vec.size < model.num_betas:
        betas_vec = np.pad(betas_vec, (0, model.num_betas - betas_vec.size))
    betas_t = torch.as_tensor(betas_vec[: model.num_betas], dtype=torch.float32).reshape(1, -1)
    betas_t = betas_t.repeat(int(poses.shape[0]), 1)

    root_orient = torch.as_tensor(poses[:, :3], dtype=torch.float32)
    body_pose = torch.as_tensor(poses[:, 3:], dtype=torch.float32)
    transl = torch.zeros((poses.shape[0], 3), dtype=torch.float32)

    with torch.no_grad():
        output = model(
            betas=betas_t,
            global_orient=root_orient,
            body_pose=body_pose,
            transl=transl,
            return_verts=True,
        )
        vertices = output.vertices.detach().cpu().numpy()

    faces = np.asarray(model.faces, dtype=np.int32)
    for idx in range(vertices.shape[0]):
        obj_path = output_folder / f"{idx:04d}.obj"
        with obj_path.open("w", encoding="utf-8") as handle:
            for vx, vy, vz in vertices[idx]:
                handle.write(f"v {vx:.9f} {vy:.9f} {vz:.9f}\n")
            for fa, fb, fc in faces + 1:
                handle.write(f"f {int(fa)} {int(fb)} {int(fc)}\n")


def _write_transfer_config(
    path: Path,
    mesh_folder: Path,
    transfer_data_root: Path,
    models_root: Path,
    gender: str,
) -> None:
    yaml_text = f"""datasets:
    mesh_folder:
        data_folder: '{mesh_folder.as_posix()}'
deformation_transfer_path: '{(transfer_data_root / "smpl2smplx_deftrafo_setup.pkl").as_posix()}'
mask_ids_fname: '{(transfer_data_root / "smplx_mask_ids.npy").as_posix()}'
summary_steps: 100

edge_fitting:
    per_part: False

optim:
    type: 'trust-ncg'
    maxiters: 100
    gtol: 1e-06

body_model:
    model_type: "smplx"
    gender: "{gender}"
    folder: "{models_root.as_posix()}"
    use_compressed: False
    use_face_contour: True
    smplx:
        betas:
            num: 10
        expression:
            num: 10
output_folder: "{(path.parent / 'transfer_output').as_posix()}"
"""
    path.write_text(yaml_text, encoding="utf-8")


def _run_transfer_model(repo_root: Path, config_path: Path, limit_threads: bool) -> None:
    cmd = [
        sys.executable,
        "-m",
        "transfer_model",
        "--exp-cfg",
        str(config_path),
    ]
    env = _build_subprocess_env(limit_threads)
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def _run_merge_output(repo_root: Path, transfer_output: Path, gender: str, limit_threads: bool) -> Path:
    cmd = [
        sys.executable,
        str(repo_root / "transfer_model" / "merge_output.py"),
        str(transfer_output),
        "--gender",
        gender,
    ]
    env = _build_subprocess_env(limit_threads)
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)
    merged_path = transfer_output / "merged.pkl"
    if not merged_path.exists():
        raise FileNotFoundError(f"Expected merged.pkl at {merged_path}")
    return merged_path


def _rotmat_to_rotvec(array: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as R

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float32, copy=False)
    if arr.ndim >= 2 and arr.shape[-2:] == (3, 3):
        flat = arr.reshape(-1, 3, 3)
        rotvec = R.from_matrix(flat.astype(np.float64)).as_rotvec().astype(np.float32)
        return rotvec.reshape(*arr.shape[:-2], 3)
    if arr.ndim >= 2 and arr.shape[-1] == 3 and arr.shape[-2] == 1:
        return arr.reshape(*arr.shape[:-2], 3).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported rotation array shape: {arr.shape}")


def _normalize_axis_angle_sequence(array: Any, expected_dim: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, expected_dim), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == expected_dim:
        return arr.astype(np.float32, copy=False)
    if arr.ndim >= 3 and arr.shape[-2:] == (3, 3):
        rotvec = _rotmat_to_rotvec(arr)
        if rotvec.ndim == 1:
            rotvec = rotvec.reshape(-1, expected_dim)
        elif rotvec.ndim > 2 and int(np.prod(rotvec.shape[1:])) == expected_dim:
            rotvec = rotvec.reshape(rotvec.shape[0], expected_dim)
        return rotvec.astype(np.float32, copy=False)
    if arr.ndim >= 2 and arr.shape[-1] == 3 and int(np.prod(arr.shape[1:])) == expected_dim:
        return arr.reshape(arr.shape[0], expected_dim).astype(np.float32, copy=False)
    if arr.ndim == 1 and arr.shape[0] == expected_dim:
        return arr.reshape(1, expected_dim).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported pose shape {arr.shape} for expected dim {expected_dim}")


def _normalize_translation_sequence(array: Any) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 3:
        return arr[:, 0].astype(np.float32, copy=False)
    if arr.ndim == 1 and arr.shape[0] == 3:
        return arr.reshape(1, 3).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported transl shape {arr.shape}")


def _source_vertices_for_take(
    runtime: InMemoryWorkerRuntime,
    poses: np.ndarray,
    betas: np.ndarray,
) -> Any:
    import torch

    pose_array = np.asarray(poses, dtype=np.float32)
    if pose_array.ndim != 2:
        raise ValueError(f"Expected 2D pose array, got shape {pose_array.shape}")
    if pose_array.shape[1] < 72:
        pose_array = np.pad(pose_array, ((0, 0), (0, 72 - pose_array.shape[1])), mode="constant")
    elif pose_array.shape[1] > 72:
        pose_array = pose_array[:, :72]

    betas_vec = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas_vec.size == 0:
        betas_vec = np.zeros((runtime.source_model.num_betas,), dtype=np.float32)
    if betas_vec.size < runtime.source_model.num_betas:
        betas_vec = np.pad(betas_vec, (0, runtime.source_model.num_betas - betas_vec.size))
    elif betas_vec.size > runtime.source_model.num_betas:
        betas_vec = betas_vec[: runtime.source_model.num_betas]

    betas_t = torch.as_tensor(
        betas_vec[: runtime.source_model.num_betas],
        dtype=torch.float32,
        device=runtime.device,
    ).reshape(1, -1)
    betas_t = betas_t.repeat(int(pose_array.shape[0]), 1)

    root_orient = torch.as_tensor(pose_array[:, :3], dtype=torch.float32, device=runtime.device)
    body_pose = torch.as_tensor(pose_array[:, 3:], dtype=torch.float32, device=runtime.device)
    transl = torch.zeros((pose_array.shape[0], 3), dtype=torch.float32, device=runtime.device)

    with torch.no_grad():
        output = runtime.source_model(
            betas=betas_t,
            global_orient=root_orient,
            body_pose=body_pose,
            transl=transl,
            return_verts=True,
        )
    return output.vertices.detach()


def _aggregate_in_memory_outputs(
    frame_outputs: list[Mapping[str, Any]],
    gender: str,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in _IN_MEMORY_MERGE_KEYS:
        values = [frame[key] for frame in frame_outputs if key in frame]
        if not values:
            continue
        merged[key] = transfer_merge_output.aggregate_function[key](values)
    merged["gender"] = gender
    return merged


def _merged_dict_to_amass_npz(
    merged: Mapping[str, Any],
    output_path: Path,
    source_path: Path,
    take_name: str,
    frame_rate_hz: float,
    pipeline: str,
    frame_batch_size: int,
) -> None:
    if merged.get("transl") is None:
        raise KeyError(f"Missing transl for {output_path}")
    if merged.get("global_orient") is None:
        raise KeyError(f"Missing global_orient for {output_path}")
    if merged.get("body_pose") is None:
        raise KeyError(f"Missing body_pose for {output_path}")

    transl = _normalize_translation_sequence(merged.get("transl"))
    root_orient = _normalize_axis_angle_sequence(merged.get("global_orient"), 3)
    body_pose = _normalize_axis_angle_sequence(merged.get("body_pose"), 63)

    left_hand = merged.get("left_hand_pose")
    right_hand = merged.get("right_hand_pose")
    if left_hand is None:
        left_hand = np.zeros((transl.shape[0], 45), dtype=np.float32)
    else:
        left_hand = _normalize_axis_angle_sequence(left_hand, 45)
    if right_hand is None:
        right_hand = np.zeros((transl.shape[0], 45), dtype=np.float32)
    else:
        right_hand = _normalize_axis_angle_sequence(right_hand, 45)

    jaw_pose = merged.get("jaw_pose")
    if jaw_pose is None:
        jaw_pose = np.zeros((transl.shape[0], 3), dtype=np.float32)
    else:
        jaw_pose = _normalize_axis_angle_sequence(jaw_pose, 3)

    leye_pose = merged.get("leye_pose")
    if leye_pose is None:
        leye_pose = np.zeros((transl.shape[0], 3), dtype=np.float32)
    else:
        leye_pose = _normalize_axis_angle_sequence(leye_pose, 3)

    reye_pose = merged.get("reye_pose")
    if reye_pose is None:
        reye_pose = np.zeros((transl.shape[0], 3), dtype=np.float32)
    else:
        reye_pose = _normalize_axis_angle_sequence(reye_pose, 3)

    betas_value = merged.get("betas")
    if betas_value is None:
        betas = np.zeros((DEFAULT_NUM_BETAS,), dtype=np.float32)
    else:
        betas = np.asarray(betas_value, dtype=np.float32)
        if betas.ndim > 1:
            betas = betas.mean(axis=0)
        if betas.size == 0:
            betas = np.zeros((DEFAULT_NUM_BETAS,), dtype=np.float32)

    gender = str(merged.get("gender", "neutral")).lower()
    n_frames = int(transl.shape[0])
    _ensure_dir(output_path.parent)
    np.savez_compressed(
        output_path,
        surface_model_type=np.array("smplx"),
        gender=np.array(gender),
        mocap_frame_rate=np.array(float(frame_rate_hz), dtype=np.float32),
        trans=transl.astype(np.float32, copy=False),
        root_orient=root_orient.astype(np.float32, copy=False),
        pose_body=body_pose.astype(np.float32, copy=False),
        pose_hand=np.concatenate([left_hand, right_hand], axis=1).astype(np.float32, copy=False),
        pose_jaw=jaw_pose.astype(np.float32, copy=False).reshape(n_frames, 3),
        pose_eye=np.concatenate([leye_pose, reye_pose], axis=1).astype(np.float32, copy=False).reshape(n_frames, 6),
        betas=betas.astype(np.float32, copy=False),
        carepd_source_path=np.array(str(source_path)),
        carepd_take_name=np.array(take_name),
        carepd_conversion_diagnostics=np.array(
            json.dumps(
                {
                    "merged_keys": sorted(str(key) for key in merged.keys()),
                    "pipeline": pipeline,
                    "frame_batch_size": int(frame_batch_size),
                    "source_path": str(source_path),
                    "take_name": take_name,
                    "frame_rate_hz": float(frame_rate_hz),
                },
                sort_keys=True,
            )
        ),
    )


def _merged_pkl_to_amass_npz(
    merged_path: Path,
    output_path: Path,
    source_path: Path,
    take_name: str,
    frame_rate_hz: float,
) -> None:
    with merged_path.open("rb") as handle:
        merged = pickle.load(handle)
    _merged_dict_to_amass_npz(
        merged=merged,
        output_path=output_path,
        source_path=source_path,
        take_name=take_name,
        frame_rate_hz=frame_rate_hz,
        pipeline="classic",
        frame_batch_size=1,
    )


def _convert_one_take_in_memory(
    take: TakeRecord,
    config: ConversionConfig,
) -> dict[str, Any]:
    from time import perf_counter

    t_take = perf_counter()
    runtime = _get_worker_runtime(config)
    target_model, prepared_assets, _ = _get_target_model_and_assets(runtime, config, take.gender)
    frame_batch_size = max(1, int(config.frame_batch_size))
    total_frames = int(take.poses.shape[0])
    freeze_betas = bool(config.freeze_betas_after_first_frame)
    if freeze_betas and total_frames > 0:
        num_chunks = 1 + ((max(0, total_frames - 1) + frame_batch_size - 1) // frame_batch_size)
    else:
        num_chunks = max(1, (total_frames + frame_batch_size - 1) // frame_batch_size)
    _verbose_print(
        config,
        f"[{take.take_name}] {_progress_message(0, total_frames, 0.0)} | "
        f"start in-memory conversion: frames={total_frames}, batch={frame_batch_size}, "
        f"gender={take.gender}, solver_init={config.solver_init}, solver_maxiters={config.solver_maxiters}, "
        f"shared_betas=True, freeze_betas_after_first_frame={freeze_betas}",
    )
    _verbose_print(config, f"[{take.take_name}] computing source vertices on {runtime.device}")
    source_vertices = _source_vertices_for_take(runtime, take.poses, take.betas)
    _verbose_print(config, f"[{take.take_name}] source vertices ready after {perf_counter() - t_take:.1f}s")

    import torch

    frame_outputs: list[Mapping[str, Any]] = []
    target_num_betas = int(getattr(target_model, "num_betas", max(1, np.asarray(take.betas).reshape(-1).size)))
    source_betas = torch.as_tensor(
        _normalize_betas(take.betas, target_num_betas).reshape(1, -1),
        dtype=torch.float32,
        device=runtime.device,
    )
    fixed_betas = None

    def _build_initial_values(start: int, stop: int, betas_value: Any | None) -> dict[str, Any] | None:
        initial_values: dict[str, Any] = {}
        if config.solver_init == "source":
            source_pose = np.asarray(take.poses[start:stop], dtype=np.float32)
            batch_size = int(stop - start)
            initial_values["transl"] = torch.zeros((batch_size, 3), dtype=torch.float32, device=runtime.device)
            initial_values["global_orient"] = torch.as_tensor(
                source_pose[:, :3].reshape(batch_size, 1, 3),
                dtype=torch.float32,
                device=runtime.device,
            )
            initial_values["body_pose"] = torch.as_tensor(
                source_pose[:, 3:66].reshape(batch_size, -1, 3),
                dtype=torch.float32,
                device=runtime.device,
            )
        if betas_value is not None:
            initial_values["betas"] = betas_value
        elif config.solver_init == "source":
            initial_values["betas"] = source_betas
        return initial_values or None

    def _append_frame_outputs(var_dict: Mapping[str, Any], batch_size: int) -> None:
        for frame_idx in range(batch_size):
            frame_output: dict[str, Any] = {}
            for key in _IN_MEMORY_MERGE_KEYS:
                value = var_dict.get(key)
                if value is None:
                    continue
                if torch.is_tensor(value):
                    frame_output[key] = value[frame_idx : frame_idx + 1].detach().cpu()
                else:
                    arr = np.asarray(value)
                    frame_output[key] = torch.as_tensor(arr[frame_idx : frame_idx + 1]).cpu()
            frame_outputs.append(frame_output)

    chunk_index = 0
    start_frame = 0

    if freeze_betas and total_frames > 0:
        chunk_index += 1
        chunk_t0 = perf_counter()
        bootstrap_batch = {
            "vertices": source_vertices[:1].to(device=runtime.device, dtype=torch.float32),
            "faces": runtime.source_faces_tensor,
        }
        bootstrap_initial_values = _build_initial_values(0, 1, source_betas)
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(0, total_frames, perf_counter() - t_take)} | "
            f"chunk {chunk_index}/{num_chunks}: bootstrap beta from frame 1",
        )
        bootstrap_var_dict = run_fitting(
            runtime.transfer_cfg,
            bootstrap_batch,
            target_model,
            runtime.def_matrix,
            mask_ids=runtime.mask_ids,
            prepared_assets=prepared_assets,
            initial_values=bootstrap_initial_values,
            shared_betas=True,
            freeze_betas=False,
        )
        fixed_betas = _cached_betas_from_output(bootstrap_var_dict)
        if fixed_betas is None:
            raise RuntimeError(f"Failed to estimate fixed betas from first frame of {take.take_name}")
        _append_frame_outputs(bootstrap_var_dict, 1)
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(1, total_frames, perf_counter() - t_take)} | "
            f"chunk {chunk_index}/{num_chunks} done in {perf_counter() - chunk_t0:.1f}s",
        )
        start_frame = 1

    cached_betas = fixed_betas
    for start in range(start_frame, total_frames, frame_batch_size):
        stop = min(start + frame_batch_size, total_frames)
        chunk_t0 = perf_counter()
        chunk_index += 1
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(start, total_frames, perf_counter() - t_take)} | "
            f"chunk {chunk_index}/{num_chunks}: frames {start + 1}-{stop} of {total_frames}",
        )
        batch_vertices = source_vertices[start:stop].to(device=runtime.device, dtype=torch.float32)
        batch_size = int(batch_vertices.shape[0])
        batch = {
            "vertices": batch_vertices,
            "faces": runtime.source_faces_tensor,
        }
        initial_values = _build_initial_values(start, stop, cached_betas)
        var_dict = run_fitting(
            runtime.transfer_cfg,
            batch,
            target_model,
            runtime.def_matrix,
            mask_ids=runtime.mask_ids,
            prepared_assets=prepared_assets,
            initial_values=initial_values,
            shared_betas=True,
            freeze_betas=freeze_betas,
        )
        if not freeze_betas:
            cached_betas = _cached_betas_from_output(var_dict)
        _append_frame_outputs(var_dict, batch_size)
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(stop, total_frames, perf_counter() - t_take)} | "
            f"chunk {chunk_index}/{num_chunks} done in {perf_counter() - chunk_t0:.1f}s",
        )

    _verbose_print(config, f"[{take.take_name}] aggregating {len(frame_outputs)} frame outputs")
    merged = _aggregate_in_memory_outputs(frame_outputs, take.gender)
    _verbose_print(config, f"[{take.take_name}] writing {take.output_path}")
    _merged_dict_to_amass_npz(
        merged=merged,
        output_path=take.output_path,
        source_path=take.source_path,
        take_name=take.take_name,
        frame_rate_hz=take.frame_rate_hz,
        pipeline="in-memory",
        frame_batch_size=max(1, int(config.frame_batch_size)),
    )
    _verbose_print(
        config,
        f"[{take.take_name}] {_progress_message(total_frames, total_frames, perf_counter() - t_take)} | written to {take.output_path}",
    )
    return {
        "source_path": str(take.source_path),
        "take_name": take.take_name,
        "output_path": str(take.output_path),
        "n_frames": int(take.poses.shape[0]),
        "gender": take.gender,
        "frame_rate_hz": float(take.frame_rate_hz),
        "status": "written",
    }


def _convert_one_take_cpu_takes(
    take: TakeRecord,
    config: ConversionConfig,
) -> dict[str, Any]:
    from time import perf_counter

    t_take = perf_counter()
    runtime = _get_worker_runtime(config)
    target_model, prepared_assets, _ = _get_target_model_and_assets(runtime, config, take.gender)
    total_frames = int(take.poses.shape[0])
    _verbose_print(
        config,
        f"[{take.take_name}] {_progress_message(0, total_frames, 0.0)} | "
        f"start cpu-take conversion: frames={total_frames}, warm_start=previous_frame, "
        f"freeze_betas_after_first_frame={config.freeze_betas_after_first_frame}, "
        f"fast_warm_start={config.fast_warm_start}, warm_solver_maxiters={config.warm_solver_maxiters}",
    )
    _verbose_print(config, f"[{take.take_name}] computing source vertices on {runtime.device}")
    source_vertices = _source_vertices_for_take(runtime, take.poses, take.betas)
    _verbose_print(config, f"[{take.take_name}] source vertices ready after {perf_counter() - t_take:.1f}s")

    import torch

    frame_outputs: list[Mapping[str, Any]] = []
    target_num_betas = int(getattr(target_model, "num_betas", max(1, np.asarray(take.betas).reshape(-1).size)))
    source_betas = torch.as_tensor(
        _normalize_betas(take.betas, target_num_betas).reshape(1, -1),
        dtype=torch.float32,
        device=runtime.device,
    )
    freeze_betas = bool(config.freeze_betas_after_first_frame)
    fixed_betas = None
    previous_seed: dict[str, Any] | None = None
    warm_transfer_cfg = None
    if config.fast_warm_start:
        warm_transfer_cfg = _transfer_exp_cfg(
            max(1, int(config.warm_solver_maxiters)),
            optim_type="lbfgsls",
            gtol=1e-4,
            ftol=1e-5,
            lbfgs_max_iter=20,
        )

    def _build_source_initial_values() -> dict[str, Any] | None:
        initial_values: dict[str, Any] = {}
        if config.solver_init == "source":
            source_pose = np.asarray(take.poses[:1], dtype=np.float32)
            initial_values["transl"] = torch.zeros((1, 3), dtype=torch.float32, device=runtime.device)
            initial_values["global_orient"] = torch.as_tensor(
                source_pose[:, :3].reshape(1, 1, 3),
                dtype=torch.float32,
                device=runtime.device,
            )
            initial_values["body_pose"] = torch.as_tensor(
                source_pose[:, 3:66].reshape(1, -1, 3),
                dtype=torch.float32,
                device=runtime.device,
            )
        initial_values["betas"] = source_betas
        return initial_values or None

    for frame_idx in range(total_frames):
        frame_t0 = perf_counter()
        batch = {
            "vertices": source_vertices[frame_idx : frame_idx + 1].to(device=runtime.device, dtype=torch.float32),
            "faces": runtime.source_faces_tensor,
        }

        if frame_idx == 0:
            initial_values = _build_source_initial_values()
            var_dict = run_fitting(
                runtime.transfer_cfg,
                batch,
                target_model,
                runtime.def_matrix,
                mask_ids=runtime.mask_ids,
                prepared_assets=prepared_assets,
                initial_values=initial_values,
                shared_betas=True,
                freeze_betas=False,
                skip_edge_init=False,
            )
            fixed_betas = _cached_betas_from_output(var_dict)
            if fixed_betas is None:
                raise RuntimeError(f"Failed to estimate fixed betas from first frame of {take.take_name}")
            previous_seed = _initial_values_from_previous_output(
                var_dict,
                runtime.device,
                betas_value=fixed_betas,
            )
        else:
            if previous_seed is None:
                raise RuntimeError(f"Missing warm-start seed for frame {frame_idx + 1} of {take.take_name}")
            initial_values = dict(previous_seed)
            if freeze_betas:
                if fixed_betas is None:
                    raise RuntimeError(f"Missing fixed betas for frame {frame_idx + 1} of {take.take_name}")
                initial_values["betas"] = fixed_betas
            current_transfer_cfg = warm_transfer_cfg or runtime.transfer_cfg
            var_dict = run_fitting(
                current_transfer_cfg,
                batch,
                target_model,
                runtime.def_matrix,
                mask_ids=runtime.mask_ids,
                prepared_assets=prepared_assets,
                initial_values=initial_values,
                shared_betas=True,
                freeze_betas=freeze_betas,
                skip_edge_init=bool(config.fast_warm_start),
            )
            if not freeze_betas:
                fixed_betas = _cached_betas_from_output(var_dict)
            previous_seed = _initial_values_from_previous_output(
                var_dict,
                runtime.device,
                betas_value=fixed_betas,
            )

        _append_frame_outputs(frame_outputs, var_dict, 1)
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(frame_idx + 1, total_frames, perf_counter() - t_take)} | "
            f"frame {frame_idx + 1}/{total_frames} done in {perf_counter() - frame_t0:.1f}s",
        )

    _verbose_print(config, f"[{take.take_name}] aggregating {len(frame_outputs)} frame outputs")
    merged = _aggregate_in_memory_outputs(frame_outputs, take.gender)
    _verbose_print(config, f"[{take.take_name}] writing {take.output_path}")
    _merged_dict_to_amass_npz(
        merged=merged,
        output_path=take.output_path,
        source_path=take.source_path,
        take_name=take.take_name,
        frame_rate_hz=take.frame_rate_hz,
        pipeline="cpu-take",
        frame_batch_size=1,
    )
    _verbose_print(
        config,
        f"[{take.take_name}] {_progress_message(total_frames, total_frames, perf_counter() - t_take)} | written to {take.output_path}",
    )
    return {
        "source_path": str(take.source_path),
        "take_name": take.take_name,
        "output_path": str(take.output_path),
        "n_frames": int(take.poses.shape[0]),
        "gender": take.gender,
        "frame_rate_hz": float(take.frame_rate_hz),
        "status": "written",
    }


def _convert_one_take(
    take: TakeRecord,
    config: ConversionConfig,
) -> dict[str, Any]:
    from time import perf_counter

    # Keep temporary workdirs off the output volume so cleanup or sync tooling
    # on the results directory cannot interfere with in-flight conversions.
    work_root = Path(tempfile.mkdtemp(prefix="carepd_transfer_"))
    try:
        t_take = perf_counter()
        total_steps = 4
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(0, total_steps, 0.0)} | start classic transfer pipeline",
        )
        models_root = _prepare_official_models_root(work_root, config.smpl_model_root)
        target_gender = _resolve_model_gender(models_root / "smplx", "SMPLX", take.gender)
        smpl_meshes = work_root / "transfer_data" / "meshes" / "smpl"
        transfer_output = work_root / "transfer_output"
        config_path = work_root / "smpl2smplx.yaml"

        _write_smpl_meshes(smpl_meshes, take.poses, take.betas, take.gender, models_root)
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(1, total_steps, perf_counter() - t_take)} | "
            f"wrote temporary SMPL meshes to {smpl_meshes}",
        )
        _write_transfer_config(config_path, smpl_meshes, config.transfer_data_root, models_root, target_gender)
        _run_transfer_model(config.repo_root, config_path, config.limit_threads)
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(2, total_steps, perf_counter() - t_take)} | transfer_model done, merging output",
        )
        merged_path = _run_merge_output(config.repo_root, transfer_output, take.gender, config.limit_threads)
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(3, total_steps, perf_counter() - t_take)} | writing final NPZ to {take.output_path}",
        )
        _merged_pkl_to_amass_npz(
            merged_path=merged_path,
            output_path=take.output_path,
            source_path=take.source_path,
            take_name=take.take_name,
            frame_rate_hz=take.frame_rate_hz,
        )
        _verbose_print(
            config,
            f"[{take.take_name}] {_progress_message(4, total_steps, perf_counter() - t_take)} | written to {take.output_path}",
        )
        return {
            "source_path": str(take.source_path),
            "take_name": take.take_name,
            "output_path": str(take.output_path),
            "n_frames": int(take.poses.shape[0]),
            "gender": take.gender,
            "frame_rate_hz": float(take.frame_rate_hz),
            "status": "written",
        }
    finally:
        if config.keep_workdirs:
            kept = work_root
            print(f"Kept workdir: {kept}")
        else:
            shutil.rmtree(work_root, ignore_errors=True)


def _plan_take_records(
    source_path: Path,
    loaded: Any,
    config: ConversionConfig,
) -> tuple[list[TakeRecord], int]:
    planned: list[TakeRecord] = []
    skipped_existing = 0
    for take_name, record in _iter_take_records(loaded, source_path):
        poses, betas, gender, frame_rate_hz, diagnostics = _extract_take_record(
            record=record,
            source_path=source_path,
            take_name=take_name,
            fallback_frame_rate=float(config.frame_rate_hz),
            num_betas=int(config.num_betas),
        )
        output_path = _build_output_path(config.output_root, config.input_root, source_path, take_name)
        take = TakeRecord(
            source_path=source_path,
            take_name=str(take_name),
            output_path=output_path,
            poses=poses,
            betas=betas,
            gender=gender,
            frame_rate_hz=frame_rate_hz,
            diagnostics=diagnostics,
        )
        if output_path.exists() and not config.force:
            skipped_existing += 1
            continue
        planned.append(take)
    return planned, skipped_existing


def _process_source_file(source_path: Path, config: ConversionConfig) -> dict[str, Any]:
    try:
        _configure_worker_runtime(config.limit_threads)
        _verbose_print(config, f"[{source_path.name}] loading pickle")
        loaded = _load_pickle(source_path)
        _verbose_print(config, f"[{source_path.name}] planning take records")
        planned, skipped_existing = _plan_take_records(source_path, loaded, config)
        _verbose_print(
            config,
            f"[{source_path.name}] planned {len(planned)} take(s), skipped {skipped_existing} existing",
        )

        if config.pipeline == "in-memory":
            convert_take = _convert_one_take_in_memory
        elif config.pipeline == "cpu-take":
            convert_take = _convert_one_take_cpu_takes
        else:
            convert_take = _convert_one_take

        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for idx, take in enumerate(planned, start=1):
            try:
                _verbose_print(
                    config,
                    f"[{source_path.name}] converting take {idx}/{len(planned)} -> {take.take_name}",
                )
                print(f"Convert {take.source_path} :: {take.take_name}", flush=True)
                result = convert_take(take, config)
                results.append(result)
                _verbose_print(config, f"[{take.take_name}] completed successfully")
            except Exception as exc:
                failures.append(
                    {
                        "source_path": str(take.source_path),
                        "take_name": take.take_name,
                        "output_path": str(take.output_path),
                        "error": str(exc),
                    }
                )
                print(f"Fail {take.source_path} :: {take.take_name} -> {exc}", file=sys.stderr, flush=True)

        return {
            "source_path": str(source_path),
            "written_count": len(results),
            "skipped_existing_count": int(skipped_existing),
            "failed_count": len(failures),
            "files": results,
            "failures": failures,
            "status": "ok",
        }
    except Exception as exc:
        return {
            "source_path": str(source_path),
            "written_count": 0,
            "skipped_existing_count": 0,
            "failed_count": 1,
            "files": [],
            "failures": [
                {
                    "source_path": str(source_path),
                    "take_name": None,
                    "output_path": None,
                    "error": str(exc),
                }
            ],
            "status": "failed",
        }


def _collect_planned_takes(
    source_files: list[Path],
    config: ConversionConfig,
) -> tuple[list[TakeRecord], int, list[dict[str, Any]]]:
    planned_all: list[TakeRecord] = []
    skipped_existing = 0
    failures: list[dict[str, Any]] = []
    for source_path in source_files:
        try:
            _verbose_print(config, f"[{source_path.name}] loading pickle")
            loaded = _load_pickle(source_path)
            _verbose_print(config, f"[{source_path.name}] planning take records")
            planned, skipped = _plan_take_records(source_path, loaded, config)
            _verbose_print(
                config,
                f"[{source_path.name}] planned {len(planned)} take(s), skipped {skipped} existing",
            )
            planned_all.extend(planned)
            skipped_existing += skipped
        except Exception as exc:
            failures.append(
                {
                    "source_path": str(source_path),
                    "take_name": None,
                    "output_path": None,
                    "error": str(exc),
                }
            )
            print(f"Fail {source_path} -> {exc}", file=sys.stderr, flush=True)
    return planned_all, skipped_existing, failures


def _process_take_record(
    take: TakeRecord,
    config: ConversionConfig,
) -> dict[str, Any]:
    _configure_worker_runtime(config.limit_threads)
    if config.pipeline == "cpu-take":
        return _convert_one_take_cpu_takes(take, config)
    if config.pipeline == "in-memory":
        return _convert_one_take_in_memory(take, config)
    return _convert_one_take(take, config)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    repo_root = _resolve_repo_root(args.smplx_repo_root)
    transfer_data_root = _resolve_transfer_data_root(args.transfer_data_root)
    smpl_model_root = _resolve_smpl_model_root(args.smpl_model_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    source_files = _discover_pickles(input_root, args.recursive)
    if not source_files:
        raise FileNotFoundError(f"No .pkl files found under {input_root}")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    effective_workers = int(args.workers)
    cpu_take_pipeline = args.pipeline == "cpu-take"
    if args.pipeline == "in-memory" and effective_workers != 1:
        print(
            f"In-memory pipeline uses a single GPU worker; ignoring --workers={effective_workers}.",
            flush=True,
        )
        effective_workers = 1
    frame_batch_size = max(1, int(args.frame_batch_size))
    if cpu_take_pipeline:
        if frame_batch_size != 1:
            print(
                f"cpu-take pipeline runs one frame at a time; ignoring --frame-batch-size={frame_batch_size}.",
                flush=True,
            )
        frame_batch_size = 1

    output_root.mkdir(parents=True, exist_ok=True)

    config = ConversionConfig(
        input_root=input_root,
        output_root=output_root,
        repo_root=repo_root,
        transfer_data_root=transfer_data_root,
        smpl_model_root=smpl_model_root,
        force=bool(args.force),
        frame_rate_hz=float(args.frame_rate),
        num_betas=int(args.num_betas),
        frame_batch_size=frame_batch_size,
        solver_init=str(args.solver_init),
        solver_maxiters=int(args.solver_maxiters),
        freeze_betas_after_first_frame=bool(args.freeze_betas_after_first_frame),
        fast_warm_start=bool(args.fast_warm_start),
        warm_solver_maxiters=int(args.warm_solver_maxiters),
        verbose=bool(args.verbose),
        keep_workdirs=bool(args.keep_workdirs),
        limit_threads=effective_workers > 1 or args.pipeline in {"in-memory", "cpu-take"},
        pipeline=str(args.pipeline),
    )
    _verbose_print(
        config,
        (
            "[startup] "
            f"pipeline={config.pipeline}, workers={effective_workers}, frame_batch_size={config.frame_batch_size}, "
            f"solver_init={config.solver_init}, solver_maxiters={config.solver_maxiters}, shared_betas=True, "
            f"freeze_betas_after_first_frame={config.freeze_betas_after_first_frame}, "
            f"fast_warm_start={config.fast_warm_start}, warm_solver_maxiters={config.warm_solver_maxiters}, "
            f"input_root={config.input_root}, output_root={config.output_root}"
        ),
    )
    if config.limit_threads:
        _configure_worker_runtime(True)

    if args.dry_run:
        planned_total = 0
        skipped_existing = 0
        for source_path in source_files:
            loaded = _load_pickle(source_path)
            planned, skipped = _plan_take_records(source_path, loaded, config)
            planned_total += len(planned)
            skipped_existing += skipped
            print(
                f"{source_path}: planned {len(planned)} conversions, skipped {skipped}",
                flush=True,
            )
            for item in planned[:10]:
                print(f"  {item.source_path} -> {item.output_path}", flush=True)
            if len(planned) > 10:
                print(f"  ... {len(planned) - 10} more", flush=True)
        print(
            f"Planned {planned_total} conversions from {len(source_files)} source files "
            f"({skipped_existing} already present).",
            flush=True,
        )
        return

    source_results: list[dict[str, Any]] = []
    if cpu_take_pipeline:
        planned_takes, skipped_existing, planning_failures = _collect_planned_takes(source_files, config)
        print(
            f"Parallel mode: {effective_workers} worker(s) over {len(planned_takes)} take(s) "
            f"with {args.pipeline} pipeline.",
            flush=True,
        )

        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = list(planning_failures)
        if not planned_takes:
            pass
        elif effective_workers == 1:
            for idx, take in enumerate(planned_takes, start=1):
                try:
                    print(
                        f"[{idx}/{len(planned_takes)}] {Path(take.source_path).name}: "
                        f"take {take.take_name}",
                        flush=True,
                    )
                    result = _process_take_record(take, config)
                    results.append(result)
                    _verbose_print(config, f"[{take.take_name}] completed successfully")
                except Exception as exc:
                    failures.append(
                        {
                            "source_path": str(take.source_path),
                            "take_name": take.take_name,
                            "output_path": str(take.output_path),
                            "error": str(exc),
                        }
                    )
                    print(f"Fail {take.source_path} :: {take.take_name} -> {exc}", file=sys.stderr, flush=True)
        else:
            max_workers = min(effective_workers, len(planned_takes))
            max_in_flight = max(1, max_workers * 2)
            pending: dict[concurrent.futures.Future, TakeRecord] = {}
            take_iter = iter(planned_takes)

            def _submit_until_full(executor: concurrent.futures.ProcessPoolExecutor) -> None:
                while len(pending) < max_in_flight:
                    try:
                        take = next(take_iter)
                    except StopIteration:
                        return
                    future = executor.submit(_process_take_record, take, config)
                    pending[future] = take

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                _submit_until_full(executor)
                completed = 0
                while pending:
                    done, _ = concurrent.futures.wait(
                        list(pending.keys()),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        take = pending.pop(future)
                        completed += 1
                        try:
                            result = future.result()
                            results.append(result)
                            print(
                                f"[{completed}/{len(planned_takes)}] {Path(take.source_path).name}: "
                                f"written 1, skipped 0, failed 0",
                                flush=True,
                            )
                        except Exception as exc:
                            failures.append(
                                {
                                    "source_path": str(take.source_path),
                                    "take_name": take.take_name,
                                    "output_path": str(take.output_path),
                                    "error": str(exc),
                                }
                            )
                            print(f"Fail {take.source_path} :: {take.take_name} -> {exc}", file=sys.stderr, flush=True)
                    _submit_until_full(executor)
    elif effective_workers == 1:
        for idx, source_path in enumerate(source_files, start=1):
            result = _process_source_file(source_path, config)
            source_results.append(result)
            print(
                f"[{idx}/{len(source_files)}] {Path(result['source_path']).name}: "
                f"written {result['written_count']}, skipped {result['skipped_existing_count']}, "
                f"failed {result['failed_count']}",
                flush=True,
            )
    else:
        max_workers = min(effective_workers, len(source_files))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_source = {
                executor.submit(_process_source_file, source_path, config): source_path
                for source_path in source_files
            }
            completed = 0
            for future in concurrent.futures.as_completed(future_to_source):
                completed += 1
                result = future.result()
                source_results.append(result)
                print(
                    f"[{completed}/{len(source_files)}] {Path(result['source_path']).name}: "
                    f"written {result['written_count']}, skipped {result['skipped_existing_count']}, "
                    f"failed {result['failed_count']}",
                    flush=True,
                )

    if not cpu_take_pipeline:
        results = []
        failures = []
        skipped_existing = 0
        for source_result in source_results:
            results.extend(source_result["files"])
            failures.extend(source_result["failures"])
            skipped_existing += int(source_result["skipped_existing_count"])

    results.sort(key=lambda item: (item["source_path"], item["take_name"]))
    failures.sort(key=lambda item: (item["source_path"], item.get("take_name") or ""))

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "smplx_repo_root": str(repo_root),
        "transfer_data_root": str(transfer_data_root),
        "smpl_model_root": str(smpl_model_root),
        "pipeline": str(args.pipeline),
        "workers": int(effective_workers),
        "frame_batch_size": int(frame_batch_size),
        "solver_init": str(args.solver_init),
        "solver_maxiters": int(args.solver_maxiters),
        "shared_betas": True,
        "freeze_betas_after_first_frame": bool(args.freeze_betas_after_first_frame),
        "source_file_count": len(source_files),
        "written_count": len(results),
        "skipped_existing_count": int(skipped_existing),
        "failed_count": len(failures),
        "files": results,
        "failures": failures,
    }
    summary_path = output_root / str(args.summary_name)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary: {summary_path}")
    if failures:
        raise SystemExit(f"Failed {len(failures)} take(s)")


if __name__ == "__main__":
    main()

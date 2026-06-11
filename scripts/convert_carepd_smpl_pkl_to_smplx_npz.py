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


DEFAULT_INPUT_ROOT = Path("/Volumes/MAEVE/dataset/CARE-PD/Canonicalized_SMPL_pickles")
DEFAULT_OUTPUT_DIR = Path("/Volumes/MAEVE/dataset/CARE-PD/npz")
DEFAULT_FRAME_RATE_HZ = 30.0
DEFAULT_NUM_BETAS = 16
DEFAULT_SMPLX_REPO_ROOT = Path(__file__).resolve().parents[1] / "smplx"
DEFAULT_TRANSFER_DATA_ROOT = Path(__file__).resolve().parents[1] / "transfer_data"
DEFAULT_SMPL_MODEL_ROOT = Path(__file__).resolve().parents[1] / "model" / "smpl"


@dataclass(frozen=True)
class ConversionConfig:
    input_root: Path
    output_root: Path
    repo_root: Path
    transfer_data_root: Path
    smpl_model_root: Path
    force: bool
    frame_rate_hz: float
    num_betas: int
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
        help="Number of parallel source-pickle workers (default: max(1, cpu_count/2)).",
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
        "--keep-workdirs",
        action="store_true",
        help="Keep temp workdirs for debugging.",
    )
    parser.add_argument(
        "--summary-name",
        default="conversion_summary.json",
        help="JSON summary filename written inside output root.",
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


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(_unwrap(value))
    except Exception:
        return float(default)


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
    frame_rate_hz = _to_float(get("mocap_frame_rate", "mocap_framerate", "fps", "frame_rate"), fallback_frame_rate)
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
    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        return R.from_matrix(arr.astype(np.float64)).as_rotvec().astype(np.float32)
    if arr.ndim == 4 and arr.shape[1] == 1 and arr.shape[2:] == (3, 3):
        return R.from_matrix(arr[:, 0].astype(np.float64)).as_rotvec().astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[1] == 1:
        return arr[:, 0].astype(np.float32, copy=False)
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


def _merged_pkl_to_amass_npz(
    merged_path: Path,
    output_path: Path,
    source_path: Path,
    take_name: str,
    frame_rate_hz: float,
) -> None:
    with merged_path.open("rb") as handle:
        merged = pickle.load(handle)

    if merged.get("transl") is None:
        raise KeyError(f"Missing transl in {merged_path}")
    if merged.get("global_orient") is None:
        raise KeyError(f"Missing global_orient in {merged_path}")
    if merged.get("body_pose") is None:
        raise KeyError(f"Missing body_pose in {merged_path}")

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

    betas = np.asarray(merged.get("betas"), dtype=np.float32)
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
                    "source_path": str(source_path),
                    "take_name": take_name,
                    "frame_rate_hz": float(frame_rate_hz),
                },
                sort_keys=True,
            )
        ),
    )


def _convert_one_take(
    take: TakeRecord,
    config: ConversionConfig,
) -> dict[str, Any]:
    work_root = Path(tempfile.mkdtemp(prefix="carepd_transfer_", dir=str(config.output_root)))
    try:
        models_root = _prepare_official_models_root(work_root, config.smpl_model_root)
        target_gender = _resolve_model_gender(models_root / "smplx", "SMPLX", take.gender)
        smpl_meshes = work_root / "transfer_data" / "meshes" / "smpl"
        transfer_output = work_root / "transfer_output"
        config_path = work_root / "smpl2smplx.yaml"

        motion_npz = work_root / "motion.npz"
        _write_motion_npz(motion_npz, take.poses, take.betas, take.gender, take.frame_rate_hz)
        _write_smpl_meshes(smpl_meshes, take.poses, take.betas, take.gender, models_root)
        _write_transfer_config(config_path, smpl_meshes, config.transfer_data_root, models_root, target_gender)
        _run_transfer_model(config.repo_root, config_path, config.limit_threads)
        merged_path = _run_merge_output(config.repo_root, transfer_output, take.gender, config.limit_threads)
        _merged_pkl_to_amass_npz(
            merged_path=merged_path,
            output_path=take.output_path,
            source_path=take.source_path,
            take_name=take.take_name,
            frame_rate_hz=take.frame_rate_hz,
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
        loaded = _load_pickle(source_path)
        planned, skipped_existing = _plan_take_records(source_path, loaded, config)

        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for take in planned:
            try:
                print(f"Convert {take.source_path} :: {take.take_name}", flush=True)
                results.append(_convert_one_take(take, config))
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
        keep_workdirs=bool(args.keep_workdirs),
        limit_threads=args.workers > 1,
    )

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

    print(
        f"Parallel mode: {args.workers} worker(s) over {len(source_files)} source pickle(s).",
        flush=True,
    )

    source_results: list[dict[str, Any]] = []
    if args.workers == 1:
        for source_path in source_files:
            source_results.append(_process_source_file(source_path, config))
    else:
        max_workers = min(int(args.workers), len(source_files))
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

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
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
        "source_file_count": len(source_files),
        "written_count": len(results),
        "skipped_existing_count": int(skipped_existing),
        "failed_count": len(failures),
        "files": results,
        "failures": failures,
    }
    summary_path = output_root / str(args.summary_name)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary: {summary_path}")
    if failures:
        raise SystemExit(f"Failed {len(failures)} take(s)")


if __name__ == "__main__":
    main()

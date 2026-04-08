from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _to_scalar(value: Any) -> Any:
    """Convert 0-d numpy arrays to Python scalars."""
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _to_str(value: Any) -> str:
    value = _to_scalar(value)
    return str(value)


def _to_float(value: Any) -> float:
    value = _to_scalar(value)
    return float(value)


@dataclass
class AMASSSequence:
    """In-memory AMASS sample split into SMPL-X fields used by the pipeline."""

    source_path: Path
    surface_model_type: str
    gender: str
    frame_rate_hz: float
    trans: np.ndarray
    root_orient: np.ndarray
    body_pose: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    jaw_pose: np.ndarray
    leye_pose: np.ndarray
    reye_pose: np.ndarray
    betas: np.ndarray

    @property
    def n_frames(self) -> int:
        return int(self.trans.shape[0])

    @property
    def sex(self) -> str:
        return self.gender.lower()

    def copy_with_gender(self, gender: str) -> "AMASSSequence":
        """Return a deep copy while overriding gender metadata."""
        return AMASSSequence(
            source_path=self.source_path,
            surface_model_type=self.surface_model_type,
            gender=gender,
            frame_rate_hz=self.frame_rate_hz,
            trans=self.trans.copy(),
            root_orient=self.root_orient.copy(),
            body_pose=self.body_pose.copy(),
            left_hand_pose=self.left_hand_pose.copy(),
            right_hand_pose=self.right_hand_pose.copy(),
            jaw_pose=self.jaw_pose.copy(),
            leye_pose=self.leye_pose.copy(),
            reye_pose=self.reye_pose.copy(),
            betas=self.betas.copy(),
        )

    def make_neutral(self, n_frames: int = 30) -> "AMASSSequence":
        """Create a short neutral pose sequence used by OpenSim scaling."""
        n = int(n_frames)
        if n <= 0:
            raise ValueError(f"n_frames must be > 0, got {n_frames}")
        zeros3 = np.zeros((n, 3), dtype=np.float32)
        return AMASSSequence(
            source_path=self.source_path,
            surface_model_type=self.surface_model_type,
            gender=self.gender,
            frame_rate_hz=self.frame_rate_hz,
            trans=zeros3.copy(),
            root_orient=zeros3.copy(),
            body_pose=np.zeros((n, self.body_pose.shape[1]), dtype=np.float32),
            left_hand_pose=np.zeros((n, self.left_hand_pose.shape[1]), dtype=np.float32),
            right_hand_pose=np.zeros((n, self.right_hand_pose.shape[1]), dtype=np.float32),
            jaw_pose=zeros3.copy(),
            leye_pose=zeros3.copy(),
            reye_pose=zeros3.copy(),
            betas=self.betas.copy(),
        )


def _check_shape(array: np.ndarray, expected_last_dim: int, name: str) -> None:
    if array.ndim != 2 or array.shape[1] != expected_last_dim:
        raise ValueError(
            f"Expected {name} shape (T, {expected_last_dim}), got {array.shape}"
        )


def _load_required(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in npz.files:
        raise KeyError(f"Missing key '{key}' in AMASS npz")
    return npz[key]


def load_amass_npz(path: str | Path) -> AMASSSequence:
    """Load and validate the AMASS npz format needed by this workflow."""
    source = Path(path).resolve()
    with np.load(source, allow_pickle=True) as npz:
        surface_model_type = _to_str(_load_required(npz, "surface_model_type"))
        if surface_model_type.lower() != "smplx":
            raise ValueError(
                "Only AMASS files with surface_model_type='smplx' are supported. "
                f"Found '{surface_model_type}'."
            )

        gender = _to_str(_load_required(npz, "gender")).lower()
        frame_rate_hz = _to_float(_load_required(npz, "mocap_frame_rate"))

        trans = np.asarray(_load_required(npz, "trans"), dtype=np.float32)
        root_orient = np.asarray(_load_required(npz, "root_orient"), dtype=np.float32)
        body_pose = np.asarray(_load_required(npz, "pose_body"), dtype=np.float32)
        pose_hand = np.asarray(_load_required(npz, "pose_hand"), dtype=np.float32)
        jaw_pose = np.asarray(_load_required(npz, "pose_jaw"), dtype=np.float32)
        pose_eye = np.asarray(_load_required(npz, "pose_eye"), dtype=np.float32)
        betas = np.asarray(_load_required(npz, "betas"), dtype=np.float32).reshape(-1)

    _check_shape(trans, 3, "trans")
    _check_shape(root_orient, 3, "root_orient")
    _check_shape(body_pose, 63, "pose_body")
    _check_shape(jaw_pose, 3, "pose_jaw")

    if pose_hand.ndim != 2 or pose_hand.shape[1] < 90:
        raise ValueError(f"Expected pose_hand shape (T, >=90), got {pose_hand.shape}")
    left_hand_pose = pose_hand[:, :45].astype(np.float32, copy=False)
    right_hand_pose = pose_hand[:, 45:90].astype(np.float32, copy=False)

    if pose_eye.ndim != 2 or pose_eye.shape[1] < 6:
        raise ValueError(f"Expected pose_eye shape (T, >=6), got {pose_eye.shape}")
    leye_pose = pose_eye[:, :3].astype(np.float32, copy=False)
    reye_pose = pose_eye[:, 3:6].astype(np.float32, copy=False)

    n_frames = trans.shape[0]
    expected_arrays = {
        "root_orient": root_orient,
        "body_pose": body_pose,
        "left_hand_pose": left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
    }
    for name, array in expected_arrays.items():
        if array.shape[0] != n_frames:
            raise ValueError(
                f"Frame count mismatch for {name}: expected {n_frames}, got {array.shape[0]}"
            )

    if betas.size < 10:
        raise ValueError(f"Expected at least 10 betas, got {betas.size}")

    return AMASSSequence(
        source_path=source,
        surface_model_type=surface_model_type.lower(),
        gender=gender,
        frame_rate_hz=frame_rate_hz,
        trans=trans,
        root_orient=root_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        betas=betas,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_REQUIRED_FIELDS = (
    "surface_model_type",
    "gender",
    "mocap_frame_rate",
    "trans",
    "root_orient",
    "pose_body",
    "pose_hand",
    "pose_jaw",
    "pose_eye",
    "betas",
)

_SHARED_FIELDS = ("surface_model_type", "gender", "mocap_frame_rate", "betas")


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


def _unwrap_object(value: Any) -> Any:
    """Normalize object arrays/scalars to regular Python containers."""
    value = _to_scalar(value)
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        return _unwrap_object(value.reshape(()).item())
    return value


def _build_sequence_from_fields(
    source: Path,
    fields: Mapping[str, Any],
) -> AMASSSequence:
    """Build and validate one AMASSSequence from a field mapping."""
    missing_fields = [key for key in _REQUIRED_FIELDS if key not in fields]
    if missing_fields:
        raise KeyError(
            f"Missing required AMASS fields {missing_fields} while parsing '{source.name}'"
        )

    surface_model_type = _to_str(fields["surface_model_type"])
    if surface_model_type.lower() != "smplx":
        raise ValueError(
            "Only AMASS files with surface_model_type='smplx' are supported. "
            f"Found '{surface_model_type}'."
        )

    gender = _to_str(fields["gender"]).lower()
    frame_rate_hz = _to_float(fields["mocap_frame_rate"])

    trans = np.asarray(fields["trans"], dtype=np.float32)
    root_orient = np.asarray(fields["root_orient"], dtype=np.float32)
    body_pose = np.asarray(fields["pose_body"], dtype=np.float32)
    pose_hand = np.asarray(fields["pose_hand"], dtype=np.float32)
    jaw_pose = np.asarray(fields["pose_jaw"], dtype=np.float32)
    pose_eye = np.asarray(fields["pose_eye"], dtype=np.float32)
    betas = np.asarray(fields["betas"], dtype=np.float32).reshape(-1)

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


def _extract_records_from_trials_obj(trials_obj: Any) -> dict[str, dict[str, Any]]:
    """Extract trial records from object-like ``trials`` containers."""
    result: dict[str, dict[str, Any]] = {}
    unwrapped = _unwrap_object(trials_obj)

    if isinstance(unwrapped, Mapping):
        for trial_name, record in unwrapped.items():
            if isinstance(record, Mapping):
                result[str(trial_name)] = dict(record)
        return result

    if isinstance(unwrapped, (list, tuple)):
        for idx, record in enumerate(unwrapped):
            record = _unwrap_object(record)
            if isinstance(record, Mapping):
                candidate_name = record.get("trial_name") or record.get("name")
                trial_name = str(candidate_name) if candidate_name else f"trial_{idx:03d}"
                result[trial_name] = dict(record)
        return result

    return result


def load_all_amass_npz(path: str | Path) -> dict[str, AMASSSequence]:
    """
    Load one or multiple AMASS sequences from a single npz.

    Supported layouts:
    - Standard single-trial AMASS file with top-level keys.
    - Multi-trial with prefixed keys like ``<trial>/trans``.
    - Multi-trial object container under ``trials`` (dict/list of records).
    """
    source = Path(path).resolve()
    with np.load(source, allow_pickle=True) as npz:
        shared_fields = {name: npz[name] for name in _SHARED_FIELDS if name in npz.files}
        prefixed_records: dict[str, dict[str, Any]] = {}

        for key in npz.files:
            if "/" not in key:
                continue
            trial_name, field_name = key.rsplit("/", 1)
            if field_name in _REQUIRED_FIELDS:
                prefixed_records.setdefault(trial_name, {})[field_name] = npz[key]

        if prefixed_records:
            sequences: dict[str, AMASSSequence] = {}
            for trial_name, fields in prefixed_records.items():
                merged_fields = dict(shared_fields)
                merged_fields.update(fields)
                sequences[str(trial_name)] = _build_sequence_from_fields(source, merged_fields)
            return sequences

        if "trials" in npz.files:
            trial_records = _extract_records_from_trials_obj(npz["trials"])
            if trial_records:
                sequences = {}
                for trial_name, fields in trial_records.items():
                    merged_fields = dict(shared_fields)
                    merged_fields.update(fields)
                    sequences[str(trial_name)] = _build_sequence_from_fields(source, merged_fields)
                return sequences

        # Fallback: standard AMASS single-trial layout.
        top_level_fields = {name: npz[name] for name in npz.files}
        default_trial_name = source.stem
        return {default_trial_name: _build_sequence_from_fields(source, top_level_fields)}


def load_amass_npz(path: str | Path) -> AMASSSequence:
    """Load one AMASS sequence from npz (first trial if multi-trial)."""
    all_sequences = load_all_amass_npz(path)
    first_trial_name = next(iter(all_sequences))
    return all_sequences[first_trial_name]

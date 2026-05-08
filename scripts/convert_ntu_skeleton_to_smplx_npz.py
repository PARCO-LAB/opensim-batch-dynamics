#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


NTU_NUM_JOINTS = 25
DEFAULT_FRAME_RATE_HZ = 30.0
TARGET_FRAME_Z_UP = "z-up"
TARGET_FRAME_Y_UP = "y-up"

# NTU RGB+D 25-joint order, zero-based.
NTU_JOINT_NAMES = (
    "spine_base",
    "spine_mid",
    "neck",
    "head",
    "shoulder_l",
    "elbow_l",
    "wrist_l",
    "hand_l",
    "shoulder_r",
    "elbow_r",
    "wrist_r",
    "hand_r",
    "hip_l",
    "knee_l",
    "ankle_l",
    "foot_l",
    "hip_r",
    "knee_r",
    "ankle_r",
    "foot_r",
    "spine_shoulder",
    "hand_tip_l",
    "thumb_l",
    "hand_tip_r",
    "thumb_r",
)

NTU_LEFT_RIGHT_PAIRS = (
    (4, 8),
    (5, 9),
    (6, 10),
    (7, 11),
    (12, 16),
    (13, 17),
    (14, 18),
    (15, 19),
    (21, 23),
    (22, 24),
)

# SMPL-X joint indices follow the standard SMPL body prefix used by smplx.
SMPLX_BODY_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}


@dataclass(frozen=True)
class NTUMetadata:
    setup: int | None
    camera: int | None
    performer: int | None
    replication: int | None
    action: int | None

    @property
    def performer_key(self) -> str:
        return f"P{self.performer:03d}" if self.performer is not None else "Punknown"

    @property
    def action_key(self) -> str:
        return f"A{self.action:03d}" if self.action is not None else "Aunknown"


@dataclass
class NTUTrack:
    body_id: str
    joints: np.ndarray
    tracking: np.ndarray

    @property
    def valid_frames(self) -> int:
        return int(np.sum(np.isfinite(self.joints).all(axis=(1, 2))))

    @property
    def motion_score(self) -> float:
        flat = self.joints.reshape(self.joints.shape[0], -1)
        if not np.isfinite(flat).any():
            return 0.0
        return float(np.nansum(np.nanvar(flat, axis=0)))

    def copy_with_joints(self, joints: np.ndarray) -> "NTUTrack":
        return NTUTrack(
            body_id=self.body_id,
            joints=joints.astype(np.float32, copy=True),
            tracking=self.tracking.copy(),
        )


@dataclass
class SelectedTrack:
    source_path: Path
    metadata: NTUMetadata
    body_rank: int
    track: NTUTrack

    @property
    def output_stem(self) -> str:
        if self.body_rank == 0:
            return self.source_path.stem
        return f"{self.source_path.stem}_body{self.body_rank:02d}"

    @property
    def shape_key(self) -> str:
        base = self.metadata.performer_key
        if self.body_rank == 0:
            return base
        return f"{base}_body{self.body_rank:02d}"


@dataclass(frozen=True)
class ShapeTarget:
    name: str
    ntu_a: int
    ntu_b: int
    smplx_a: int
    smplx_b: int
    weight: float


@dataclass(frozen=True)
class PoseTarget:
    name: str
    smplx_joint: int
    ntu_joint: int | None
    ntu_average: tuple[int, int] | None
    weight: float


SHAPE_TARGETS = (
    ShapeTarget("shoulder_width", 4, 8, 16, 17, 1.5),
    ShapeTarget("hip_width", 12, 16, 1, 2, 1.2),
    ShapeTarget("torso_height", 0, 20, 0, 12, 1.0),
    ShapeTarget("left_upper_arm", 4, 5, 16, 18, 1.0),
    ShapeTarget("left_forearm", 5, 6, 18, 20, 1.0),
    ShapeTarget("right_upper_arm", 8, 9, 17, 19, 1.0),
    ShapeTarget("right_forearm", 9, 10, 19, 21, 1.0),
    ShapeTarget("left_thigh", 12, 13, 1, 4, 1.2),
    ShapeTarget("left_shank", 13, 14, 4, 7, 1.2),
    ShapeTarget("right_thigh", 16, 17, 2, 5, 1.2),
    ShapeTarget("right_shank", 17, 18, 5, 8, 1.2),
)

POSE_TARGETS = (
    PoseTarget("pelvis", 0, None, (12, 16), 1.2),
    PoseTarget("left_hip", 1, 12, None, 1.0),
    PoseTarget("right_hip", 2, 16, None, 1.0),
    PoseTarget("left_knee", 4, 13, None, 1.4),
    PoseTarget("right_knee", 5, 17, None, 1.4),
    PoseTarget("left_ankle", 7, 14, None, 1.4),
    PoseTarget("right_ankle", 8, 18, None, 1.4),
    PoseTarget("left_foot", 10, 15, None, 0.7),
    PoseTarget("right_foot", 11, 19, None, 0.7),
    PoseTarget("neck", 12, 2, None, 1.0),
    PoseTarget("head", 15, 3, None, 0.6),
    PoseTarget("left_shoulder", 16, 4, None, 1.1),
    PoseTarget("right_shoulder", 17, 8, None, 1.1),
    PoseTarget("left_elbow", 18, 5, None, 1.3),
    PoseTarget("right_elbow", 19, 9, None, 1.3),
    PoseTarget("left_wrist", 20, 6, None, 1.2),
    PoseTarget("right_wrist", 21, 10, None, 1.2),
)


def parse_ntu_metadata(path: str | Path) -> NTUMetadata:
    match = re.search(
        r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<performer>\d{3})R(?P<replication>\d{3})A(?P<action>\d{3})",
        Path(path).stem,
    )
    if not match:
        return NTUMetadata(None, None, None, None, None)
    return NTUMetadata(
        setup=int(match.group("setup")),
        camera=int(match.group("camera")),
        performer=int(match.group("performer")),
        replication=int(match.group("replication")),
        action=int(match.group("action")),
    )


def discover_skeleton_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.skeleton" if recursive else "*.skeleton"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def normalize_performer_key(raw: str) -> str:
    value = str(raw).strip()
    if not value:
        raise ValueError("Empty performer key")
    if value.lower() in {"unknown", "punknown"}:
        return "Punknown"
    if value[0].lower() == "p":
        value = value[1:]
    return f"P{int(value):03d}"


def read_ntu_skeleton(path: str | Path) -> dict[str, NTUTrack]:
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Empty NTU skeleton file: {source}")

    try:
        num_frames = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid frame count in {source}: {lines[0]!r}") from exc

    tracks: dict[str, NTUTrack] = {}
    line_idx = 1
    for frame_idx in range(num_frames):
        if line_idx >= len(lines):
            raise ValueError(f"Unexpected end of file in {source} at frame {frame_idx}")
        num_bodies = int(lines[line_idx].strip())
        line_idx += 1

        for _ in range(num_bodies):
            if line_idx >= len(lines):
                raise ValueError(f"Missing body header in {source} at frame {frame_idx}")
            body_fields = lines[line_idx].strip().split()
            body_id = body_fields[0] if body_fields else f"body_{len(tracks)}"
            line_idx += 1

            if line_idx >= len(lines):
                raise ValueError(f"Missing joint count in {source} at frame {frame_idx}")
            num_joints = int(lines[line_idx].strip())
            line_idx += 1

            if body_id not in tracks:
                tracks[body_id] = NTUTrack(
                    body_id=body_id,
                    joints=np.full((num_frames, NTU_NUM_JOINTS, 3), np.nan, dtype=np.float32),
                    tracking=np.zeros((num_frames, NTU_NUM_JOINTS), dtype=np.float32),
                )
            track = tracks[body_id]

            for joint_idx in range(num_joints):
                if line_idx >= len(lines):
                    raise ValueError(f"Missing joint row in {source} at frame {frame_idx}")
                fields = lines[line_idx].strip().split()
                line_idx += 1
                if joint_idx >= NTU_NUM_JOINTS:
                    continue
                if len(fields) < 3:
                    continue
                track.joints[frame_idx, joint_idx, :] = np.asarray(fields[:3], dtype=np.float32)
                if len(fields) >= 12:
                    try:
                        track.tracking[frame_idx, joint_idx] = float(fields[11])
                    except ValueError:
                        track.tracking[frame_idx, joint_idx] = 0.0
                else:
                    track.tracking[frame_idx, joint_idx] = 2.0

    return tracks


def rank_tracks(tracks: dict[str, NTUTrack]) -> list[NTUTrack]:
    return sorted(
        tracks.values(),
        key=lambda track: (-track.valid_frames, -track.motion_score, track.body_id),
    )


def select_tracks(path: Path, actor_mode: str) -> list[SelectedTrack]:
    metadata = parse_ntu_metadata(path)
    tracks = rank_tracks(read_ntu_skeleton(path))
    if actor_mode == "primary":
        tracks = tracks[:1]
    return [
        SelectedTrack(source_path=path, metadata=metadata, body_rank=rank, track=track)
        for rank, track in enumerate(tracks)
        if track.valid_frames > 0
    ]


def transform_joints_to_target_frame(joints: np.ndarray, target_frame: str) -> np.ndarray:
    """
    Convert raw NTU camera coordinates to the pipeline world frame.

    NTU skeletons are Y-up in camera coordinates. The BSM/Nimble/OpenSim path in
    this repo uses model gravity along -Z, so generated SMPL-X world motion must
    be Z-up. Rx(+90 deg) maps old Y to new Z while preserving handedness:
        [x, y, z] -> [x, -z, y]
    """
    if target_frame == TARGET_FRAME_Y_UP:
        return joints.astype(np.float32, copy=True)
    if target_frame != TARGET_FRAME_Z_UP:
        raise ValueError(f"Unsupported target frame: {target_frame}")
    transformed = joints.astype(np.float32, copy=True)
    x = transformed[..., 0].copy()
    y = transformed[..., 1].copy()
    z = transformed[..., 2].copy()
    transformed[..., 0] = x
    transformed[..., 1] = -z
    transformed[..., 2] = y
    return transformed


def swap_ntu_left_right(joints: np.ndarray) -> np.ndarray:
    swapped = joints.astype(np.float32, copy=True)
    for left_idx, right_idx in NTU_LEFT_RIGHT_PAIRS:
        left = swapped[:, left_idx, :].copy()
        swapped[:, left_idx, :] = swapped[:, right_idx, :]
        swapped[:, right_idx, :] = left
    return swapped


def maybe_swap_track_left_right(track: NTUTrack, swap_left_right: bool) -> NTUTrack:
    if not swap_left_right:
        return track
    swapped_joints = swap_ntu_left_right(track.joints)
    swapped_tracking = track.tracking.copy()
    for left_idx, right_idx in NTU_LEFT_RIGHT_PAIRS:
        left = swapped_tracking[:, left_idx].copy()
        swapped_tracking[:, left_idx] = swapped_tracking[:, right_idx]
        swapped_tracking[:, right_idx] = left
    return NTUTrack(body_id=track.body_id, joints=swapped_joints, tracking=swapped_tracking)


def transform_track_to_target_frame(track: NTUTrack, target_frame: str) -> NTUTrack:
    return track.copy_with_joints(transform_joints_to_target_frame(track.joints, target_frame))


def target_frame_up_axis(target_frame: str) -> int:
    if target_frame == TARGET_FRAME_Z_UP:
        return 2
    if target_frame == TARGET_FRAME_Y_UP:
        return 1
    raise ValueError(f"Unsupported target frame: {target_frame}")


def target_frame_root_init(target_frame: str) -> np.ndarray:
    if target_frame == TARGET_FRAME_Z_UP:
        return np.array([math.pi / 2.0, 0.0, 0.0], dtype=np.float32)
    if target_frame == TARGET_FRAME_Y_UP:
        return np.zeros(3, dtype=np.float32)
    raise ValueError(f"Unsupported target frame: {target_frame}")


def interpolate_missing_joints(joints: np.ndarray) -> np.ndarray:
    """Fill missing NTU frames/joints by per-coordinate linear interpolation."""
    filled = joints.astype(np.float32, copy=True)
    frame_index = np.arange(filled.shape[0], dtype=np.float32)
    for joint_idx in range(filled.shape[1]):
        valid_joint = np.isfinite(filled[:, joint_idx, :]).all(axis=1)
        if not np.any(valid_joint):
            filled[:, joint_idx, :] = 0.0
            continue
        for axis in range(3):
            values = filled[:, joint_idx, axis]
            valid_axis = np.isfinite(values) & valid_joint
            if np.sum(valid_axis) == 1:
                values[~valid_axis] = values[valid_axis][0]
            elif np.sum(valid_axis) > 1:
                values[:] = np.interp(frame_index, frame_index[valid_axis], values[valid_axis])
            else:
                values[:] = 0.0
            filled[:, joint_idx, axis] = values
    return filled


def tracking_to_weight(tracking: np.ndarray) -> np.ndarray:
    """Map NTU tracking state 0/1/2 to continuous fitting weights."""
    weights = np.zeros_like(tracking, dtype=np.float32)
    weights[tracking >= 2.0] = 1.0
    weights[(tracking > 0.0) & (tracking < 2.0)] = 0.35
    return weights


def collect_segment_lengths(track: NTUTrack) -> dict[str, float]:
    joints = track.joints
    weights = tracking_to_weight(track.tracking)
    lengths: dict[str, float] = {}
    for target in SHAPE_TARGETS:
        valid = (
            np.isfinite(joints[:, target.ntu_a, :]).all(axis=1)
            & np.isfinite(joints[:, target.ntu_b, :]).all(axis=1)
            & (weights[:, target.ntu_a] > 0.0)
            & (weights[:, target.ntu_b] > 0.0)
        )
        if not np.any(valid):
            continue
        distance = np.linalg.norm(joints[valid, target.ntu_a, :] - joints[valid, target.ntu_b, :], axis=1)
        distance = distance[np.isfinite(distance) & (distance > 0.03)]
        if distance.size:
            lengths[target.name] = float(np.median(distance))
    return lengths


def aggregate_shape_targets(length_records: list[dict[str, float]]) -> dict[str, float]:
    by_name: dict[str, list[float]] = {}
    for record in length_records:
        for name, value in record.items():
            by_name.setdefault(name, []).append(float(value))
    return {name: float(np.median(values)) for name, values in by_name.items() if values}


def resolve_smplx_model_path(model_dir: str | Path, gender: str) -> tuple[Path, str]:
    path = Path(model_dir).resolve()
    normalized_gender = gender.lower()
    if normalized_gender not in {"neutral", "male", "female"}:
        normalized_gender = "neutral"

    candidate_file = path / f"SMPLX_{normalized_gender.upper()}.npz"
    neutral_file = path / "SMPLX_NEUTRAL.npz"
    subdir = path / "smplx"
    if candidate_file.exists():
        return candidate_file, normalized_gender
    if neutral_file.exists():
        return neutral_file, "neutral"
    if subdir.exists():
        return subdir, normalized_gender
    raise FileNotFoundError(
        "Could not find SMPL-X model files. Expected one of:\n"
        f"- {candidate_file}\n"
        f"- {neutral_file}\n"
        f"- {subdir}/SMPLX_{normalized_gender.upper()}.npz"
    )


def _import_torch_and_smplx():
    try:
        import smplx
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "SMPL-X fitting requires torch and smplx. Activate the repo conda env "
            "(`conda activate opensim-torque`) or install the dependencies from environment.yml."
        ) from exc
    return torch, smplx


def _zero_pose_tensors(torch, n_frames: int, device: str) -> dict[str, object]:
    dtype = torch.float32
    return {
        "left_hand_pose": torch.zeros((n_frames, 45), dtype=dtype, device=device),
        "right_hand_pose": torch.zeros((n_frames, 45), dtype=dtype, device=device),
        "jaw_pose": torch.zeros((n_frames, 3), dtype=dtype, device=device),
        "leye_pose": torch.zeros((n_frames, 3), dtype=dtype, device=device),
        "reye_pose": torch.zeros((n_frames, 3), dtype=dtype, device=device),
    }


def fit_shape_betas(
    segment_targets: dict[str, float],
    smplx_model_dir: Path,
    gender: str,
    num_betas: int,
    device: str,
    iterations: int,
    lr: float,
    beta_prior_weight: float,
) -> np.ndarray:
    torch, smplx = _import_torch_and_smplx()
    model_path, resolved_gender = resolve_smplx_model_path(smplx_model_dir, gender)
    model = smplx.create(
        model_path=str(model_path),
        model_type="smplx",
        gender=resolved_gender,
        ext="npz",
        use_pca=False,
        flat_hand_mean=False,
        num_betas=num_betas,
        batch_size=1,
    ).to(device)

    available = [target for target in SHAPE_TARGETS if target.name in segment_targets]
    if not available:
        return np.zeros(num_betas, dtype=np.float32)

    betas = torch.zeros((1, num_betas), dtype=torch.float32, device=device, requires_grad=True)
    zeros = {
        "global_orient": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "body_pose": torch.zeros((1, 63), dtype=torch.float32, device=device),
        "transl": torch.zeros((1, 3), dtype=torch.float32, device=device),
        **_zero_pose_tensors(torch, 1, device),
    }
    optimizer = torch.optim.Adam([betas], lr=float(lr))
    target_lengths = torch.tensor(
        [segment_targets[item.name] for item in available],
        dtype=torch.float32,
        device=device,
    )
    target_weights = torch.tensor(
        [item.weight for item in available],
        dtype=torch.float32,
        device=device,
    )

    for _ in range(max(0, int(iterations))):
        optimizer.zero_grad(set_to_none=True)
        output = model(betas=betas, return_verts=False, **zeros)
        joints = output.joints[0]
        smpl_lengths = []
        for item in available:
            smpl_lengths.append(torch.linalg.norm(joints[item.smplx_a] - joints[item.smplx_b]))
        smpl_lengths_t = torch.stack(smpl_lengths)
        loss = torch.mean(target_weights * (smpl_lengths_t - target_lengths) ** 2)
        loss = loss + float(beta_prior_weight) * torch.mean(betas**2)
        loss.backward()
        optimizer.step()

    return betas.detach().cpu().numpy().reshape(-1).astype(np.float32)


def build_pose_targets(
    joints: np.ndarray,
    tracking: np.ndarray,
    up_axis: int,
) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    filled = interpolate_missing_joints(joints)
    joint_weights = tracking_to_weight(tracking)
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    smplx_indices: list[int] = []

    for item in POSE_TARGETS:
        if item.ntu_joint is not None:
            target = filled[:, item.ntu_joint, :]
            weight = joint_weights[:, item.ntu_joint] * float(item.weight)
        elif item.ntu_average is not None:
            a, b = item.ntu_average
            target = 0.5 * (filled[:, a, :] + filled[:, b, :])
            weight = np.minimum(joint_weights[:, a], joint_weights[:, b]) * float(item.weight)
        else:
            raise ValueError(f"Pose target {item.name} has no NTU source")
        targets.append(target.astype(np.float32, copy=False))
        weights.append(weight.astype(np.float32, copy=False))
        smplx_indices.append(item.smplx_joint)

    target_array = np.stack(targets, axis=1).astype(np.float32)
    weight_array = np.stack(weights, axis=1).astype(np.float32)
    floor_height = estimate_floor_height(filled, joint_weights, up_axis=up_axis)
    return target_array, weight_array, smplx_indices, floor_height


def estimate_floor_height(joints: np.ndarray, weights: np.ndarray, up_axis: int) -> float:
    foot_indices = [14, 15, 18, 19]
    values = []
    for idx in foot_indices:
        valid = np.isfinite(joints[:, idx, up_axis]) & (weights[:, idx] > 0.0)
        if np.any(valid):
            values.extend(joints[valid, idx, up_axis].astype(float).tolist())
    if not values:
        finite_values = joints[:, :, up_axis][np.isfinite(joints[:, :, up_axis])]
        if finite_values.size == 0:
            return 0.0
        values = finite_values.astype(float).tolist()
    return float(np.percentile(np.asarray(values, dtype=np.float64), 2.0))


def _second_difference_loss(torch, value) -> object:
    if value.shape[0] < 3:
        return torch.zeros((), dtype=value.dtype, device=value.device)
    diff2 = value[2:] - 2.0 * value[1:-1] + value[:-2]
    return torch.mean(diff2**2)


def _axis_angle_to_matrix(torch, rotvec):
    angle = torch.linalg.norm(rotvec, dim=1, keepdim=True).clamp_min(1e-8)
    axis = rotvec / angle
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = torch.zeros_like(x)
    k = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).unsqueeze(0)
    sin = torch.sin(angle).reshape(-1, 1, 1)
    cos = torch.cos(angle).reshape(-1, 1, 1)
    return eye + sin * k + (1.0 - cos) * torch.bmm(k, k)


def _root_up_loss(torch, root_orient, target_frame: str):
    matrices = _axis_angle_to_matrix(torch, root_orient)
    local_up = matrices[:, :, 1]
    target = torch.zeros_like(local_up)
    target[:, target_frame_up_axis(target_frame)] = 1.0
    return torch.mean((local_up - target) ** 2)


def fit_pose_sequence(
    track: NTUTrack,
    betas_np: np.ndarray,
    smplx_model_dir: Path,
    gender: str,
    num_betas: int,
    device: str,
    iterations: int,
    lr: float,
    pose_prior_weight: float,
    root_up_prior_weight: float,
    temporal_smooth_weight: float,
    floor_prior_weight: float,
    target_frame: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    torch, smplx = _import_torch_and_smplx()
    model_path, resolved_gender = resolve_smplx_model_path(smplx_model_dir, gender)
    up_axis = target_frame_up_axis(target_frame)
    targets_np, weights_np, smplx_indices, floor_height = build_pose_targets(
        track.joints,
        track.tracking,
        up_axis=up_axis,
    )
    n_frames = int(targets_np.shape[0])

    model = smplx.create(
        model_path=str(model_path),
        model_type="smplx",
        gender=resolved_gender,
        ext="npz",
        use_pca=False,
        flat_hand_mean=False,
        num_betas=num_betas,
        batch_size=n_frames,
    ).to(device)

    betas_np = np.asarray(betas_np, dtype=np.float32).reshape(-1)
    if betas_np.size < num_betas:
        betas_np = np.pad(betas_np, (0, num_betas - betas_np.size))
    betas = torch.as_tensor(betas_np[:num_betas], dtype=torch.float32, device=device).reshape(1, -1)
    betas = betas.repeat(n_frames, 1)

    targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)
    selected = torch.as_tensor(smplx_indices, dtype=torch.long, device=device)
    zeros = _zero_pose_tensors(torch, n_frames, device)

    with torch.no_grad():
        root_init_np = target_frame_root_init(target_frame)
        root_init_t = torch.as_tensor(root_init_np, dtype=torch.float32, device=device).reshape(1, 3)
        root_init_batch = root_init_t.repeat(n_frames, 1)
        neutral_out = model(
            betas=betas,
            global_orient=root_init_batch,
            body_pose=torch.zeros((n_frames, 63), dtype=torch.float32, device=device),
            transl=torch.zeros((n_frames, 3), dtype=torch.float32, device=device),
            return_verts=False,
            **zeros,
        )
        pelvis_idx_in_targets = next(
            idx for idx, item in enumerate(POSE_TARGETS) if item.smplx_joint == SMPLX_BODY_JOINTS["pelvis"]
        )
        transl_init = targets[:, pelvis_idx_in_targets, :] - neutral_out.joints[:, SMPLX_BODY_JOINTS["pelvis"], :]

    root_orient = root_init_batch.detach().clone().requires_grad_(True)
    body_pose = torch.zeros((n_frames, 63), dtype=torch.float32, device=device, requires_grad=True)
    transl = transl_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            {"params": [root_orient, body_pose], "lr": float(lr)},
            {"params": [transl], "lr": float(lr) * 0.5},
        ]
    )

    final_joint_error = math.nan
    for _ in range(max(0, int(iterations))):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            betas=betas,
            global_orient=root_orient,
            body_pose=body_pose,
            transl=transl,
            return_verts=False,
            **zeros,
        )
        pred = output.joints.index_select(1, selected)
        err = torch.sqrt(torch.sum((pred - targets) ** 2, dim=2) + 1e-8)
        joint_loss = torch.sum(weights * err) / torch.clamp(torch.sum(weights), min=1.0)
        loss = joint_loss
        loss = loss + float(pose_prior_weight) * torch.mean(body_pose**2)
        loss = loss + float(root_up_prior_weight) * _root_up_loss(torch, root_orient, target_frame)
        loss = loss + float(temporal_smooth_weight) * (
            _second_difference_loss(torch, body_pose)
            + 0.2 * _second_difference_loss(torch, root_orient)
            + 2.0 * _second_difference_loss(torch, transl)
        )
        if floor_prior_weight > 0.0:
            foot_indices = torch.as_tensor(
                [SMPLX_BODY_JOINTS["left_ankle"], SMPLX_BODY_JOINTS["right_ankle"], SMPLX_BODY_JOINTS["left_foot"], SMPLX_BODY_JOINTS["right_foot"]],
                dtype=torch.long,
                device=device,
            )
            foot_height = output.joints.index_select(1, foot_indices)[:, :, up_axis]
            floor = torch.as_tensor(float(floor_height), dtype=torch.float32, device=device)
            loss = loss + float(floor_prior_weight) * torch.mean(torch.relu(floor - foot_height) ** 2)
        loss.backward()
        optimizer.step()
        final_joint_error = float(joint_loss.detach().cpu())

    return (
        transl.detach().cpu().numpy().astype(np.float32),
        root_orient.detach().cpu().numpy().astype(np.float32),
        body_pose.detach().cpu().numpy().astype(np.float32),
        {
            "weighted_joint_error_m": final_joint_error,
            "floor_height_m": float(floor_height),
            "up_axis": int(up_axis),
            "fit_joint_count": int(len(smplx_indices)),
        },
    )


def write_amass_like_npz(
    output_path: Path,
    selected: SelectedTrack,
    trans: np.ndarray,
    root_orient: np.ndarray,
    body_pose: np.ndarray,
    betas: np.ndarray,
    gender: str,
    frame_rate_hz: float,
    diagnostics: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(trans.shape[0])
    np.savez_compressed(
        output_path,
        surface_model_type=np.array("smplx"),
        gender=np.array(gender),
        mocap_frame_rate=np.array(float(frame_rate_hz), dtype=np.float32),
        trans=trans.astype(np.float32, copy=False),
        root_orient=root_orient.astype(np.float32, copy=False),
        pose_body=body_pose.astype(np.float32, copy=False),
        pose_hand=np.zeros((n_frames, 90), dtype=np.float32),
        pose_jaw=np.zeros((n_frames, 3), dtype=np.float32),
        pose_eye=np.zeros((n_frames, 6), dtype=np.float32),
        betas=betas.astype(np.float32, copy=False),
        ntu_source_path=np.array(str(selected.source_path)),
        ntu_body_id=np.array(selected.track.body_id),
        ntu_body_rank=np.array(selected.body_rank, dtype=np.int32),
        ntu_setup=np.array(-1 if selected.metadata.setup is None else selected.metadata.setup, dtype=np.int32),
        ntu_camera=np.array(-1 if selected.metadata.camera is None else selected.metadata.camera, dtype=np.int32),
        ntu_performer=np.array(-1 if selected.metadata.performer is None else selected.metadata.performer, dtype=np.int32),
        ntu_replication=np.array(-1 if selected.metadata.replication is None else selected.metadata.replication, dtype=np.int32),
        ntu_action=np.array(-1 if selected.metadata.action is None else selected.metadata.action, dtype=np.int32),
        ntu_conversion_diagnostics=np.array(json.dumps(diagnostics, sort_keys=True)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert NTU RGB+D .skeleton files to AMASS-like SMPL-X .npz files "
            "compatible with scripts/run_amass_to_bsm_csv.py."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/ntu"))
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional single .skeleton file. When set, --input-dir is ignored.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/ntu_smplx_npz"))
    parser.add_argument("--smplx-model-dir", type=Path, default=Path("model/smpl"))
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE_HZ)
    parser.add_argument("--num-betas", type=int, default=16)
    parser.add_argument("--actor-mode", choices=["primary", "all"], default="primary")
    parser.add_argument(
        "--target-frame",
        choices=[TARGET_FRAME_Z_UP, TARGET_FRAME_Y_UP],
        default=TARGET_FRAME_Z_UP,
        help=(
            "World frame for generated SMPL-X params. Default z-up matches "
            "this repo's BSM/Nimble gravity; y-up preserves raw NTU vertical."
        ),
    )
    parser.add_argument(
        "--swap-left-right",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Swap NTU left/right body labels before fitting. Default true because "
            "NTU skeleton x-axis labels are mirrored relative to SMPL-X in the sample files."
        ),
    )
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--performer",
        action="append",
        default=[],
        help="Optional performer filter, e.g. P001 or 1. Can be passed multiple times.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fit-shapes-only",
        action="store_true",
        help="Fit/write performer beta cache from input skeletons, then stop before pose fitting.",
    )
    parser.add_argument("--shape-cache-dir", type=Path, default=None)
    parser.add_argument("--shape-iters", type=int, default=300)
    parser.add_argument("--shape-lr", type=float, default=0.04)
    parser.add_argument("--shape-beta-prior-weight", type=float, default=0.2)
    parser.add_argument("--shape-max-sequences-per-performer", type=int, default=20)
    parser.add_argument("--pose-iters", type=int, default=900)
    parser.add_argument("--pose-lr", type=float, default=0.035)
    parser.add_argument("--pose-prior-weight", type=float, default=0.05)
    parser.add_argument("--root-up-prior-weight", type=float, default=10.0)
    parser.add_argument("--temporal-smooth-weight", type=float, default=0.12)
    parser.add_argument("--floor-prior-weight", type=float, default=2.0)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional conversion summary path. Defaults to <output-dir>/conversion_summary.json.",
    )
    return parser


def collect_shape_records(
    files: Iterable[Path],
    actor_mode: str,
    max_per_performer: int,
    swap_left_right: bool,
) -> dict[str, list[dict[str, float]]]:
    records: dict[str, list[dict[str, float]]] = {}
    for path in files:
        for selected in select_tracks(path, actor_mode=actor_mode):
            track = maybe_swap_track_left_right(selected.track, swap_left_right)
            bucket = records.setdefault(selected.shape_key, [])
            if len(bucket) >= int(max_per_performer):
                continue
            lengths = collect_segment_lengths(track)
            if lengths:
                bucket.append(lengths)
    return records


def load_or_fit_betas(
    shape_key: str,
    shape_targets: dict[str, float],
    args: argparse.Namespace,
    shape_cache_dir: Path,
) -> tuple[np.ndarray, bool]:
    cache_path = shape_cache_dir / f"{shape_key}.npy"
    if cache_path.exists() and not args.force:
        return np.load(cache_path).astype(np.float32), True
    betas = fit_shape_betas(
        segment_targets=shape_targets,
        smplx_model_dir=args.smplx_model_dir,
        gender=args.gender,
        num_betas=args.num_betas,
        device=args.device,
        iterations=args.shape_iters,
        lr=args.shape_lr,
        beta_prior_weight=args.shape_beta_prior_weight,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, betas.astype(np.float32, copy=False))
    return betas, False


def run_conversion(args: argparse.Namespace) -> dict[str, object]:
    input_file = args.input_file.resolve() if args.input_file is not None else None
    input_dir = args.input_dir.resolve() if input_file is None else input_file.parent.resolve()
    output_dir = args.output_dir.resolve()
    if input_file is not None and not input_file.exists():
        raise FileNotFoundError(f"NTU input file not found: {input_file}")
    if input_file is not None and input_file.suffix.lower() != ".skeleton":
        raise ValueError(f"--input-file must point to a .skeleton file: {input_file}")
    if input_file is None and not input_dir.exists():
        raise FileNotFoundError(f"NTU input directory not found: {input_dir}")

    files = [input_file] if input_file is not None else discover_skeleton_files(input_dir, recursive=args.recursive)
    performer_filter = {normalize_performer_key(value) for value in args.performer}
    if performer_filter:
        files = [path for path in files if parse_ntu_metadata(path).performer_key in performer_filter]
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]
    if not files:
        suffix = f" for performers {sorted(performer_filter)}" if performer_filter else ""
        raise FileNotFoundError(f"No .skeleton files found under {input_dir}{suffix}")

    dry_run_items = []
    if args.dry_run:
        for path in files:
            selected = select_tracks(path, actor_mode=args.actor_mode)
            dry_run_items.append(
                {
                    "file": str(path),
                    "selected_tracks": [
                        {
                            "output_stem": item.output_stem,
                            "shape_key": item.shape_key,
                            "body_id": item.track.body_id,
                            "valid_frames": item.track.valid_frames,
                            "motion_score": item.track.motion_score,
                            "action": item.metadata.action_key,
                        }
                        for item in selected
                    ],
                }
            )
        return {"mode": "dry_run", "input_dir": str(input_dir), "files": dry_run_items}

    output_dir.mkdir(parents=True, exist_ok=True)
    shape_cache_dir = (args.shape_cache_dir or (output_dir / "_shape_cache")).resolve()
    shape_records = collect_shape_records(
        files=files,
        actor_mode=args.actor_mode,
        max_per_performer=args.shape_max_sequences_per_performer,
        swap_left_right=args.swap_left_right,
    )
    aggregated_shape_targets = {
        shape_key: aggregate_shape_targets(records) for shape_key, records in shape_records.items()
    }

    betas_by_shape_key: dict[str, np.ndarray] = {}
    shape_cache_hits: dict[str, bool] = {}
    for shape_key, targets in sorted(aggregated_shape_targets.items()):
        betas, cache_hit = load_or_fit_betas(shape_key, targets, args, shape_cache_dir)
        betas_by_shape_key[shape_key] = betas
        shape_cache_hits[shape_key] = cache_hit

    if args.fit_shapes_only:
        summary = {
            "mode": "fit_shapes",
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "shape_cache_dir": str(shape_cache_dir),
            "smplx_model_dir": str(args.smplx_model_dir.resolve()),
            "actor_mode": args.actor_mode,
            "performer_filter": sorted(performer_filter),
            "target_frame": args.target_frame,
            "swap_left_right": bool(args.swap_left_right),
            "num_files": len(files),
            "num_shape_keys": len(aggregated_shape_targets),
            "shape_keys": sorted(aggregated_shape_targets.keys()),
            "shape_cache_hits": shape_cache_hits,
        }
        summary_path = (args.summary_json or (output_dir / "shape_fit_summary.json")).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary["summary_json"] = str(summary_path)
        return summary

    converted = []
    skipped = []
    failed = []
    for path in files:
        try:
            for selected in select_tracks(path, actor_mode=args.actor_mode):
                source_track = maybe_swap_track_left_right(selected.track, args.swap_left_right)
                target_track = transform_track_to_target_frame(source_track, args.target_frame)
                selected_for_output = SelectedTrack(
                    source_path=selected.source_path,
                    metadata=selected.metadata,
                    body_rank=selected.body_rank,
                    track=target_track,
                )
                output_path = output_dir / f"{selected.output_stem}.npz"
                if output_path.exists() and not args.force:
                    skipped.append(str(output_path))
                    continue
                shape_key = selected.shape_key
                betas = betas_by_shape_key.get(shape_key)
                if betas is None:
                    betas = np.zeros(int(args.num_betas), dtype=np.float32)
                trans, root_orient, body_pose, fit_diag = fit_pose_sequence(
                    track=target_track,
                    betas_np=betas,
                    smplx_model_dir=args.smplx_model_dir,
                    gender=args.gender,
                    num_betas=args.num_betas,
                    device=args.device,
                    iterations=args.pose_iters,
                    lr=args.pose_lr,
                    pose_prior_weight=args.pose_prior_weight,
                    root_up_prior_weight=args.root_up_prior_weight,
                    temporal_smooth_weight=args.temporal_smooth_weight,
                    floor_prior_weight=args.floor_prior_weight,
                    target_frame=args.target_frame,
                )
                diagnostics = {
                    **fit_diag,
                    "target_frame": args.target_frame,
                        "shape_key": shape_key,
                        "shape_cache_hit": bool(shape_cache_hits.get(shape_key, False)),
                        "swap_left_right": bool(args.swap_left_right),
                        "valid_frames": target_track.valid_frames,
                    "body_id": target_track.body_id,
                    "body_rank": selected.body_rank,
                }
                write_amass_like_npz(
                    output_path=output_path,
                    selected=selected_for_output,
                    trans=trans,
                    root_orient=root_orient,
                    body_pose=body_pose,
                    betas=betas,
                    gender=args.gender,
                    frame_rate_hz=args.frame_rate,
                    diagnostics=diagnostics,
                )
                converted.append(
                    {
                        "source": str(path),
                        "output": str(output_path),
                        "shape_key": shape_key,
                        "action": selected.metadata.action_key,
                        "frames": int(trans.shape[0]),
                        **fit_diag,
                    }
                )
        except Exception as exc:  # Keep batch conversion moving.
            failed.append({"source": str(path), "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "mode": "convert",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "smplx_model_dir": str(args.smplx_model_dir.resolve()),
        "actor_mode": args.actor_mode,
        "performer_filter": sorted(performer_filter),
        "target_frame": args.target_frame,
        "swap_left_right": bool(args.swap_left_right),
        "frame_rate_hz": float(args.frame_rate),
        "num_files": len(files),
        "num_converted": len(converted),
        "num_skipped": len(skipped),
        "num_failed": len(failed),
        "shape_keys": sorted(aggregated_shape_targets.keys()),
        "converted": converted,
        "skipped": skipped,
        "failed": failed,
    }
    summary_path = (args.summary_json or (output_dir / "conversion_summary.json")).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def main() -> int:
    # Match the rest of this repo on macOS when torch/OpenSim stacks coexist.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_conversion(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

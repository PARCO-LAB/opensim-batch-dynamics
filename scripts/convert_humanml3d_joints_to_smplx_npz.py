#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from convert_ntu_skeleton_to_smplx_npz import (
    TARGET_FRAME_Y_UP,
    TARGET_FRAME_Z_UP,
    _import_torch_and_smplx,
    _second_difference_loss,
    resolve_smplx_model_path,
    target_frame_root_init,
    target_frame_up_axis,
    transform_joints_to_target_frame,
)


DEFAULT_FRAME_RATE_HZ = 20.0
NUM_HUMANML3D_JOINTS = 52

# HumanML3D "joints" are SMPL/SMPL-H style 52-joint positions. The first
# 22 body joints match the SMPL-X body prefix; hands are mapped to SMPL-X names.
HUMANML3D_TO_SMPLX = (
    *[(idx, idx, 1.0) for idx in range(22)],
    (22, 25, 0.45),
    (23, 26, 0.45),
    (24, 27, 0.45),
    (34, 37, 0.45),
    (35, 38, 0.45),
    (36, 39, 0.45),
    (25, 28, 0.45),
    (26, 29, 0.45),
    (27, 30, 0.45),
    (31, 34, 0.45),
    (32, 35, 0.45),
    (33, 36, 0.45),
    (28, 31, 0.45),
    (29, 32, 0.45),
    (30, 33, 0.45),
    (43, 40, 0.45),
    (44, 41, 0.45),
    (45, 42, 0.45),
    (46, 52, 0.45),
    (47, 53, 0.45),
    (48, 54, 0.45),
    (40, 43, 0.45),
    (41, 44, 0.45),
    (42, 45, 0.45),
    (37, 49, 0.45),
    (38, 50, 0.45),
    (39, 51, 0.45),
    (49, 46, 0.45),
    (50, 47, 0.45),
    (51, 48, 0.45),
)

SHAPE_SEGMENTS = (
    (0, 1, 0, 1, 1.0),
    (0, 2, 0, 2, 1.0),
    (1, 4, 1, 4, 1.4),
    (4, 7, 4, 7, 1.4),
    (7, 10, 7, 10, 0.8),
    (2, 5, 2, 5, 1.4),
    (5, 8, 5, 8, 1.4),
    (8, 11, 8, 11, 0.8),
    (0, 3, 0, 3, 0.8),
    (3, 6, 3, 6, 0.8),
    (6, 9, 6, 9, 0.8),
    (9, 12, 9, 12, 0.8),
    (12, 15, 12, 15, 0.8),
    (9, 13, 9, 13, 1.0),
    (13, 16, 13, 16, 1.0),
    (16, 18, 16, 18, 1.2),
    (18, 20, 18, 20, 1.2),
    (9, 14, 9, 14, 1.0),
    (14, 17, 14, 17, 1.0),
    (17, 19, 17, 19, 1.2),
    (19, 21, 19, 21, 1.2),
    (20, 22, 20, 25, 0.25),
    (21, 43, 21, 40, 0.25),
)


@dataclass(frozen=True)
class FitResult:
    trans: np.ndarray
    root_orient: np.ndarray
    pose_body: np.ndarray
    pose_hand: np.ndarray
    betas: np.ndarray
    diagnostics: dict[str, object]


def load_humanml3d_joints(path: Path) -> np.ndarray:
    joints = np.load(path, allow_pickle=False)
    if joints.ndim != 3 or joints.shape[1:] != (NUM_HUMANML3D_JOINTS, 3):
        raise ValueError(f"Expected shape (T, 52, 3), got {joints.shape}: {path}")
    return joints.astype(np.float32, copy=False)


def collect_shape_targets(joints: np.ndarray) -> dict[tuple[int, int, int, int], float]:
    targets: dict[tuple[int, int, int, int], float] = {}
    for human_a, human_b, smplx_a, smplx_b, _weight in SHAPE_SEGMENTS:
        values = np.linalg.norm(joints[:, human_a, :] - joints[:, human_b, :], axis=1)
        values = values[np.isfinite(values)]
        if values.size:
            targets[(human_a, human_b, smplx_a, smplx_b)] = float(np.median(values))
    return targets


def fit_shape_betas_from_humanml3d(
    joints: np.ndarray,
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

    segment_targets = collect_shape_targets(joints)
    if not segment_targets:
        return np.zeros(num_betas, dtype=np.float32)

    betas = torch.zeros((1, num_betas), dtype=torch.float32, device=device, requires_grad=True)
    zeros = {
        "global_orient": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "body_pose": torch.zeros((1, 63), dtype=torch.float32, device=device),
        "left_hand_pose": torch.zeros((1, 45), dtype=torch.float32, device=device),
        "right_hand_pose": torch.zeros((1, 45), dtype=torch.float32, device=device),
        "jaw_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "leye_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "reye_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "transl": torch.zeros((1, 3), dtype=torch.float32, device=device),
    }
    optimizer = torch.optim.Adam([betas], lr=float(lr))
    keys = list(segment_targets.keys())
    target_lengths = torch.as_tensor([segment_targets[key] for key in keys], dtype=torch.float32, device=device)
    weights = torch.as_tensor(
        [next(w for ha, hb, sa, sb, w in SHAPE_SEGMENTS if (ha, hb, sa, sb) == key) for key in keys],
        dtype=torch.float32,
        device=device,
    )

    for _ in range(max(0, int(iterations))):
        optimizer.zero_grad(set_to_none=True)
        out = model(betas=betas, return_verts=False, **zeros)
        pred_lengths = torch.stack([torch.linalg.norm(out.joints[0, key[2]] - out.joints[0, key[3]]) for key in keys])
        loss = torch.mean(weights * (pred_lengths - target_lengths) ** 2)
        loss = loss + float(beta_prior_weight) * torch.mean(betas**2)
        loss.backward()
        optimizer.step()

    return betas.detach().cpu().numpy().reshape(-1).astype(np.float32)


def estimate_floor_height(joints: np.ndarray, up_axis: int) -> float:
    foot_indices = [7, 10, 8, 11]
    values = joints[:, foot_indices, up_axis].reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        values = joints[:, :, up_axis].reshape(-1)
        values = values[np.isfinite(values)]
    return 0.0 if values.size == 0 else float(np.percentile(values, 2.0))


def fit_pose_sequence_from_humanml3d(
    joints: np.ndarray,
    betas_np: np.ndarray,
    smplx_model_dir: Path,
    gender: str,
    num_betas: int,
    device: str,
    iterations: int,
    lr: float,
    pose_prior_weight: float,
    hand_prior_weight: float,
    temporal_smooth_weight: float,
    floor_prior_weight: float,
    target_frame: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    torch, smplx = _import_torch_and_smplx()
    model_path, resolved_gender = resolve_smplx_model_path(smplx_model_dir, gender)
    n_frames = int(joints.shape[0])
    up_axis = target_frame_up_axis(target_frame)
    floor_height = estimate_floor_height(joints, up_axis)

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

    human_indices = [item[0] for item in HUMANML3D_TO_SMPLX]
    smplx_indices = [item[1] for item in HUMANML3D_TO_SMPLX]
    joint_weights = [item[2] for item in HUMANML3D_TO_SMPLX]
    targets = torch.as_tensor(joints[:, human_indices, :], dtype=torch.float32, device=device)
    weights = torch.as_tensor(joint_weights, dtype=torch.float32, device=device).reshape(1, -1)
    selected = torch.as_tensor(smplx_indices, dtype=torch.long, device=device)

    with torch.no_grad():
        root_init = torch.as_tensor(target_frame_root_init(target_frame), dtype=torch.float32, device=device).reshape(1, 3)
        root_init_batch = root_init.repeat(n_frames, 1)
        zeros = {
            "global_orient": root_init_batch,
            "body_pose": torch.zeros((n_frames, 63), dtype=torch.float32, device=device),
            "left_hand_pose": torch.zeros((n_frames, 45), dtype=torch.float32, device=device),
            "right_hand_pose": torch.zeros((n_frames, 45), dtype=torch.float32, device=device),
            "jaw_pose": torch.zeros((n_frames, 3), dtype=torch.float32, device=device),
            "leye_pose": torch.zeros((n_frames, 3), dtype=torch.float32, device=device),
            "reye_pose": torch.zeros((n_frames, 3), dtype=torch.float32, device=device),
            "transl": torch.zeros((n_frames, 3), dtype=torch.float32, device=device),
        }
        neutral = model(betas=betas, return_verts=False, **zeros)
        transl_init = targets[:, 0, :] - neutral.joints[:, 0, :]

    root_orient = root_init_batch.detach().clone().requires_grad_(True)
    body_pose = torch.zeros((n_frames, 63), dtype=torch.float32, device=device, requires_grad=True)
    left_hand_pose = torch.zeros((n_frames, 45), dtype=torch.float32, device=device, requires_grad=True)
    right_hand_pose = torch.zeros((n_frames, 45), dtype=torch.float32, device=device, requires_grad=True)
    transl = transl_init.detach().clone().requires_grad_(True)
    zero_face = {
        "jaw_pose": torch.zeros((n_frames, 3), dtype=torch.float32, device=device),
        "leye_pose": torch.zeros((n_frames, 3), dtype=torch.float32, device=device),
        "reye_pose": torch.zeros((n_frames, 3), dtype=torch.float32, device=device),
    }
    optimizer = torch.optim.Adam(
        [
            {"params": [root_orient, body_pose], "lr": float(lr)},
            {"params": [left_hand_pose, right_hand_pose], "lr": float(lr) * 0.65},
            {"params": [transl], "lr": float(lr) * 0.5},
        ]
    )

    final_joint_error = math.nan
    for _ in range(max(0, int(iterations))):
        optimizer.zero_grad(set_to_none=True)
        out = model(
            betas=betas,
            global_orient=root_orient,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            return_verts=False,
            **zero_face,
        )
        pred = out.joints.index_select(1, selected)
        err = torch.sqrt(torch.sum((pred - targets) ** 2, dim=2) + 1e-8)
        weighted_err = weights * err
        joint_loss = torch.sum(weighted_err) / torch.clamp(torch.sum(weights.expand_as(err)), min=1.0)
        loss = joint_loss
        loss = loss + float(pose_prior_weight) * torch.mean(body_pose**2)
        loss = loss + float(hand_prior_weight) * (torch.mean(left_hand_pose**2) + torch.mean(right_hand_pose**2))
        loss = loss + float(temporal_smooth_weight) * (
            _second_difference_loss(torch, body_pose)
            + 0.5 * _second_difference_loss(torch, left_hand_pose)
            + 0.5 * _second_difference_loss(torch, right_hand_pose)
            + 0.2 * _second_difference_loss(torch, root_orient)
            + 2.0 * _second_difference_loss(torch, transl)
        )
        if floor_prior_weight > 0.0:
            foot_indices = torch.as_tensor([7, 8, 10, 11], dtype=torch.long, device=device)
            foot_height = out.joints.index_select(1, foot_indices)[:, :, up_axis]
            floor = torch.as_tensor(float(floor_height), dtype=torch.float32, device=device)
            loss = loss + float(floor_prior_weight) * torch.mean(torch.relu(floor - foot_height) ** 2)
        loss.backward()
        optimizer.step()
        final_joint_error = float(joint_loss.detach().cpu())

    pose_hand = torch.cat([left_hand_pose, right_hand_pose], dim=1)
    return (
        transl.detach().cpu().numpy().astype(np.float32),
        root_orient.detach().cpu().numpy().astype(np.float32),
        body_pose.detach().cpu().numpy().astype(np.float32),
        pose_hand.detach().cpu().numpy().astype(np.float32),
        {
            "weighted_joint_error_m": final_joint_error,
            "floor_height_m": float(floor_height),
            "up_axis": int(up_axis),
            "fit_joint_count": int(len(HUMANML3D_TO_SMPLX)),
        },
    )


def fit_humanml3d(
    joints: np.ndarray,
    args: argparse.Namespace,
) -> FitResult:
    target_joints = transform_joints_to_target_frame(joints, args.target_frame)
    betas = fit_shape_betas_from_humanml3d(
        joints=target_joints,
        smplx_model_dir=args.smplx_model_dir,
        gender=args.gender,
        num_betas=args.num_betas,
        device=args.device,
        iterations=args.shape_iters,
        lr=args.shape_lr,
        beta_prior_weight=args.shape_beta_prior_weight,
    )
    trans, root_orient, pose_body, pose_hand, fit_diag = fit_pose_sequence_from_humanml3d(
        joints=target_joints,
        betas_np=betas,
        smplx_model_dir=args.smplx_model_dir,
        gender=args.gender,
        num_betas=args.num_betas,
        device=args.device,
        iterations=args.pose_iters,
        lr=args.pose_lr,
        pose_prior_weight=args.pose_prior_weight,
        hand_prior_weight=args.hand_prior_weight,
        temporal_smooth_weight=args.temporal_smooth_weight,
        floor_prior_weight=args.floor_prior_weight,
        target_frame=args.target_frame,
    )
    return FitResult(
        trans=trans,
        root_orient=root_orient,
        pose_body=pose_body,
        pose_hand=pose_hand,
        betas=betas,
        diagnostics={
            **fit_diag,
            "source_format": "HumanML3D/joints",
            "source_shape": list(joints.shape),
            "target_frame": args.target_frame,
        },
    )


def write_smplx_npz(
    output_path: Path,
    input_path: Path,
    result: FitResult,
    gender: str,
    frame_rate_hz: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(result.trans.shape[0])
    np.savez_compressed(
        output_path,
        surface_model_type=np.array("smplx"),
        gender=np.array(gender),
        mocap_frame_rate=np.array(float(frame_rate_hz), dtype=np.float32),
        trans=result.trans.astype(np.float32, copy=False),
        root_orient=result.root_orient.astype(np.float32, copy=False),
        pose_body=result.pose_body.astype(np.float32, copy=False),
        pose_hand=result.pose_hand.astype(np.float32, copy=False),
        pose_jaw=np.zeros((n_frames, 3), dtype=np.float32),
        pose_eye=np.zeros((n_frames, 6), dtype=np.float32),
        betas=result.betas.astype(np.float32, copy=False),
        humanml3d_source_path=np.array(str(input_path)),
        humanml3d_conversion_diagnostics=np.array(json.dumps(result.diagnostics, sort_keys=True)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit SMPL-X params from HumanML3D joints/*.npy (52 joints) and export AMASS-like .npz."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to data/HumanML3D/joints/*.npy")
    parser.add_argument("--output-dir", type=Path, default=Path("data/humanml3d_smplx_npz"))
    parser.add_argument("--smplx-model-dir", type=Path, default=Path("model/smpl"))
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE_HZ)
    parser.add_argument("--target-frame", choices=[TARGET_FRAME_Z_UP, TARGET_FRAME_Y_UP], default=TARGET_FRAME_Z_UP)
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
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args = build_arg_parser().parse_args()
    input_path = args.input.resolve()
    output_path = args.output_dir.resolve() / f"{input_path.stem}.npz"
    if output_path.exists() and not args.force:
        print(json.dumps({"status": "skipped_existing", "output": str(output_path)}, indent=2))
        return 0

    joints = load_humanml3d_joints(input_path)
    result = fit_humanml3d(joints, args)
    write_smplx_npz(output_path, input_path, result, args.gender, args.frame_rate)
    print(
        json.dumps(
            {
                "status": "ok",
                "input": str(input_path),
                "output": str(output_path),
                "frames": int(result.trans.shape[0]),
                **result.diagnostics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

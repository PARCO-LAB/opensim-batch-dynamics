import argparse
from pathlib import Path

import nimblephysics as nimble
import numpy as np
import pandas as pd

from rt_library import (
    binary_classification_metrics,
    get_model_dof_names,
    initialize_rt_state,
    mae,
    per_column_rmse,
    qpid,
    rmse,
    top_k_rmse,
)


BSM_JOINT_NAMES = [
    "walker_knee_r",
    "wrist_l",
    "hip_r",
    "GlenoHumeral_r",
    "elbow_l",
    "hip_l",
    "elbow_r",
    "wrist_r",
    "walker_knee_l",
    "GlenoHumeral_l",
    "ankle_r",
    "ankle_l",
]


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Run real-time 2-stage inverse dynamics against offline CSV.")
    parser.add_argument("--csv", type=Path, default=repo_root / "data" / "AMASS" / "BMLhandball" / "Trial_upper_left_012_poses.csv")
    parser.add_argument("--model", type=Path, default=repo_root / "model" / "bsm" / "bsm.osim")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--drop-joint-prob", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    frame_table = pd.read_csv(args.csv.resolve())
    if args.max_frames is not None:
        frame_table = frame_table.iloc[: args.max_frames].reset_index(drop=True)
    if len(frame_table) < 2:
        raise ValueError("Need at least 2 frames")

    osim = nimble.biomechanics.OpenSimParser.parseOsim(str(args.model.resolve()))
    skeleton = osim.skeleton
    skeleton_gt = osim.skeleton.clone()

    ordered_scales = []
    for body_idx in range(skeleton.getNumBodyNodes()):
        body_name = skeleton.getBodyNode(body_idx).getName()
        ordered_scales += [
            float(frame_table.iloc[0][f"{body_name}_scale_x"]),
            float(frame_table.iloc[0][f"{body_name}_scale_y"]),
            float(frame_table.iloc[0][f"{body_name}_scale_z"]),
        ]
    skeleton.setBodyScales(ordered_scales)
    skeleton_gt.setBodyScales(ordered_scales)
    skeleton.setGravity([0.0, 0.0, -9.81])
    skeleton_gt.setGravity([0.0, 0.0, -9.81])

    dof_names = get_model_dof_names(skeleton)
    q_ref = frame_table[dof_names].to_numpy(dtype=float)
    dq_ref = frame_table[[name + "_vel" for name in dof_names]].to_numpy(dtype=float)
    ddq_ref = frame_table[[name + "_acc" for name in dof_names]].to_numpy(dtype=float)
    tau_ref = frame_table[[name + "_tau" for name in dof_names]].to_numpy(dtype=float)
    time_ref = frame_table["time"].to_numpy(dtype=float)

    left_force_ref = np.zeros((len(frame_table), 3), dtype=float)
    right_force_ref = np.zeros((len(frame_table), 3), dtype=float)
    for body_name in ["calcn_l", "toes_l", "talus_l"]:
        cols = [f"{body_name}_grf_x", f"{body_name}_grf_y", f"{body_name}_grf_z"]
        if all(col in frame_table.columns for col in cols):
            left_force_ref += frame_table[cols].to_numpy(dtype=float)
    for body_name in ["calcn_r", "toes_r", "talus_r"]:
        cols = [f"{body_name}_grf_x", f"{body_name}_grf_y", f"{body_name}_grf_z"]
        if all(col in frame_table.columns for col in cols):
            right_force_ref += frame_table[cols].to_numpy(dtype=float)

    left_contact_ref = np.zeros(len(frame_table), dtype=bool)
    right_contact_ref = np.zeros(len(frame_table), dtype=bool)
    for body_name in ["calcn_l", "toes_l", "talus_l"]:
        col = f"{body_name}_contact"
        if col in frame_table.columns:
            left_contact_ref |= frame_table[col].to_numpy(dtype=float) > 0.5
    for body_name in ["calcn_r", "toes_r", "talus_r"]:
        col = f"{body_name}_contact"
        if col in frame_table.columns:
            right_contact_ref |= frame_table[col].to_numpy(dtype=float) > 0.5

    skeleton.setPositions(q_ref[0])
    skeleton.setVelocities(dq_ref[0])
    skeleton.setAccelerations(ddq_ref[0])
    skeleton.setControlForces(np.concatenate([np.zeros(6, dtype=float), tau_ref[0, 6:]]))
    skeleton_gt.setPositions(q_ref[0])
    skeleton_gt.setVelocities(dq_ref[0])

    joints = [skeleton.getJoint(name) for name in BSM_JOINT_NAMES]
    joints_gt = [skeleton_gt.getJoint(name) for name in BSM_JOINT_NAMES]

    rt_state = initialize_rt_state(
        skeleton,
        q=q_ref[0],
        dq=dq_ref[0],
        ddq=ddq_ref[0],
        tau=tau_ref[0, 6:],
        tau_full=tau_ref[0],
        root_residual=tau_ref[0, :6],
        contact_state={"left": bool(left_contact_ref[0]), "right": bool(right_contact_ref[0])},
    )

    rng = np.random.default_rng(args.seed)
    q_rt = []
    dq_rt = []
    ddq_rt = []
    tau_rt = []
    left_force_rt = []
    right_force_rt = []
    left_contact_rt = []
    right_contact_rt = []
    mpjpe_rt = []
    dyn_residual_rt = []

    for frame_idx in range(1, len(frame_table)):
        dt = float(time_ref[frame_idx] - time_ref[frame_idx - 1])
        skeleton_gt.setPositions(q_ref[frame_idx])
        skeleton_gt.setVelocities(dq_ref[frame_idx])

        x_t = np.array(skeleton_gt.getJointWorldPositions(joints_gt), dtype=float).reshape(-1, 3)
        x_clean = x_t.copy()

        if args.noise_std > 0.0:
            x_t += rng.normal(0.0, float(args.noise_std), size=x_t.shape)
        if args.drop_joint_prob > 0.0:
            drop_mask = rng.random(x_t.shape[0]) < float(args.drop_joint_prob)
            x_t[drop_mask, :] = np.nan

        result = qpid(
            skeleton=skeleton,
            x_t=x_t.reshape(-1),
            dt=dt,
            mu=args.mu,
            measurement_joints=joints,
            state=rt_state,
        )
        if result is None:
            raise RuntimeError(f"qpid failed at frame {frame_idx}")

        rt_state = result["state"]
        q_rt.append(result["q"])
        dq_rt.append(result["dq"])
        ddq_rt.append(result["ddq"])
        tau_rt.append(result["tau_full"])
        left_force_rt.append(result["foot_forces"]["left"])
        right_force_rt.append(result["foot_forces"]["right"])
        left_contact_rt.append(bool(result["contact_state"]["left"]))
        right_contact_rt.append(bool(result["contact_state"]["right"]))
        dyn_residual_rt.append(float(result["dynamics_residual_norm"]))

        x_est = np.array(skeleton.getJointWorldPositions(joints), dtype=float).reshape(-1, 3)
        mpjpe_rt.append(float(np.mean(np.linalg.norm(x_est - x_clean, axis=1))))

    q_rt = np.array(q_rt, dtype=float)
    dq_rt = np.array(dq_rt, dtype=float)
    ddq_rt = np.array(ddq_rt, dtype=float)
    tau_rt = np.array(tau_rt, dtype=float)
    left_force_rt = np.array(left_force_rt, dtype=float)
    right_force_rt = np.array(right_force_rt, dtype=float)
    left_contact_rt = np.array(left_contact_rt, dtype=bool)
    right_contact_rt = np.array(right_contact_rt, dtype=bool)

    q_ref = q_ref[1 : 1 + len(q_rt)]
    dq_ref = dq_ref[1 : 1 + len(dq_rt)]
    ddq_ref = ddq_ref[1 : 1 + len(ddq_rt)]
    tau_ref = tau_ref[1 : 1 + len(tau_rt)]
    left_force_ref = left_force_ref[1 : 1 + len(left_force_rt)]
    right_force_ref = right_force_ref[1 : 1 + len(right_force_rt)]
    left_contact_ref = left_contact_ref[1 : 1 + len(left_contact_rt)]
    right_contact_ref = right_contact_ref[1 : 1 + len(right_contact_rt)]

    print(f"frames: {len(q_rt)}")
    print(f"noise_std_m: {args.noise_std:.6f}")
    print(f"drop_joint_prob: {args.drop_joint_prob:.6f}")
    print(f"mpjpe_m: mean={np.mean(mpjpe_rt):.6f} max={np.max(mpjpe_rt):.6f}")
    print(f"dyn_residual_norm: mean={np.mean(dyn_residual_rt):.6f} max={np.max(dyn_residual_rt):.6f}")
    print()
    print(f"q_rmse: {rmse(q_rt, q_ref):.6f}")
    print(f"q_mae: {mae(q_rt, q_ref):.6f}")
    print(f"dq_rmse: {rmse(dq_rt, dq_ref):.6f}")
    print(f"dq_mae: {mae(dq_rt, dq_ref):.6f}")
    print(f"ddq_rmse: {rmse(ddq_rt, ddq_ref):.6f}")
    print(f"ddq_mae: {mae(ddq_rt, ddq_ref):.6f}")
    print(f"tau_full_rmse: {rmse(tau_rt, tau_ref):.6f}")
    print(f"tau_full_mae: {mae(tau_rt, tau_ref):.6f}")
    print(f"tau_actuated_rmse: {rmse(tau_rt[:, 6:], tau_ref[:, 6:]):.6f}")
    print(f"tau_actuated_mae: {mae(tau_rt[:, 6:], tau_ref[:, 6:]):.6f}")
    print(f"left_grf_rmse: {rmse(left_force_rt, left_force_ref):.6f}")
    print(f"right_grf_rmse: {rmse(right_force_rt, right_force_ref):.6f}")

    left_metrics = binary_classification_metrics(left_contact_rt, left_contact_ref)
    right_metrics = binary_classification_metrics(right_contact_rt, right_contact_ref)
    print(
        "left_contact:"
        f" acc={left_metrics['accuracy']:.4f}"
        f" prec={left_metrics['precision']:.4f}"
        f" rec={left_metrics['recall']:.4f}"
        f" f1={left_metrics['f1']:.4f}"
    )
    print(
        "right_contact:"
        f" acc={right_metrics['accuracy']:.4f}"
        f" prec={right_metrics['precision']:.4f}"
        f" rec={right_metrics['recall']:.4f}"
        f" f1={right_metrics['f1']:.4f}"
    )
    print()

    print("worst_q_rmse:")
    for name, value in top_k_rmse(q_rt, q_ref, dof_names, k=8):
        print(f"  {name}: {value:.6f}")

    print("worst_tau_rmse:")
    for name, value in top_k_rmse(tau_rt, tau_ref, dof_names, k=8):
        print(f"  {name}: {value:.6f}")

    print("left_grf_axis_rmse:")
    for name, value in per_column_rmse(left_force_rt, left_force_ref, ["fx", "fy", "fz"]):
        print(f"  {name}: {value:.6f}")

    print("right_grf_axis_rmse:")
    for name, value in per_column_rmse(right_force_rt, right_force_ref, ["fx", "fy", "fz"]):
        print(f"  {name}: {value:.6f}")

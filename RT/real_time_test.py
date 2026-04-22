import argparse
import time
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

METRIC_EXCLUDE_PREFIXES = (
    "ankle_angle_",
    "subtalar_angle_",
    "head_",
    "wrist_",
    "pro_sup_",
)


def include_in_precision_metrics(dof_name: str) -> bool:
    return not any(dof_name.startswith(prefix) for prefix in METRIC_EXCLUDE_PREFIXES)


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
    parser.add_argument("--stage1-kin-filter", dest="stage1_kin_filter", action="store_true", help="Enable causal alpha-beta-gamma filtering of q/dq/ddq before Stage 2")
    parser.add_argument("--no-stage1-kin-filter", dest="stage1_kin_filter", action="store_false", help="Disable causal Stage 1 kinematic filtering")
    parser.set_defaults(stage1_kin_filter=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path where the realtime reconstructed sequence is written as CSV",
    )
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
    solve_time_rt = []
    left_wrench_rt = []
    right_wrench_rt = []

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

        t0 = time.perf_counter()
        result = qpid(
            skeleton=skeleton,
            x_t=x_t.reshape(-1),
            dt=dt,
            mu=args.mu,
            measurement_joints=joints,
            state=rt_state,
            use_stage1_kin_filter=args.stage1_kin_filter,
        )
        solve_time_rt.append(time.perf_counter() - t0)
        if result is None:
            raise RuntimeError(f"qpid failed at frame {frame_idx}")

        rt_state = result["state"]
        q_rt.append(result["q"])
        dq_rt.append(result["dq"])
        ddq_rt.append(result["ddq"])
        tau_rt.append(result["tau_full"])
        left_force_rt.append(result["foot_forces"]["left"])
        right_force_rt.append(result["foot_forces"]["right"])
        left_wrench_rt.append(result["foot_wrenches"]["left"])
        right_wrench_rt.append(result["foot_wrenches"]["right"])
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
    solve_time_rt = np.array(solve_time_rt, dtype=float)
    left_wrench_rt = np.array(left_wrench_rt, dtype=float)
    right_wrench_rt = np.array(right_wrench_rt, dtype=float)

    frame_values = frame_table["frame"].to_numpy(dtype=float) if "frame" in frame_table.columns else np.arange(len(frame_table), dtype=float)
    q_ref = q_ref[1 : 1 + len(q_rt)]
    dq_ref = dq_ref[1 : 1 + len(dq_rt)]
    ddq_ref = ddq_ref[1 : 1 + len(ddq_rt)]
    tau_ref = tau_ref[1 : 1 + len(tau_rt)]
    time_ref = time_ref[1 : 1 + len(q_rt)]
    frame_values = frame_values[1 : 1 + len(q_rt)]
    left_force_ref = left_force_ref[1 : 1 + len(left_force_rt)]
    right_force_ref = right_force_ref[1 : 1 + len(right_force_rt)]
    left_contact_ref = left_contact_ref[1 : 1 + len(left_contact_rt)]
    right_contact_ref = right_contact_ref[1 : 1 + len(right_contact_rt)]
    metric_mask = np.array([include_in_precision_metrics(name) for name in dof_names], dtype=bool)
    metric_dof_names = [name for name, keep in zip(dof_names, metric_mask) if keep]
    metric_q_rt = q_rt[:, metric_mask]
    metric_q_ref = q_ref[:, metric_mask]
    metric_dq_rt = dq_rt[:, metric_mask]
    metric_dq_ref = dq_ref[:, metric_mask]
    metric_ddq_rt = ddq_rt[:, metric_mask]
    metric_ddq_ref = ddq_ref[:, metric_mask]
    metric_tau_rt = tau_rt[:, metric_mask]
    metric_tau_ref = tau_ref[:, metric_mask]
    metric_act_mask = metric_mask[6:] if len(metric_mask) > 6 else np.zeros(0, dtype=bool)
    metric_tau_act_rt = tau_rt[:, 6:][:, metric_act_mask] if len(metric_mask) > 6 else metric_tau_rt
    metric_tau_act_ref = tau_ref[:, 6:][:, metric_act_mask] if len(metric_mask) > 6 else metric_tau_ref
    tau_jerk_rt = np.diff(metric_tau_act_rt, axis=0) if len(metric_tau_act_rt) > 1 else np.zeros((0, metric_tau_act_rt.shape[1]), dtype=float)
    tau_jerk_ref = np.diff(metric_tau_act_ref, axis=0) if len(metric_tau_act_ref) > 1 else np.zeros((0, metric_tau_act_ref.shape[1]), dtype=float)

    if args.output_csv is not None:
        output_csv = args.output_csv.resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        rt_columns = {
            "frame": frame_values,
            "time": time_ref,
        }

        for metadata_col in ["subject_mass_kg", "subject_height_m"]:
            if metadata_col in frame_table.columns:
                rt_columns[metadata_col] = frame_table[metadata_col].iloc[1 : 1 + len(q_rt)].to_numpy(dtype=float)

        scale_cols = [col for col in frame_table.columns if "_scale_" in col]
        for scale_col in scale_cols:
            rt_columns[scale_col] = frame_table[scale_col].iloc[1 : 1 + len(q_rt)].to_numpy(dtype=float)

        for idx, dof_name in enumerate(dof_names):
            rt_columns[dof_name] = q_rt[:, idx]
            rt_columns[dof_name + "_vel"] = dq_rt[:, idx]
            rt_columns[dof_name + "_acc"] = ddq_rt[:, idx]
            rt_columns[dof_name + "_tau"] = tau_rt[:, idx]

        rt_columns["left_grf_x"] = left_force_rt[:, 0]
        rt_columns["left_grf_y"] = left_force_rt[:, 1]
        rt_columns["left_grf_z"] = left_force_rt[:, 2]
        rt_columns["right_grf_x"] = right_force_rt[:, 0]
        rt_columns["right_grf_y"] = right_force_rt[:, 1]
        rt_columns["right_grf_z"] = right_force_rt[:, 2]
        rt_columns["grf_total_x"] = left_force_rt[:, 0] + right_force_rt[:, 0]
        rt_columns["grf_total_y"] = left_force_rt[:, 1] + right_force_rt[:, 1]
        rt_columns["grf_total_z"] = left_force_rt[:, 2] + right_force_rt[:, 2]
        rt_columns["left_contact"] = left_contact_rt.astype(float)
        rt_columns["right_contact"] = right_contact_rt.astype(float)

        rt_columns["left_wrench_fx"] = left_wrench_rt[:, 0]
        rt_columns["left_wrench_fy"] = left_wrench_rt[:, 1]
        rt_columns["left_wrench_fz"] = left_wrench_rt[:, 2]
        rt_columns["left_wrench_mx"] = left_wrench_rt[:, 3]
        rt_columns["left_wrench_my"] = left_wrench_rt[:, 4]
        rt_columns["left_wrench_mz"] = left_wrench_rt[:, 5]
        rt_columns["right_wrench_fx"] = right_wrench_rt[:, 0]
        rt_columns["right_wrench_fy"] = right_wrench_rt[:, 1]
        rt_columns["right_wrench_fz"] = right_wrench_rt[:, 2]
        rt_columns["right_wrench_mx"] = right_wrench_rt[:, 3]
        rt_columns["right_wrench_my"] = right_wrench_rt[:, 4]
        rt_columns["right_wrench_mz"] = right_wrench_rt[:, 5]

        rt_columns["mpjpe_m"] = np.array(mpjpe_rt, dtype=float)
        rt_columns["dynamics_residual_norm"] = np.array(dyn_residual_rt, dtype=float)
        rt_columns["solve_time_ms"] = 1000.0 * solve_time_rt
        rt_columns["input_noise_std_m"] = np.full(len(q_rt), float(args.noise_std), dtype=float)
        rt_columns["input_drop_joint_prob"] = np.full(len(q_rt), float(args.drop_joint_prob), dtype=float)
        rt_columns["input_mu"] = np.full(len(q_rt), float(args.mu), dtype=float)
        rt_columns["input_stage1_kin_filter"] = np.full(len(q_rt), float(args.stage1_kin_filter), dtype=float)

        rt_table = pd.DataFrame(rt_columns)
        rt_table.to_csv(output_csv, index=False)

    print(f"frames: {len(q_rt)}")
    print(f"noise_std_m: {args.noise_std:.6f}")
    print(f"drop_joint_prob: {args.drop_joint_prob:.6f}")
    print(f"stage1_kin_filter: {int(args.stage1_kin_filter)}")
    print(f"mpjpe_m: mean={np.mean(mpjpe_rt):.6f} max={np.max(mpjpe_rt):.6f}")
    print(f"dyn_residual_norm: mean={np.mean(dyn_residual_rt):.6f} max={np.max(dyn_residual_rt):.6f}")
    print(f"solve_time_ms: mean={1000.0 * np.mean(solve_time_rt):.6f} p95={1000.0 * np.percentile(solve_time_rt, 95.0):.6f}")
    print(f"precision_metric_dofs: {len(metric_dof_names)}/{len(dof_names)} (excluding ankle/head/wrist-related angles)")
    print()
    print(f"q_rmse: {rmse(metric_q_rt, metric_q_ref):.6f}")
    print(f"q_mae: {mae(metric_q_rt, metric_q_ref):.6f}")
    print(f"dq_rmse: {rmse(metric_dq_rt, metric_dq_ref):.6f}")
    print(f"dq_mae: {mae(metric_dq_rt, metric_dq_ref):.6f}")
    print(f"ddq_rmse: {rmse(metric_ddq_rt, metric_ddq_ref):.6f}")
    print(f"ddq_mae: {mae(metric_ddq_rt, metric_ddq_ref):.6f}")
    print(f"tau_full_rmse: {rmse(metric_tau_rt, metric_tau_ref):.6f}")
    print(f"tau_full_mae: {mae(metric_tau_rt, metric_tau_ref):.6f}")
    print(f"tau_actuated_rmse: {rmse(metric_tau_act_rt, metric_tau_act_ref):.6f}")
    print(f"tau_actuated_mae: {mae(metric_tau_act_rt, metric_tau_act_ref):.6f}")
    if tau_jerk_rt.size > 0:
        print(f"tau_actuated_jerk_rmse: {rmse(tau_jerk_rt, tau_jerk_ref):.6f}")
        print(f"tau_actuated_jerk_l2_mean: {float(np.mean(np.linalg.norm(tau_jerk_rt, axis=1))):.6f}")
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
    for name, value in top_k_rmse(metric_q_rt, metric_q_ref, metric_dof_names, k=8):
        print(f"  {name}: {value:.6f}")

    print("worst_tau_rmse:")
    for name, value in top_k_rmse(metric_tau_rt, metric_tau_ref, metric_dof_names, k=8):
        print(f"  {name}: {value:.6f}")

    print("left_grf_axis_rmse:")
    for name, value in per_column_rmse(left_force_rt, left_force_ref, ["fx", "fy", "fz"]):
        print(f"  {name}: {value:.6f}")

    print("right_grf_axis_rmse:")
    for name, value in per_column_rmse(right_force_rt, right_force_ref, ["fx", "fy", "fz"]):
        print(f"  {name}: {value:.6f}")

    if args.output_csv is not None:
        print()
        print(f"realtime_csv: {args.output_csv.resolve()}")

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import nimblephysics as nimble
import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opensim_batch_dynamics.metrics import (  # noqa: E402
    binary_classification_metrics,
    include_in_precision_metrics,
    mae,
    per_column_rmse,
    rmse,
    top_k_rmse,
)
from opensim_batch_dynamics.realtime_dynamics import (  # noqa: E402
    RealtimeConfig,
    RealtimeWorkspace,
    get_model_dof_names,
    initialize_rt_state,
    qpid,
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


def _set_positions_safely(skeleton, positions: np.ndarray) -> None:
    for idx, value in enumerate(np.asarray(positions, dtype=float).reshape(-1)):
        skeleton.setPosition(idx, float(value))


def _set_velocities_safely(skeleton, velocities: np.ndarray) -> None:
    for idx, value in enumerate(np.asarray(velocities, dtype=float).reshape(-1)):
        skeleton.setVelocity(idx, float(value))


def _set_accelerations_safely(skeleton, accelerations: np.ndarray) -> None:
    for idx, value in enumerate(np.asarray(accelerations, dtype=float).reshape(-1)):
        skeleton.setAcceleration(idx, float(value))


def _set_control_forces_safely(skeleton, forces: np.ndarray) -> None:
    for idx, value in enumerate(np.asarray(forces, dtype=float).reshape(-1)):
        skeleton.setControlForce(idx, float(value))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path where the realtime reconstructed sequence is written as CSV")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional path where aggregate metrics are written as JSON")
    parser.add_argument(
        "--init-policy",
        choices=("offline_first_frame", "keypoint_bootstrap", "zero_velocity"),
        default="offline_first_frame",
        help="Initial state policy for the realtime sequence",
    )
    return parser.parse_args(argv)


def load_reference_csv(csv_path: Path, max_frames: int | None = None) -> pd.DataFrame:
    frame_table = pd.read_csv(csv_path.resolve())
    if max_frames is not None:
        frame_table = frame_table.iloc[: max_frames].reset_index(drop=True)
    if len(frame_table) < 2:
        raise ValueError("Need at least 2 frames")
    return frame_table


def configure_skeletons(model_path: Path, frame_table: pd.DataFrame) -> tuple[nimble.dynamics.Skeleton, nimble.dynamics.Skeleton]:
    osim = nimble.biomechanics.OpenSimParser.parseOsim(str(model_path.resolve()))
    skeleton = osim.skeleton
    skeleton_gt = osim.skeleton.clone()
    # ponytail: body-scale mutation crashes on this Nimble build; keep the parsed model
    # and export scale metadata separately instead of mutating the skeleton here.
    return skeleton, skeleton_gt


def extract_reference_signals(frame_table: pd.DataFrame, dof_names: list[str]) -> dict[str, np.ndarray]:
    signals = {
        "q": frame_table[dof_names].to_numpy(dtype=float),
        "dq": frame_table[[name + "_vel" for name in dof_names]].to_numpy(dtype=float),
        "ddq": frame_table[[name + "_acc" for name in dof_names]].to_numpy(dtype=float),
        "tau": frame_table[[name + "_tau" for name in dof_names]].to_numpy(dtype=float),
        "time": frame_table["time"].to_numpy(dtype=float),
    }

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

    signals["left_force"] = left_force_ref
    signals["right_force"] = right_force_ref
    signals["left_contact"] = left_contact_ref
    signals["right_contact"] = right_contact_ref
    signals["frame"] = frame_table["frame"].to_numpy(dtype=float) if "frame" in frame_table.columns else np.arange(len(frame_table), dtype=float)
    return signals


def _extract_subject_scalars(frame_table: pd.DataFrame) -> tuple[float, float]:
    mass_kg = float(frame_table["subject_mass_kg"].iloc[0]) if "subject_mass_kg" in frame_table.columns else float("nan")
    height_m = float(frame_table["subject_height_m"].iloc[0]) if "subject_height_m" in frame_table.columns else float("nan")
    return mass_kg, height_m


def bootstrap_initial_state(
    skeleton,
    signals: dict[str, np.ndarray],
    config: RealtimeConfig,
    first_measurement: np.ndarray | None = None,
) -> tuple[object, list[object], RealtimeWorkspace, object]:
    joints = [skeleton.getJoint(name) for name in BSM_JOINT_NAMES]
    workspace = RealtimeWorkspace()

    if config.init_policy == "offline_first_frame":
        _set_positions_safely(skeleton, signals["q"][0])
        _set_velocities_safely(skeleton, signals["dq"][0])
        _set_accelerations_safely(skeleton, signals["ddq"][0])
        _set_control_forces_safely(skeleton, np.concatenate([np.zeros(6, dtype=float), signals["tau"][0, 6:]]))
        rt_state = initialize_rt_state(
            skeleton,
            q=signals["q"][0],
            dq=signals["dq"][0],
            ddq=signals["ddq"][0],
            tau=signals["tau"][0, 6:],
            tau_full=signals["tau"][0],
            root_residual=signals["tau"][0, :6],
            contact_state={"left": bool(signals["left_contact"][0]), "right": bool(signals["right_contact"][0])},
        )
        return rt_state, joints, workspace, skeleton

    if config.init_policy == "zero_velocity":
        _set_positions_safely(skeleton, signals["q"][0])
        _set_velocities_safely(skeleton, np.zeros_like(signals["dq"][0], dtype=float))
        _set_accelerations_safely(skeleton, np.zeros_like(signals["ddq"][0], dtype=float))
        _set_control_forces_safely(skeleton, np.zeros_like(signals["tau"][0], dtype=float))
        rt_state = initialize_rt_state(
            skeleton,
            q=signals["q"][0],
            dq=np.zeros_like(signals["dq"][0]),
            ddq=np.zeros_like(signals["ddq"][0]),
            tau=np.zeros_like(signals["tau"][0, 6:]),
            tau_full=np.zeros_like(signals["tau"][0]),
            root_residual=np.zeros(6, dtype=float),
            contact_state={"left": False, "right": False},
        )
        return rt_state, joints, workspace, skeleton

    if config.init_policy == "keypoint_bootstrap":
        if first_measurement is None:
            raise ValueError("first_measurement required for keypoint_bootstrap")
        bootstrap = qpid(
            skeleton=skeleton,
            x_t=first_measurement.reshape(-1),
            measurement_joints=joints,
            state=None,
            config=config,
            workspace=workspace,
        )
        if bootstrap is None:
            raise RuntimeError("keypoint_bootstrap failed on first frame")
        return bootstrap["state"], joints, workspace, skeleton

    raise ValueError(f"Unknown init policy: {config.init_policy}")


def run_realtime_sequence(
    args: argparse.Namespace,
    frame_table: pd.DataFrame,
    skeleton,
    skeleton_gt,
    dof_names: list[str],
    signals: dict[str, np.ndarray],
    joints: list[object],
    rt_state,
    workspace: RealtimeWorkspace,
    config: RealtimeConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    joints_gt = [skeleton_gt.getJoint(name) for name in BSM_JOINT_NAMES]

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
    root_residual_rt = []

    subject_mass_kg, subject_height_m = _extract_subject_scalars(frame_table)

    for frame_idx in range(1, len(frame_table)):
        dt = float(signals["time"][frame_idx] - signals["time"][frame_idx - 1])
        _set_positions_safely(skeleton_gt, signals["q"][frame_idx])
        _set_velocities_safely(skeleton_gt, signals["dq"][frame_idx])

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
            measurement_joints=joints,
            state=rt_state,
            config=config,
            workspace=workspace,
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
        root_residual_rt.append(result["root_residual"])
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
    root_residual_rt = np.array(root_residual_rt, dtype=float)

    q_ref = signals["q"][1 : 1 + len(q_rt)]
    dq_ref = signals["dq"][1 : 1 + len(dq_rt)]
    ddq_ref = signals["ddq"][1 : 1 + len(ddq_rt)]
    tau_ref = signals["tau"][1 : 1 + len(tau_rt)]
    time_ref = signals["time"][1 : 1 + len(q_rt)]
    frame_values = signals["frame"][1 : 1 + len(q_rt)]
    left_force_ref = signals["left_force"][1 : 1 + len(left_force_rt)]
    right_force_ref = signals["right_force"][1 : 1 + len(right_force_rt)]
    left_contact_ref = signals["left_contact"][1 : 1 + len(left_contact_rt)]
    right_contact_ref = signals["right_contact"][1 : 1 + len(right_contact_rt)]

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

    return {
        "q_rt": q_rt,
        "dq_rt": dq_rt,
        "ddq_rt": ddq_rt,
        "tau_rt": tau_rt,
        "left_force_rt": left_force_rt,
        "right_force_rt": right_force_rt,
        "left_contact_rt": left_contact_rt,
        "right_contact_rt": right_contact_rt,
        "left_wrench_rt": left_wrench_rt,
        "right_wrench_rt": right_wrench_rt,
        "mpjpe_rt": np.array(mpjpe_rt, dtype=float),
        "dyn_residual_rt": np.array(dyn_residual_rt, dtype=float),
        "solve_time_rt": solve_time_rt,
        "q_ref": q_ref,
        "dq_ref": dq_ref,
        "ddq_ref": ddq_ref,
        "tau_ref": tau_ref,
        "time_ref": time_ref,
        "frame_values": frame_values,
        "left_force_ref": left_force_ref,
        "right_force_ref": right_force_ref,
        "left_contact_ref": left_contact_ref,
        "right_contact_ref": right_contact_ref,
        "subject_mass_kg": subject_mass_kg,
        "subject_height_m": subject_height_m,
        "root_residual_rt": root_residual_rt,
        "metric_mask": metric_mask,
        "metric_dof_names": metric_dof_names,
        "metric_q_rt": metric_q_rt,
        "metric_q_ref": metric_q_ref,
        "metric_dq_rt": metric_dq_rt,
        "metric_dq_ref": metric_dq_ref,
        "metric_ddq_rt": metric_ddq_rt,
        "metric_ddq_ref": metric_ddq_ref,
        "metric_tau_rt": metric_tau_rt,
        "metric_tau_ref": metric_tau_ref,
        "metric_tau_act_rt": metric_tau_act_rt,
        "metric_tau_act_ref": metric_tau_act_ref,
        "tau_jerk_rt": tau_jerk_rt,
        "tau_jerk_ref": tau_jerk_ref,
        "rt_state": rt_state,
    }


def write_realtime_csv(args: argparse.Namespace, frame_table: pd.DataFrame, dof_names: list[str], results: dict[str, object]) -> Path:
    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    subject_mass_kg = float(results.get("subject_mass_kg", float("nan")))
    subject_height_m = float(results.get("subject_height_m", float("nan")))
    body_weight_n = subject_mass_kg * 9.81 if np.isfinite(subject_mass_kg) and subject_mass_kg > 0.0 else float("nan")
    body_weight_height_nm = body_weight_n * subject_height_m if np.isfinite(body_weight_n) and np.isfinite(subject_height_m) and subject_height_m > 0.0 else float("nan")

    rt_columns = {
        "frame": results["frame_values"],
        "time": results["time_ref"],
    }
    for metadata_col in ["subject_mass_kg", "subject_height_m"]:
        if metadata_col in frame_table.columns:
            rt_columns[metadata_col] = frame_table[metadata_col].iloc[1 : 1 + len(results["q_rt"])].to_numpy(dtype=float)

    scale_cols = [col for col in frame_table.columns if "_scale_" in col]
    for scale_col in scale_cols:
        rt_columns[scale_col] = frame_table[scale_col].iloc[1 : 1 + len(results["q_rt"])].to_numpy(dtype=float)

    for idx, dof_name in enumerate(dof_names):
        rt_columns[dof_name] = results["q_rt"][:, idx]
        rt_columns[dof_name + "_vel"] = results["dq_rt"][:, idx]
        rt_columns[dof_name + "_acc"] = results["ddq_rt"][:, idx]
        rt_columns[dof_name + "_tau"] = results["tau_rt"][:, idx]
        rt_columns[dof_name + "_tau_nm_per_kg"] = results["tau_rt"][:, idx] / subject_mass_kg if np.isfinite(subject_mass_kg) and subject_mass_kg > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)

    rt_columns["left_grf_x"] = results["left_force_rt"][:, 0]
    rt_columns["left_grf_y"] = results["left_force_rt"][:, 1]
    rt_columns["left_grf_z"] = results["left_force_rt"][:, 2]
    rt_columns["right_grf_x"] = results["right_force_rt"][:, 0]
    rt_columns["right_grf_y"] = results["right_force_rt"][:, 1]
    rt_columns["right_grf_z"] = results["right_force_rt"][:, 2]
    rt_columns["grf_total_x"] = results["left_force_rt"][:, 0] + results["right_force_rt"][:, 0]
    rt_columns["grf_total_y"] = results["left_force_rt"][:, 1] + results["right_force_rt"][:, 1]
    rt_columns["grf_total_z"] = results["left_force_rt"][:, 2] + results["right_force_rt"][:, 2]
    rt_columns["left_grf_bw_x"] = results["left_force_rt"][:, 0] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["left_grf_bw_y"] = results["left_force_rt"][:, 1] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["left_grf_bw_z"] = results["left_force_rt"][:, 2] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["right_grf_bw_x"] = results["right_force_rt"][:, 0] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["right_grf_bw_y"] = results["right_force_rt"][:, 1] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["right_grf_bw_z"] = results["right_force_rt"][:, 2] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["grf_total_bw_x"] = rt_columns["grf_total_x"] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["grf_total_bw_y"] = rt_columns["grf_total_y"] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["grf_total_bw_z"] = rt_columns["grf_total_z"] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["left_contact"] = results["left_contact_rt"].astype(float)
    rt_columns["right_contact"] = results["right_contact_rt"].astype(float)

    rt_columns["left_wrench_fx"] = results["left_wrench_rt"][:, 0]
    rt_columns["left_wrench_fy"] = results["left_wrench_rt"][:, 1]
    rt_columns["left_wrench_fz"] = results["left_wrench_rt"][:, 2]
    rt_columns["left_wrench_mx"] = results["left_wrench_rt"][:, 3]
    rt_columns["left_wrench_my"] = results["left_wrench_rt"][:, 4]
    rt_columns["left_wrench_mz"] = results["left_wrench_rt"][:, 5]
    rt_columns["right_wrench_fx"] = results["right_wrench_rt"][:, 0]
    rt_columns["right_wrench_fy"] = results["right_wrench_rt"][:, 1]
    rt_columns["right_wrench_fz"] = results["right_wrench_rt"][:, 2]
    rt_columns["right_wrench_mx"] = results["right_wrench_rt"][:, 3]
    rt_columns["right_wrench_my"] = results["right_wrench_rt"][:, 4]
    rt_columns["right_wrench_mz"] = results["right_wrench_rt"][:, 5]
    rt_columns["root_residual_fx_bw"] = results["root_residual_rt"][:, 0] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["root_residual_fy_bw"] = results["root_residual_rt"][:, 1] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["root_residual_fz_bw"] = results["root_residual_rt"][:, 2] / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["root_residual_mx_bwh"] = results["root_residual_rt"][:, 3] / body_weight_height_nm if np.isfinite(body_weight_height_nm) and body_weight_height_nm > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["root_residual_my_bwh"] = results["root_residual_rt"][:, 4] / body_weight_height_nm if np.isfinite(body_weight_height_nm) and body_weight_height_nm > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["root_residual_mz_bwh"] = results["root_residual_rt"][:, 5] / body_weight_height_nm if np.isfinite(body_weight_height_nm) and body_weight_height_nm > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["root_residual_force_bw_norm"] = np.linalg.norm(results["root_residual_rt"][:, :3], axis=1) / body_weight_n if np.isfinite(body_weight_n) and body_weight_n > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["root_residual_moment_bwh_norm"] = np.linalg.norm(results["root_residual_rt"][:, 3:], axis=1) / body_weight_height_nm if np.isfinite(body_weight_height_nm) and body_weight_height_nm > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)

    rt_columns["mpjpe_m"] = results["mpjpe_rt"]
    rt_columns["dynamics_residual_norm"] = results["dyn_residual_rt"]
    rt_columns["dynamics_residual_norm_nm_per_kg"] = results["dyn_residual_rt"] / subject_mass_kg if np.isfinite(subject_mass_kg) and subject_mass_kg > 0.0 else np.full(len(results["q_rt"]), np.nan, dtype=float)
    rt_columns["solve_time_ms"] = 1000.0 * results["solve_time_rt"]
    rt_columns["input_noise_std_m"] = np.full(len(results["q_rt"]), float(args.noise_std), dtype=float)
    rt_columns["input_drop_joint_prob"] = np.full(len(results["q_rt"]), float(args.drop_joint_prob), dtype=float)
    rt_columns["input_mu"] = np.full(len(results["q_rt"]), float(args.mu), dtype=float)
    rt_columns["input_stage1_kin_filter"] = np.full(len(results["q_rt"]), float(args.stage1_kin_filter), dtype=float)
    rt_columns["input_init_policy"] = np.full(len(results["q_rt"]), args.init_policy, dtype=object)

    pd.DataFrame(rt_columns).to_csv(output_csv, index=False)
    return output_csv


def compute_realtime_metrics(results: dict[str, object]) -> dict[str, object]:
    left_metrics = binary_classification_metrics(results["left_contact_rt"], results["left_contact_ref"])
    right_metrics = binary_classification_metrics(results["right_contact_rt"], results["right_contact_ref"])
    subject_mass_kg = float(results.get("subject_mass_kg", float("nan")))
    subject_height_m = float(results.get("subject_height_m", float("nan")))
    body_weight_n = subject_mass_kg * 9.81 if np.isfinite(subject_mass_kg) and subject_mass_kg > 0.0 else float("nan")
    body_weight_height_nm = body_weight_n * subject_height_m if np.isfinite(body_weight_n) and np.isfinite(subject_height_m) and subject_height_m > 0.0 else float("nan")
    payload = {
        "frames": int(len(results["q_rt"])),
        "mpjpe_m_mean": float(np.mean(results["mpjpe_rt"])),
        "mpjpe_m_max": float(np.max(results["mpjpe_rt"])),
        "dyn_residual_norm_mean": float(np.mean(results["dyn_residual_rt"])),
        "dyn_residual_norm_max": float(np.max(results["dyn_residual_rt"])),
        "solve_time_ms_mean": float(1000.0 * np.mean(results["solve_time_rt"])),
        "solve_time_ms_p95": float(1000.0 * np.percentile(results["solve_time_rt"], 95.0)),
        "precision_metric_dofs": len(results["metric_dof_names"]),
        "dof_count": len(results["metric_mask"]),
        "q_rmse": rmse(results["metric_q_rt"], results["metric_q_ref"]),
        "q_mae": mae(results["metric_q_rt"], results["metric_q_ref"]),
        "dq_rmse": rmse(results["metric_dq_rt"], results["metric_dq_ref"]),
        "dq_mae": mae(results["metric_dq_rt"], results["metric_dq_ref"]),
        "ddq_rmse": rmse(results["metric_ddq_rt"], results["metric_ddq_ref"]),
        "ddq_mae": mae(results["metric_ddq_rt"], results["metric_ddq_ref"]),
        "tau_full_rmse": rmse(results["metric_tau_rt"], results["metric_tau_ref"]),
        "tau_full_mae": mae(results["metric_tau_rt"], results["metric_tau_ref"]),
        "tau_actuated_rmse": rmse(results["metric_tau_act_rt"], results["metric_tau_act_ref"]),
        "tau_actuated_mae": mae(results["metric_tau_act_rt"], results["metric_tau_act_ref"]),
        "left_grf_rmse": rmse(results["left_force_rt"], results["left_force_ref"]),
        "right_grf_rmse": rmse(results["right_force_rt"], results["right_force_ref"]),
        "tau_nm_per_kg_mean_abs": float(np.mean(np.abs(results["tau_rt"]) / subject_mass_kg)) if np.isfinite(subject_mass_kg) and subject_mass_kg > 0.0 else float("nan"),
        "left_grf_bw_mean": float(np.mean(np.linalg.norm(results["left_force_rt"], axis=1) / body_weight_n)) if np.isfinite(body_weight_n) and body_weight_n > 0.0 else float("nan"),
        "right_grf_bw_mean": float(np.mean(np.linalg.norm(results["right_force_rt"], axis=1) / body_weight_n)) if np.isfinite(body_weight_n) and body_weight_n > 0.0 else float("nan"),
        "root_residual_force_bw_mean": float(np.mean(np.linalg.norm(results["root_residual_rt"][:, :3], axis=1) / body_weight_n)) if np.isfinite(body_weight_n) and body_weight_n > 0.0 else float("nan"),
        "root_residual_moment_bwh_mean": float(np.mean(np.linalg.norm(results["root_residual_rt"][:, 3:], axis=1) / body_weight_height_nm)) if np.isfinite(body_weight_height_nm) and body_weight_height_nm > 0.0 else float("nan"),
        "left_contact": left_metrics,
        "right_contact": right_metrics,
        "worst_q_rmse": top_k_rmse(results["metric_q_rt"], results["metric_q_ref"], results["metric_dof_names"], k=8),
        "worst_tau_rmse": top_k_rmse(results["metric_tau_rt"], results["metric_tau_ref"], results["metric_dof_names"], k=8),
        "left_grf_axis_rmse": per_column_rmse(results["left_force_rt"], results["left_force_ref"], ["fx", "fy", "fz"]),
        "right_grf_axis_rmse": per_column_rmse(results["right_force_rt"], results["right_force_ref"], ["fx", "fy", "fz"]),
    }
    if results["tau_jerk_rt"].size > 0:
        payload["tau_actuated_jerk_rmse"] = rmse(results["tau_jerk_rt"], results["tau_jerk_ref"])
        payload["tau_actuated_jerk_l2_mean"] = float(np.mean(np.linalg.norm(results["tau_jerk_rt"], axis=1)))
    return payload


compute_metrics = compute_realtime_metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    frame_table = load_reference_csv(args.csv, args.max_frames)
    skeleton, skeleton_gt = configure_skeletons(args.model, frame_table)
    dof_names = get_model_dof_names(skeleton)
    signals = extract_reference_signals(frame_table, dof_names)
    joints_gt = [skeleton_gt.getJoint(name) for name in BSM_JOINT_NAMES]
    _set_positions_safely(skeleton_gt, signals["q"][0])
    _set_velocities_safely(skeleton_gt, signals["dq"][0])
    first_measurement = np.array(skeleton_gt.getJointWorldPositions(joints_gt), dtype=float).reshape(-1, 3)
    config = RealtimeConfig(
        dt=0.033,
        mu=args.mu,
        steps=1,
        use_stage1_kin_filter=args.stage1_kin_filter,
        init_policy=args.init_policy,
    )
    rt_state, joints, workspace, skeleton = bootstrap_initial_state(skeleton, signals, config, first_measurement=first_measurement)
    results = run_realtime_sequence(args, frame_table, skeleton, skeleton_gt, dof_names, signals, joints, rt_state, workspace, config)
    metrics = compute_realtime_metrics(results)

    if args.output_csv is not None:
        output_csv = write_realtime_csv(args, frame_table, dof_names, results)
    else:
        output_csv = None

    if args.metrics_json is not None:
        metrics_json = args.metrics_json.resolve()
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        metrics["init_policy"] = config.init_policy
        metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"frames: {metrics['frames']}")
    print(f"noise_std_m: {args.noise_std:.6f}")
    print(f"drop_joint_prob: {args.drop_joint_prob:.6f}")
    print(f"stage1_kin_filter: {int(args.stage1_kin_filter)}")
    print(f"init_policy: {config.init_policy}")
    print(f"mpjpe_m: mean={metrics['mpjpe_m_mean']:.6f} max={metrics['mpjpe_m_max']:.6f}")
    print(f"dyn_residual_norm: mean={metrics['dyn_residual_norm_mean']:.6f} max={metrics['dyn_residual_norm_max']:.6f}")
    print(f"solve_time_ms: mean={metrics['solve_time_ms_mean']:.6f} p95={metrics['solve_time_ms_p95']:.6f}")
    print(f"precision_metric_dofs: {metrics['precision_metric_dofs']}/{metrics['dof_count']} (excluding ankle/head/wrist-related angles)")
    print()
    print(f"q_rmse: {metrics['q_rmse']:.6f}")
    print(f"q_mae: {metrics['q_mae']:.6f}")
    print(f"dq_rmse: {metrics['dq_rmse']:.6f}")
    print(f"dq_mae: {metrics['dq_mae']:.6f}")
    print(f"ddq_rmse: {metrics['ddq_rmse']:.6f}")
    print(f"ddq_mae: {metrics['ddq_mae']:.6f}")
    print(f"tau_full_rmse: {metrics['tau_full_rmse']:.6f}")
    print(f"tau_full_mae: {metrics['tau_full_mae']:.6f}")
    print(f"tau_actuated_rmse: {metrics['tau_actuated_rmse']:.6f}")
    print(f"tau_actuated_mae: {metrics['tau_actuated_mae']:.6f}")
    if "tau_actuated_jerk_rmse" in metrics:
        print(f"tau_actuated_jerk_rmse: {metrics['tau_actuated_jerk_rmse']:.6f}")
        print(f"tau_actuated_jerk_l2_mean: {metrics['tau_actuated_jerk_l2_mean']:.6f}")
    print(f"left_grf_rmse: {metrics['left_grf_rmse']:.6f}")
    print(f"right_grf_rmse: {metrics['right_grf_rmse']:.6f}")

    left_metrics = metrics["left_contact"]
    right_metrics = metrics["right_contact"]
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
    for name, value in metrics["worst_q_rmse"]:
        print(f"  {name}: {value:.6f}")

    print("worst_tau_rmse:")
    for name, value in metrics["worst_tau_rmse"]:
        print(f"  {name}: {value:.6f}")

    print("left_grf_axis_rmse:")
    for name, value in metrics["left_grf_axis_rmse"]:
        print(f"  {name}: {value:.6f}")

    print("right_grf_axis_rmse:")
    for name, value in metrics["right_grf_axis_rmse"]:
        print(f"  {name}: {value:.6f}")

    if output_csv is not None:
        print()
        print(f"realtime_csv: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

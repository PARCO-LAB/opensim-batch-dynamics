import cvxpy as cp
import numpy as np


EPS = 1e-8
PARAM_DENSE_EPS = 1e-9
STAGE1_MAX_ITERS = 5
STAGE1_EPS_ABS = 1e-4
STAGE1_EPS_REL = 1e-4
STAGE1_MAX_OSQP_ITERS = 3000
STAGE2_EPS_ABS = 2e-4
STAGE2_EPS_REL = 2e-4
STAGE2_MAX_OSQP_ITERS = 5000
STAGE1_GEOM_ROWS = 36
CONTACT_ON_SCORE = 0.52
CONTACT_OFF_SCORE = 0.34
FOOT_COP_HALF_LENGTH = 0.11
FOOT_COP_HALF_WIDTH = 0.045
FOOT_TORSION_RADIUS = 0.07
KIN_BETA_POS = 0.18
KIN_BETA_ROOT_ROT = 0.10
KIN_BETA_JOINT = 0.06
KIN_GAMMA_POS = 0.020
KIN_GAMMA_ROOT_ROT = 0.010
KIN_GAMMA_JOINT = 0.006
ORI_PELVIS_YAW_GAIN = 0.08
ORI_PELVIS_ROLL_GAIN = 0.06
ORI_PELVIS_TILT_GAIN = 0.04
ORI_TRUNK_EXT_GAIN = 0.12
ORI_TRUNK_BEND_GAIN = 0.10
ORI_TRUNK_TWIST_GAIN = 0.08
FOOT_BODIES = {
    "left": ("calcn_l", "toes_l"),
    "right": ("calcn_r", "toes_r"),
}
QPIK_CACHE = {}
DYN_QP_CACHE = {}


def get_model_dof_names(skeleton):
    model = skeleton._nimble if hasattr(skeleton, "_nimble") else skeleton
    return [model.getDofByIndex(i).getName() for i in range(model.getNumDofs())]


def initialize_rt_state(
    skeleton,
    q=None,
    dq=None,
    ddq=None,
    tau=None,
    tau_full=None,
    root_residual=None,
    foot_forces=None,
    foot_wrenches=None,
    contact_state=None,
    contact_prob=None,
    floor_height=np.nan,
):
    model = skeleton._nimble if hasattr(skeleton, "_nimble") else skeleton
    n_dof = model.getNumDofs()
    n_act = n_dof - 6
    state = {
        "q": np.array(model.getPositions(), dtype=float) if q is None else np.array(q, dtype=float).copy(),
        "dq": np.array(model.getVelocities(), dtype=float) if dq is None else np.array(dq, dtype=float).copy(),
        "ddq": np.zeros(n_dof, dtype=float) if ddq is None else np.array(ddq, dtype=float).copy(),
        "q_kin": np.array(model.getPositions(), dtype=float) if q is None else np.array(q, dtype=float).copy(),
        "dq_kin": np.array(model.getVelocities(), dtype=float) if dq is None else np.array(dq, dtype=float).copy(),
        "ddq_kin": np.zeros(n_dof, dtype=float) if ddq is None else np.array(ddq, dtype=float).copy(),
        "tau": np.zeros(n_act, dtype=float) if tau is None else np.array(tau, dtype=float).copy(),
        "tau_full": np.zeros(n_dof, dtype=float) if tau_full is None else np.array(tau_full, dtype=float).copy(),
        "root_residual": np.zeros(6, dtype=float) if root_residual is None else np.array(root_residual, dtype=float).copy(),
        "foot_forces": {
            "left": np.zeros(3, dtype=float),
            "right": np.zeros(3, dtype=float),
        },
        "foot_wrenches": {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        },
        "contact_state": {
            "left": False,
            "right": False,
        },
        "contact_prob": {
            "left": 0.0,
            "right": 0.0,
        },
        "floor_height": float(floor_height),
        "step_index": 0,
    }
    if foot_forces is not None:
        for side in state["foot_forces"]:
            if side in foot_forces:
                state["foot_forces"][side] = np.array(foot_forces[side], dtype=float).copy()
    if foot_wrenches is not None:
        for side in state["foot_wrenches"]:
            if side in foot_wrenches:
                state["foot_wrenches"][side] = np.array(foot_wrenches[side], dtype=float).copy()
    if contact_state is not None:
        for side in state["contact_state"]:
            if side in contact_state:
                state["contact_state"][side] = bool(contact_state[side])
    if contact_prob is not None:
        for side in state["contact_prob"]:
            if side in contact_prob:
                state["contact_prob"][side] = float(np.clip(contact_prob[side], 0.0, 1.0))
    return state


def get_world_contact_points(contact_info):
    world_points = []
    for body_node, local_offset, _ in contact_info:
        offset = np.array(local_offset, dtype=float).reshape(3)
        transform = np.array(body_node.getWorldTransform().matrix(), dtype=float)
        world_h = transform @ np.array([offset[0], offset[1], offset[2], 1.0], dtype=float)
        world_points.append(world_h[:3])
    return world_points


def estimate_contact_points(skeleton, target=None, floor_z=0.05, chair_z=0.7):
    del target, chair_z
    model = skeleton._nimble if hasattr(skeleton, "_nimble") else skeleton
    contact_info = [
        (model.getBodyNode("calcn_l"), np.zeros(3, dtype=float), floor_z),
        (model.getBodyNode("toes_l"), np.zeros(3, dtype=float), floor_z),
        (model.getBodyNode("calcn_r"), np.zeros(3, dtype=float), floor_z),
        (model.getBodyNode("toes_r"), np.zeros(3, dtype=float), floor_z),
    ]
    world_points = get_world_contact_points(contact_info)
    to_keep = []
    for point, item in zip(world_points, contact_info):
        if float(point[2]) <= float(item[2]):
            to_keep.append(item)
    return to_keep


def get_contact_jacobian_deriv_times_dq(skeleton, contact_info, dq):
    model = skeleton._nimble if hasattr(skeleton, "_nimble") else skeleton
    dJc_dq = np.zeros(len(contact_info) * 3, dtype=float)
    for i, (body_node, offset, _) in enumerate(contact_info):
        dJc_dq[3 * i : 3 * i + 3] = np.array(model.getLinearJacobianDeriv(body_node, offset), dtype=float) @ dq
    return dJc_dq


def get_task_jacobian_derivative_times_dq(skeleton, keypoint_joints, dq):
    model = skeleton._nimble if hasattr(skeleton, "_nimble") else skeleton
    dJ_dq = []
    for joint in keypoint_joints:
        body_node = joint.getChildBodyNode()
        local_offset = np.array(joint.getTransformFromChildBodyNode().translation(), dtype=float)
        dJ_dq.append(np.array(model.getLinearJacobianDeriv(body_node, local_offset), dtype=float) @ dq)
    if len(dJ_dq) == 0:
        return np.zeros(0, dtype=float)
    return np.concatenate(dJ_dq)


def get_contact_jacobian(skeleton, contact_info):
    model = skeleton._nimble if hasattr(skeleton, "_nimble") else skeleton
    J_c = []
    for body_node, local_offset, _ in contact_info:
        J_c.append(np.array(model.getLinearJacobian(body_node, local_offset), dtype=float))
    if len(J_c) == 0:
        return np.zeros((0, model.getNumDofs()), dtype=float)
    return np.vstack(J_c)


def solve_system(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def qpid(
    skeleton,
    x_t,
    dt=0.033,
    mu=0.8,
    excluded_DOFs=None,
    ddq_prev=None,
    tau_prev=None,
    steps=1,
    measurement_joints=None,
    state=None,
    use_stage1_kin_filter=True,
):
    model = skeleton._nimble if hasattr(skeleton, "_nimble") else skeleton

    if measurement_joints is None:
        if hasattr(skeleton, "joints"):
            measurement_joints = list(skeleton.joints)
        else:
            raise ValueError("measurement_joints required")

    if excluded_DOFs is None:
        excluded_DOFs = []

    n_dof_full = model.getNumDofs()
    n_act_full = n_dof_full - 6
    dof_names_full = [model.getDofByIndex(i).getName() for i in range(n_dof_full)]
    pelvis_orient_mask = np.array(
        [name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation") for name in dof_names_full],
        dtype=bool,
    )
    trunk_mask = np.array([name.startswith("lumbar_") or name.startswith("thorax_") for name in dof_names_full], dtype=bool)
    scapula_mask = np.array([name.startswith("scapula_") for name in dof_names_full], dtype=bool)
    shoulder_mask = np.array([name.startswith("shoulder_") for name in dof_names_full], dtype=bool)
    weak_obs_mask = pelvis_orient_mask | trunk_mask | scapula_mask | shoulder_mask
    dof_index = {name: i for i, name in enumerate(dof_names_full)}
    if state is None:
        state = initialize_rt_state(
            skeleton,
            ddq=np.zeros(n_dof_full, dtype=float) if ddq_prev is None else ddq_prev,
            tau=np.zeros(n_act_full, dtype=float) if tau_prev is None else tau_prev,
        )
    else:
        foot_force_state = state.get(
            "foot_forces",
            {
                "left": np.zeros(3, dtype=float),
                "right": np.zeros(3, dtype=float),
            },
        )
        foot_wrench_state = state.get(
            "foot_wrenches",
            {
                "left": np.zeros(6, dtype=float),
                "right": np.zeros(6, dtype=float),
            },
        )
        state = {
            "q": np.array(state["q"], dtype=float).copy(),
            "dq": np.array(state["dq"], dtype=float).copy(),
            "ddq": np.array(state["ddq"], dtype=float).copy(),
            "q_kin": np.array(state.get("q_kin", state["q"]), dtype=float).copy(),
            "dq_kin": np.array(state.get("dq_kin", state["dq"]), dtype=float).copy(),
            "ddq_kin": np.array(state.get("ddq_kin", state["ddq"]), dtype=float).copy(),
            "tau": np.array(state["tau"], dtype=float).copy(),
            "tau_full": np.array(state["tau_full"], dtype=float).copy(),
            "root_residual": np.array(state["root_residual"], dtype=float).copy(),
            "foot_forces": {
                "left": np.array(foot_force_state["left"], dtype=float).copy(),
                "right": np.array(foot_force_state["right"], dtype=float).copy(),
            },
            "foot_wrenches": {
                "left": np.array(foot_wrench_state["left"], dtype=float).copy(),
                "right": np.array(foot_wrench_state["right"], dtype=float).copy(),
            },
            "contact_state": {
                "left": bool(state["contact_state"]["left"]),
                "right": bool(state["contact_state"]["right"]),
            },
            "contact_prob": {
                "left": float(state.get("contact_prob", {}).get("left", 1.0 if state["contact_state"]["left"] else 0.0)),
                "right": float(state.get("contact_prob", {}).get("right", 1.0 if state["contact_state"]["right"] else 0.0)),
            },
            "floor_height": float(state["floor_height"]),
            "step_index": int(state["step_index"]),
        }

    if ddq_prev is not None:
        state["ddq"] = np.array(ddq_prev, dtype=float).copy()
    if tau_prev is not None:
        state["tau"] = np.array(tau_prev, dtype=float).copy()

    if isinstance(x_t, dict):
        x_obs = x_t.get("joint_positions", x_t.get("keypoints", x_t.get("x_t")))
        x_obs = np.array(x_obs, dtype=float)
        joint_weights = x_t.get("joint_weights", x_t.get("confidence"))
        if joint_weights is None:
            joint_weights = np.ones((len(measurement_joints), 3), dtype=float)
        else:
            joint_weights = np.array(joint_weights, dtype=float)
            if joint_weights.ndim == 1 and joint_weights.size == len(measurement_joints):
                joint_weights = np.repeat(joint_weights.reshape(-1, 1), 3, axis=1)
            elif joint_weights.ndim == 1 and joint_weights.size == len(measurement_joints) * 3:
                joint_weights = joint_weights.reshape(-1, 3)
        q_meas = x_t.get("q_meas", x_t.get("q"))
        q_weights = x_t.get("q_weights")
        dq_meas = x_t.get("dq_meas", x_t.get("dq"))
        dq_weights = x_t.get("dq_weights")
        use_stage1_kin_filter = bool(x_t.get("use_stage1_kin_filter", use_stage1_kin_filter))
    else:
        x_obs = np.array(x_t, dtype=float)
        joint_weights = np.ones((len(measurement_joints), 3), dtype=float)
        q_meas = None
        q_weights = None
        dq_meas = None
        dq_weights = None

    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(-1, 3)
    if x_obs.shape != (len(measurement_joints), 3):
        raise ValueError(f"Unsupported joint measurement shape: {x_obs.shape}")

    joint_weights = np.where(np.isfinite(x_obs), joint_weights, 0.0)
    joint_weights = np.clip(joint_weights, 0.0, None)
    joint_name_to_idx = {joint.getName(): i for i, joint in enumerate(measurement_joints)}

    if q_meas is not None:
        q_meas = np.array(q_meas, dtype=float).reshape(-1)
    if q_weights is not None:
        q_weights = np.clip(np.array(q_weights, dtype=float).reshape(-1), 0.0, None)
    if dq_meas is not None:
        dq_meas = np.array(dq_meas, dtype=float).reshape(-1)
    if dq_weights is not None:
        dq_weights = np.clip(np.array(dq_weights, dtype=float).reshape(-1), 0.0, None)

    active_dofs = np.array([i for i in range(n_dof_full) if i not in excluded_DOFs], dtype=int)
    active_act_dofs = np.array([i for i in active_dofs if i >= 6], dtype=int)

    q_lower = np.array(model.getPositionLowerLimits(), dtype=float)
    q_upper = np.array(model.getPositionUpperLimits(), dtype=float)
    dq_cap = np.ones(n_dof_full, dtype=float) * 25.0
    dq_cap[:3] = 12.0
    dq_cap[3:6] = 8.0
    dt_sim = float(dt) / max(int(steps), 1)
    x_start = np.array(model.getJointWorldPositions(measurement_joints), dtype=float).reshape(-1, 3)
    result = None

    for step_idx in range(max(int(steps), 1)):
        alpha = float(step_idx + 1) / max(int(steps), 1)
        x_step = x_start + alpha * (x_obs - x_start)
        x_step = np.where(joint_weights > 0.0, x_step, np.nan)
        up_stage1 = np.array([0.0, 0.0, 1.0], dtype=float)

        q_prev = np.array(state["q"], dtype=float)
        dq_prev = np.array(state["dq"], dtype=float)
        ddq_prev_full = np.array(state["ddq"], dtype=float)

        q_pred = q_prev + dt_sim * dq_prev + 0.5 * dt_sim * dt_sim * ddq_prev_full
        q_pred = np.clip(q_pred, q_lower, q_upper)
        dq_pred = dq_prev + dt_sim * ddq_prev_full
        q_hat = q_pred.copy()
        measurement_weights = joint_weights.copy()
        dof_reliability = np.zeros(n_dof_full, dtype=float)

        if np.any(measurement_weights > 0.0):
            dq_step_cap = dt_sim * dq_cap
            dq_cumulative = np.zeros(n_dof_full, dtype=float)
            prior_w = np.ones(n_dof_full, dtype=float) * 0.08
            prior_w[:3] = 0.04
            prior_w[3:6] = 0.015
            step_w = np.ones(n_dof_full, dtype=float) * 0.02
            step_w[:3] = 0.006
            step_w[3:6] = 0.01
            prior_w[pelvis_orient_mask] *= 3.0
            prior_w[trunk_mask] *= 2.8
            prior_w[scapula_mask] *= 3.4
            prior_w[shoulder_mask] *= 1.8
            step_w[pelvis_orient_mask] *= 2.5
            step_w[trunk_mask] *= 2.8
            step_w[scapula_mask] *= 3.2
            step_w[shoulder_mask] *= 1.7
            if q_weights is not None:
                prior_w += 0.15 * q_weights

            for _ in range(STAGE1_MAX_ITERS + 1):
                model.setPositions(q_hat)
                model.setVelocities(dq_pred)
                x_current = np.array(model.getJointWorldPositions(measurement_joints), dtype=float).reshape(-1, 3)
                J_full = np.array(model.getJointWorldPositionsJacobianWrtJointPositions(measurement_joints), dtype=float)
                residual = x_step - x_current
                valid_xyz = measurement_weights > 0.0
                if np.any(valid_xyz):
                    root_shift = np.sum(np.where(valid_xyz, residual, 0.0), axis=0) / max(float(np.sum(valid_xyz[:, 0])), 1.0)
                    q_hat[3:6] += root_shift
                    q_hat = np.clip(q_hat, q_lower, q_upper)
                    model.setPositions(q_hat)
                    model.setVelocities(dq_pred)
                    x_current = np.array(model.getJointWorldPositions(measurement_joints), dtype=float).reshape(-1, 3)
                    J_full = np.array(model.getJointWorldPositionsJacobianWrtJointPositions(measurement_joints), dtype=float)
                    residual = x_step - x_current

                innovation = np.linalg.norm(np.where(measurement_weights > 0.0, residual, 0.0), axis=1)
                robust = 1.0 / (1.0 + (innovation / 0.05) ** 2)
                measurement_weights = joint_weights * robust.reshape(-1, 1)
                geom_rows = []
                geom_targets = []
                geom_weights = []
                hip_l_idx = joint_name_to_idx.get("hip_l")
                hip_r_idx = joint_name_to_idx.get("hip_r")
                sh_l_idx = joint_name_to_idx.get("GlenoHumeral_l")
                sh_r_idx = joint_name_to_idx.get("GlenoHumeral_r")
                el_l_idx = joint_name_to_idx.get("elbow_l")
                el_r_idx = joint_name_to_idx.get("elbow_r")
                wr_l_idx = joint_name_to_idx.get("wrist_l")
                wr_r_idx = joint_name_to_idx.get("wrist_r")
                knee_l_idx = joint_name_to_idx.get("walker_knee_l")
                knee_r_idx = joint_name_to_idx.get("walker_knee_r")
                ank_l_idx = joint_name_to_idx.get("ankle_l")
                ank_r_idx = joint_name_to_idx.get("ankle_r")
                if hip_l_idx is not None and hip_r_idx is not None:
                    if np.all(measurement_weights[[hip_l_idx, hip_r_idx], :] > 0.0):
                        J_hip = J_full[3 * hip_r_idx : 3 * hip_r_idx + 3, :] - J_full[3 * hip_l_idx : 3 * hip_l_idx + 3, :]
                        hip_target = x_step[hip_r_idx] - x_step[hip_l_idx]
                        hip_current = x_current[hip_r_idx] - x_current[hip_l_idx]
                        geom_rows.append(J_hip)
                        geom_targets.append(hip_target - hip_current)
                        geom_weights.append(np.ones(3, dtype=float) * 18.0 * float(np.mean(measurement_weights[[hip_l_idx, hip_r_idx], :])))
                if sh_l_idx is not None and sh_r_idx is not None:
                    if np.all(measurement_weights[[sh_l_idx, sh_r_idx], :] > 0.0):
                        J_sh = J_full[3 * sh_r_idx : 3 * sh_r_idx + 3, :] - J_full[3 * sh_l_idx : 3 * sh_l_idx + 3, :]
                        sh_target = x_step[sh_r_idx] - x_step[sh_l_idx]
                        sh_current = x_current[sh_r_idx] - x_current[sh_l_idx]
                        geom_rows.append(J_sh)
                        geom_targets.append(sh_target - sh_current)
                        geom_weights.append(np.ones(3, dtype=float) * 16.0 * float(np.mean(measurement_weights[[sh_l_idx, sh_r_idx], :])))
                if hip_l_idx is not None and hip_r_idx is not None and sh_l_idx is not None and sh_r_idx is not None:
                    if np.all(measurement_weights[[hip_l_idx, hip_r_idx, sh_l_idx, sh_r_idx], :] > 0.0):
                        J_trunk = 0.5 * (J_full[3 * sh_l_idx : 3 * sh_l_idx + 3, :] + J_full[3 * sh_r_idx : 3 * sh_r_idx + 3, :])
                        J_trunk -= 0.5 * (J_full[3 * hip_l_idx : 3 * hip_l_idx + 3, :] + J_full[3 * hip_r_idx : 3 * hip_r_idx + 3, :])
                        trunk_target = 0.5 * (x_step[sh_l_idx] + x_step[sh_r_idx]) - 0.5 * (x_step[hip_l_idx] + x_step[hip_r_idx])
                        trunk_current = 0.5 * (x_current[sh_l_idx] + x_current[sh_r_idx]) - 0.5 * (x_current[hip_l_idx] + x_current[hip_r_idx])
                        geom_rows.append(J_trunk)
                        geom_targets.append(trunk_target - trunk_current)
                        geom_weights.append(np.array([12.0, 12.0, 18.0], dtype=float) * float(np.mean(measurement_weights[[hip_l_idx, hip_r_idx, sh_l_idx, sh_r_idx], :])))
                if knee_l_idx is not None and knee_r_idx is not None:
                    if np.all(measurement_weights[[knee_l_idx, knee_r_idx], :] > 0.0):
                        J_knee = J_full[3 * knee_r_idx : 3 * knee_r_idx + 3, :] - J_full[3 * knee_l_idx : 3 * knee_l_idx + 3, :]
                        knee_target = x_step[knee_r_idx] - x_step[knee_l_idx]
                        knee_current = x_current[knee_r_idx] - x_current[knee_l_idx]
                        geom_rows.append(J_knee)
                        geom_targets.append(knee_target - knee_current)
                        geom_weights.append(np.ones(3, dtype=float) * 10.0 * float(np.mean(measurement_weights[[knee_l_idx, knee_r_idx], :])))
                for a_idx, b_idx, base_w in [
                    (sh_l_idx, el_l_idx, 14.0),
                    (sh_r_idx, el_r_idx, 14.0),
                    (el_l_idx, wr_l_idx, 10.0),
                    (el_r_idx, wr_r_idx, 10.0),
                    (hip_l_idx, knee_l_idx, 13.0),
                    (hip_r_idx, knee_r_idx, 13.0),
                    (knee_l_idx, ank_l_idx, 10.0),
                    (knee_r_idx, ank_r_idx, 10.0),
                ]:
                    if a_idx is None or b_idx is None:
                        continue
                    if not np.all(measurement_weights[[a_idx, b_idx], :] > 0.0):
                        continue
                    J_seg = J_full[3 * b_idx : 3 * b_idx + 3, :] - J_full[3 * a_idx : 3 * a_idx + 3, :]
                    seg_target = x_step[b_idx] - x_step[a_idx]
                    seg_current = x_current[b_idx] - x_current[a_idx]
                    geom_rows.append(J_seg)
                    geom_targets.append(seg_target - seg_current)
                    geom_weights.append(np.ones(3, dtype=float) * base_w * float(np.mean(measurement_weights[[a_idx, b_idx], :])))
                valid = measurement_weights.reshape(-1) > 0.0
                if not np.any(valid):
                    break
                if np.max(np.abs(np.where(measurement_weights > 0.0, residual, 0.0))) < 1e-4:
                    dof_reliability = np.sum((measurement_weights.reshape(-1)[valid].reshape(-1, 1) * J_full[valid, :]) ** 2, axis=0)
                    break

                has_q_sensor = bool(q_meas is not None and q_weights is not None)
                cache_key = (n_dof_full, len(measurement_joints), has_q_sensor)
                if cache_key not in QPIK_CACHE:
                    dq_var = cp.Variable(n_dof_full)
                    q_param = cp.Parameter(n_dof_full)
                    meas_rows = 3 * len(measurement_joints)
                    qik_A_meas_param = cp.Parameter((meas_rows, n_dof_full))
                    qik_b_meas_param = cp.Parameter(meas_rows)
                    qik_A_geom_param = cp.Parameter((STAGE1_GEOM_ROWS, n_dof_full))
                    qik_b_geom_param = cp.Parameter(STAGE1_GEOM_ROWS)
                    dq_accum_param = cp.Parameter(n_dof_full)
                    dq_l_param = cp.Parameter(n_dof_full)
                    dq_u_param = cp.Parameter(n_dof_full)
                    q_l_param = cp.Parameter(n_dof_full)
                    q_u_param = cp.Parameter(n_dof_full)
                    qik_A_prior_param = cp.Parameter((n_dof_full, n_dof_full))
                    qik_b_prior_param = cp.Parameter(n_dof_full)
                    objective = cp.sum_squares(qik_A_meas_param @ dq_var - qik_b_meas_param)
                    objective += cp.sum_squares(qik_A_geom_param @ dq_var - qik_b_geom_param)
                    objective += cp.sum_squares(cp.multiply(step_w, dq_var))
                    if has_q_sensor:
                        qik_A_sensor_param = cp.Parameter((n_dof_full, n_dof_full))
                        qik_b_sensor_param = cp.Parameter(n_dof_full)
                        objective += cp.sum_squares(qik_A_prior_param @ dq_var - qik_b_prior_param)
                        objective += cp.sum_squares(qik_A_sensor_param @ dq_var - qik_b_sensor_param)
                    else:
                        qik_A_sensor_param = None
                        qik_b_sensor_param = None
                        objective += cp.sum_squares(qik_A_prior_param @ dq_var - qik_b_prior_param)
                    constraints = [
                        q_param + dq_var >= q_l_param,
                        q_param + dq_var <= q_u_param,
                        dq_accum_param + dq_var >= dq_l_param,
                        dq_accum_param + dq_var <= dq_u_param,
                    ]
                    QPIK_CACHE[cache_key] = {
                        "problem": cp.Problem(cp.Minimize(objective), constraints),
                        "dq": dq_var,
                        "q": q_param,
                        "A_meas": qik_A_meas_param,
                        "b_meas": qik_b_meas_param,
                        "A_geom": qik_A_geom_param,
                        "b_geom": qik_b_geom_param,
                        "dq_accum": dq_accum_param,
                        "dq_l": dq_l_param,
                        "dq_u": dq_u_param,
                        "q_l": q_l_param,
                        "q_u": q_u_param,
                        "A_prior": qik_A_prior_param,
                        "b_prior": qik_b_prior_param,
                        "A_sensor": qik_A_sensor_param,
                        "b_sensor": qik_b_sensor_param,
                    }

                problem = QPIK_CACHE[cache_key]
                problem["q"].value = q_hat
                meas_w = 30.0 * np.maximum(measurement_weights.reshape(-1), 0.0)
                meas_residual_full = x_step.reshape(-1) - x_current.reshape(-1)
                meas_residual_full = np.where(np.isfinite(meas_residual_full), meas_residual_full, 0.0)
                problem["A_meas"].value = np.diag(meas_w) @ (J_full + PARAM_DENSE_EPS)
                problem["b_meas"].value = meas_w * meas_residual_full
                geom_A_full = np.zeros((STAGE1_GEOM_ROWS, n_dof_full), dtype=float)
                geom_b_full = np.zeros(STAGE1_GEOM_ROWS, dtype=float)
                geom_offset = 0
                for row_block, target_block, weight_block in zip(geom_rows, geom_targets, geom_weights):
                    rows_here = row_block.shape[0]
                    end = min(geom_offset + rows_here, STAGE1_GEOM_ROWS)
                    take = end - geom_offset
                    if take <= 0:
                        break
                    geom_A_full[geom_offset:end, :] = weight_block[:take].reshape(-1, 1) * row_block[:take, :]
                    geom_b_full[geom_offset:end] = weight_block[:take] * target_block[:take]
                    geom_offset = end
                problem["A_geom"].value = geom_A_full + PARAM_DENSE_EPS
                problem["b_geom"].value = geom_b_full
                problem["dq_accum"].value = dq_cumulative
                problem["dq_l"].value = -dq_step_cap
                problem["dq_u"].value = dq_step_cap
                problem["q_l"].value = q_lower
                problem["q_u"].value = q_upper
                problem["A_prior"].value = np.diag(prior_w)
                problem["b_prior"].value = prior_w * (q_pred - q_hat)
                if problem["A_sensor"] is not None:
                    q_sensor = np.where(np.isfinite(q_meas), q_meas, q_pred)
                    q_sensor_w = 1.5 * np.where(np.isfinite(q_meas), q_weights, 0.0)
                    problem["A_sensor"].value = np.diag(q_sensor_w)
                    problem["b_sensor"].value = q_sensor_w * (q_sensor - q_hat)

                try:
                    problem["problem"].solve(
                        solver=cp.OSQP,
                        warm_start=True,
                        ignore_dpp=False,
                        verbose=False,
                        max_iter=STAGE1_MAX_OSQP_ITERS,
                        eps_abs=STAGE1_EPS_ABS,
                        eps_rel=STAGE1_EPS_REL,
                        polish=False,
                    )
                except cp.SolverError:
                    try:
                        problem["problem"].solve(
                            solver=cp.SCS,
                            warm_start=True,
                            ignore_dpp=False,
                            verbose=False,
                            max_iters=5000,
                            eps=1e-5,
                        )
                    except cp.SolverError:
                        break

                delta_q = None if problem["dq"].value is None else np.array(problem["dq"].value, dtype=float).reshape(-1)
                if delta_q is None or not np.all(np.isfinite(delta_q)):
                    break
                dq_cumulative += delta_q
                q_hat = np.clip(q_hat + delta_q, q_lower, q_upper)
                dof_reliability = np.sum((measurement_weights.reshape(-1)[valid].reshape(-1, 1) * J_full[valid, :]) ** 2, axis=0)
                if np.linalg.norm(delta_q) < 1e-5:
                    break

        if q_weights is not None:
            dof_reliability += q_weights * q_weights
        dof_reliability[3:6] += np.sum(measurement_weights[:, 0] > 0.0)
        if np.max(dof_reliability) > EPS:
            dof_reliability = np.clip(dof_reliability / np.max(dof_reliability), 0.0, 1.0)
        else:
            dof_reliability[:] = 0.0

        hip_l_idx = joint_name_to_idx.get("hip_l")
        hip_r_idx = joint_name_to_idx.get("hip_r")
        sh_l_idx = joint_name_to_idx.get("GlenoHumeral_l")
        sh_r_idx = joint_name_to_idx.get("GlenoHumeral_r")
        if (
            hip_l_idx is not None
            and hip_r_idx is not None
            and sh_l_idx is not None
            and sh_r_idx is not None
            and np.all(joint_weights[[hip_l_idx, hip_r_idx, sh_l_idx, sh_r_idx], :] > 0.0)
        ):
            model.setPositions(q_hat)
            model.setVelocities(dq_pred)
            x_geom_current = np.array(model.getJointWorldPositions(measurement_joints), dtype=float).reshape(-1, 3)
            hip_target_l = x_step[hip_l_idx]
            hip_target_r = x_step[hip_r_idx]
            sh_target_l = x_step[sh_l_idx]
            sh_target_r = x_step[sh_r_idx]
            hip_curr_l = x_geom_current[hip_l_idx]
            hip_curr_r = x_geom_current[hip_r_idx]
            sh_curr_l = x_geom_current[sh_l_idx]
            sh_curr_r = x_geom_current[sh_r_idx]
            hip_target_center = 0.5 * (hip_target_l + hip_target_r)
            hip_curr_center = 0.5 * (hip_curr_l + hip_curr_r)
            sh_target_center = 0.5 * (sh_target_l + sh_target_r)
            sh_curr_center = 0.5 * (sh_curr_l + sh_curr_r)
            pelvis_target_lr = hip_target_r - hip_target_l
            pelvis_curr_lr = hip_curr_r - hip_curr_l
            shoulder_target_lr = sh_target_r - sh_target_l
            shoulder_curr_lr = sh_curr_r - sh_curr_l
            trunk_target = sh_target_center - hip_target_center
            trunk_curr = sh_curr_center - hip_curr_center
            target_conf = float(np.mean(joint_weights[[hip_l_idx, hip_r_idx, sh_l_idx, sh_r_idx], :]))

            pelvis_target_lr_h = pelvis_target_lr - np.dot(pelvis_target_lr, up_stage1) * up_stage1
            pelvis_curr_lr_h = pelvis_curr_lr - np.dot(pelvis_curr_lr, up_stage1) * up_stage1
            shoulder_target_lr_h = shoulder_target_lr - np.dot(shoulder_target_lr, up_stage1) * up_stage1
            shoulder_curr_lr_h = shoulder_curr_lr - np.dot(shoulder_curr_lr, up_stage1) * up_stage1
            pelvis_target_h_norm = float(np.linalg.norm(pelvis_target_lr_h))
            pelvis_curr_h_norm = float(np.linalg.norm(pelvis_curr_lr_h))
            shoulder_target_h_norm = float(np.linalg.norm(shoulder_target_lr_h))
            shoulder_curr_h_norm = float(np.linalg.norm(shoulder_curr_lr_h))
            trunk_target_norm = float(np.linalg.norm(trunk_target))
            trunk_curr_norm = float(np.linalg.norm(trunk_curr))

            yaw_err = 0.0
            if pelvis_target_h_norm > 1e-6 and pelvis_curr_h_norm > 1e-6:
                pelvis_target_lr_h /= pelvis_target_h_norm
                pelvis_curr_lr_h /= pelvis_curr_h_norm
                yaw_err = float(
                        np.arctan2(
                        np.dot(np.cross(pelvis_curr_lr_h, pelvis_target_lr_h), up_stage1),
                        np.clip(np.dot(pelvis_curr_lr_h, pelvis_target_lr_h), -1.0, 1.0),
                    )
                )

            roll_target = float(np.arctan2(np.dot(pelvis_target_lr, up_stage1), max(pelvis_target_h_norm, 1e-6)))
            roll_curr = float(np.arctan2(np.dot(pelvis_curr_lr, up_stage1), max(pelvis_curr_h_norm, 1e-6)))
            roll_err = roll_target - roll_curr

            pitch_err = 0.0
            bend_err = 0.0
            twist_err = 0.0
            if trunk_target_norm > 1e-6 and trunk_curr_norm > 1e-6 and pelvis_target_h_norm > 1e-6 and pelvis_curr_h_norm > 1e-6:
                trunk_target_unit = trunk_target / trunk_target_norm
                trunk_curr_unit = trunk_curr / trunk_curr_norm
                forward_target = np.cross(up_stage1, pelvis_target_lr_h)
                forward_curr = np.cross(up_stage1, pelvis_curr_lr_h)
                forward_target /= max(np.linalg.norm(forward_target), 1e-6)
                forward_curr /= max(np.linalg.norm(forward_curr), 1e-6)
                pitch_target = float(np.arctan2(np.dot(trunk_target_unit, forward_target), np.dot(trunk_target_unit, up_stage1)))
                pitch_curr = float(np.arctan2(np.dot(trunk_curr_unit, forward_curr), np.dot(trunk_curr_unit, up_stage1)))
                pitch_err = pitch_target - pitch_curr
                bend_target = float(np.arctan2(np.dot(trunk_target_unit, pelvis_target_lr_h), np.dot(trunk_target_unit, up_stage1)))
                bend_curr = float(np.arctan2(np.dot(trunk_curr_unit, pelvis_curr_lr_h), np.dot(trunk_curr_unit, up_stage1)))
                bend_err = bend_target - bend_curr
                if shoulder_target_h_norm > 1e-6 and shoulder_curr_h_norm > 1e-6:
                    shoulder_target_lr_h /= shoulder_target_h_norm
                    shoulder_curr_lr_h /= shoulder_curr_h_norm
                    target_twist = float(
                        np.arctan2(
                            np.dot(np.cross(pelvis_target_lr_h, shoulder_target_lr_h), up_stage1),
                            np.clip(np.dot(pelvis_target_lr_h, shoulder_target_lr_h), -1.0, 1.0),
                        )
                    )
                    curr_twist = float(
                        np.arctan2(
                            np.dot(np.cross(pelvis_curr_lr_h, shoulder_curr_lr_h), up_stage1),
                            np.clip(np.dot(pelvis_curr_lr_h, shoulder_curr_lr_h), -1.0, 1.0),
                        )
                    )
                    twist_err = target_twist - curr_twist

            weak_geom_scale = float(np.clip(target_conf, 0.0, 1.0))
            pelvis_rel = float(np.mean(dof_reliability[[dof_index[name] for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation") if name in dof_index]]) if any(name in dof_index for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation")) else 0.0)
            trunk_names = [name for name in ("lumbar_extension", "lumbar_bending", "lumbar_twist", "thorax_extension", "thorax_bending", "thorax_twist") if name in dof_index]
            trunk_rel = float(np.mean(dof_reliability[[dof_index[name] for name in trunk_names]]) if len(trunk_names) > 0 else 0.0)
            pelvis_gain = weak_geom_scale * np.clip(1.0 - pelvis_rel, 0.0, 1.0)
            trunk_gain = weak_geom_scale * np.clip(1.0 - trunk_rel, 0.0, 1.0)

            if "pelvis_rotation" in dof_index:
                q_hat[dof_index["pelvis_rotation"]] += ORI_PELVIS_YAW_GAIN * pelvis_gain * np.clip(yaw_err, -0.50, 0.50)
            if "pelvis_list" in dof_index:
                q_hat[dof_index["pelvis_list"]] += ORI_PELVIS_ROLL_GAIN * pelvis_gain * np.clip(roll_err, -0.35, 0.35)
            if "pelvis_tilt" in dof_index:
                q_hat[dof_index["pelvis_tilt"]] += ORI_PELVIS_TILT_GAIN * pelvis_gain * np.clip(pitch_err, -0.35, 0.35)
            if "lumbar_extension" in dof_index:
                q_hat[dof_index["lumbar_extension"]] += 0.65 * ORI_TRUNK_EXT_GAIN * trunk_gain * np.clip(pitch_err, -0.45, 0.45)
            if "thorax_extension" in dof_index:
                q_hat[dof_index["thorax_extension"]] += 0.35 * ORI_TRUNK_EXT_GAIN * trunk_gain * np.clip(pitch_err, -0.45, 0.45)
            if "lumbar_bending" in dof_index:
                q_hat[dof_index["lumbar_bending"]] += 0.60 * ORI_TRUNK_BEND_GAIN * trunk_gain * np.clip(bend_err, -0.35, 0.35)
            if "thorax_bending" in dof_index:
                q_hat[dof_index["thorax_bending"]] += 0.40 * ORI_TRUNK_BEND_GAIN * trunk_gain * np.clip(bend_err, -0.35, 0.35)
            if "lumbar_twist" in dof_index:
                q_hat[dof_index["lumbar_twist"]] += 0.55 * ORI_TRUNK_TWIST_GAIN * trunk_gain * np.clip(twist_err, -0.45, 0.45)
            if "thorax_twist" in dof_index:
                q_hat[dof_index["thorax_twist"]] += 0.45 * ORI_TRUNK_TWIST_GAIN * trunk_gain * np.clip(twist_err, -0.45, 0.45)
            q_hat = np.clip(q_hat, q_lower, q_upper)
        dq_hat = dof_reliability * ((q_hat - q_prev) / max(dt_sim, 1e-3)) + (1.0 - dof_reliability) * dq_pred
        if dq_meas is not None and dq_weights is not None:
            dq_sensor = np.where(np.isfinite(dq_meas), dq_meas, dq_hat)
            blend = np.clip(dq_weights / np.maximum(dq_weights + 1.0, EPS), 0.0, 1.0)
            dq_hat = blend * dq_sensor + (1.0 - blend) * dq_hat
        q_kin_prev = np.array(state.get("q_kin", q_prev), dtype=float)
        dq_kin_prev = np.array(state.get("dq_kin", dq_prev), dtype=float)
        ddq_kin_prev = np.array(state.get("ddq_kin", ddq_prev_full), dtype=float)
        weak_low_rel = np.clip((0.60 - dof_reliability) / 0.60, 0.0, 1.0)
        if np.any(pelvis_orient_mask):
            gain = 0.35 * weak_low_rel[pelvis_orient_mask]
            q_hat[pelvis_orient_mask] = (1.0 - gain) * q_hat[pelvis_orient_mask] + gain * q_pred[pelvis_orient_mask]
            dq_hat[pelvis_orient_mask] = (1.0 - gain) * dq_hat[pelvis_orient_mask] + gain * dq_pred[pelvis_orient_mask]
        if np.any(trunk_mask):
            gain = 0.45 * weak_low_rel[trunk_mask]
            q_hat[trunk_mask] = (1.0 - gain) * q_hat[trunk_mask] + gain * q_kin_prev[trunk_mask]
            dq_hat[trunk_mask] = (1.0 - gain) * dq_hat[trunk_mask] + gain * dq_kin_prev[trunk_mask]
        if np.any(scapula_mask):
            gain = 0.60 * weak_low_rel[scapula_mask]
            q_hat[scapula_mask] = (1.0 - gain) * q_hat[scapula_mask] + gain * q_kin_prev[scapula_mask]
            dq_hat[scapula_mask] = (1.0 - gain) * dq_hat[scapula_mask] + gain * dq_kin_prev[scapula_mask]
        if np.any(shoulder_mask):
            gain = 0.35 * weak_low_rel[shoulder_mask]
            q_hat[shoulder_mask] = (1.0 - gain) * q_hat[shoulder_mask] + gain * q_kin_prev[shoulder_mask]
            dq_hat[shoulder_mask] = (1.0 - gain) * dq_hat[shoulder_mask] + gain * dq_kin_prev[shoulder_mask]
        q_hat = np.clip(q_hat, q_lower, q_upper)
        ddq_hat = (dq_hat - dq_prev) / max(dt_sim, 1e-3)

        if use_stage1_kin_filter:
            q_kin_pred = q_kin_prev + dt_sim * dq_kin_prev + 0.5 * dt_sim * dt_sim * ddq_kin_prev
            dq_kin_pred = dq_kin_prev + dt_sim * ddq_kin_prev
            q_residual = q_hat - q_kin_pred
            kin_rel = np.clip(dof_reliability, 0.0, 1.0)
            beta_vec = KIN_BETA_JOINT + 0.08 * kin_rel
            gamma_vec = KIN_GAMMA_JOINT + 0.012 * kin_rel
            beta_vec[:3] = KIN_BETA_POS + 0.05 * kin_rel[:3]
            gamma_vec[:3] = KIN_GAMMA_POS + 0.010 * kin_rel[:3]
            beta_vec[3:6] = KIN_BETA_ROOT_ROT + 0.05 * kin_rel[3:6]
            gamma_vec[3:6] = KIN_GAMMA_ROOT_ROT + 0.008 * kin_rel[3:6]
            beta_vec[pelvis_orient_mask] *= 0.65
            gamma_vec[pelvis_orient_mask] *= 0.55
            beta_vec[trunk_mask] *= 0.42
            gamma_vec[trunk_mask] *= 0.34
            beta_vec[scapula_mask] *= 0.35
            gamma_vec[scapula_mask] *= 0.28
            beta_vec[shoulder_mask] *= 0.70
            gamma_vec[shoulder_mask] *= 0.60
            beta_vec = np.clip(beta_vec, 0.0, 0.45)
            gamma_vec = np.clip(gamma_vec, 0.0, 0.08)
            q_kin = q_hat.copy()
            dq_kin = dq_kin_pred + (beta_vec / max(dt_sim, 1e-3)) * q_residual
            ddq_kin = ddq_kin_prev + (2.0 * gamma_vec / max(dt_sim * dt_sim, 1e-6)) * q_residual
            dq_kin = 0.65 * dq_kin + 0.35 * dq_hat
            ddq_kin = 0.70 * ddq_kin + 0.30 * ddq_hat
            if np.any(pelvis_orient_mask):
                gain = 0.28 * weak_low_rel[pelvis_orient_mask]
                dq_kin[pelvis_orient_mask] = (1.0 - gain) * dq_kin[pelvis_orient_mask] + gain * dq_kin_prev[pelvis_orient_mask]
                ddq_kin[pelvis_orient_mask] = (1.0 - gain) * ddq_kin[pelvis_orient_mask] + gain * ddq_kin_prev[pelvis_orient_mask]
            if np.any(trunk_mask):
                gain = 0.45 * weak_low_rel[trunk_mask]
                dq_kin[trunk_mask] = (1.0 - gain) * dq_kin[trunk_mask] + gain * dq_kin_prev[trunk_mask]
                ddq_kin[trunk_mask] = (1.0 - gain) * ddq_kin[trunk_mask] + gain * ddq_kin_prev[trunk_mask]
            if np.any(scapula_mask):
                gain = 0.50 * weak_low_rel[scapula_mask]
                dq_kin[scapula_mask] = (1.0 - gain) * dq_kin[scapula_mask] + gain * dq_kin_prev[scapula_mask]
                ddq_kin[scapula_mask] = (1.0 - gain) * ddq_kin[scapula_mask] + gain * ddq_kin_prev[scapula_mask]
            if np.any(shoulder_mask):
                gain = 0.25 * weak_low_rel[shoulder_mask]
                dq_kin[shoulder_mask] = (1.0 - gain) * dq_kin[shoulder_mask] + gain * dq_kin_prev[shoulder_mask]
                ddq_kin[shoulder_mask] = (1.0 - gain) * ddq_kin[shoulder_mask] + gain * ddq_kin_prev[shoulder_mask]
            dq_kin = np.clip(dq_kin, -dq_cap, dq_cap)
            ddq_filter_cap = np.ones(n_dof_full, dtype=float) * 80.0
            ddq_filter_cap[:3] = 35.0
            ddq_filter_cap[3:6] = 30.0
            ddq_kin = np.clip(ddq_kin, -ddq_filter_cap, ddq_filter_cap)
            ddq_stage1 = 0.70 * ddq_kin + 0.30 * ddq_hat
        else:
            q_kin = q_hat.copy()
            dq_kin = dq_hat.copy()
            ddq_kin = ddq_hat.copy()
            ddq_stage1 = ddq_hat
        model.setPositions(q_kin)
        model.setVelocities(dq_kin)
        model.setAccelerations(ddq_stage1)

        gravity = np.array(model.getGravity(), dtype=float).reshape(3)
        mass_kg = float(model.getMass())
        body_weight = mass_kg * max(np.linalg.norm(gravity), EPS)
        if np.linalg.norm(gravity) < EPS:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up = -gravity / max(np.linalg.norm(gravity), EPS)

        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(ref, up)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        tangent_1 = np.cross(up, ref)
        tangent_1 /= max(np.linalg.norm(tangent_1), EPS)
        tangent_2 = np.cross(up, tangent_1)
        tangent_2 /= max(np.linalg.norm(tangent_2), EPS)
        ground_basis = np.vstack([tangent_1, tangent_2, up])
        com_position = np.array(model.getCOM(), dtype=float).reshape(3)
        com_velocity = np.array(model.getCOMLinearVelocity(), dtype=float).reshape(3)
        com_acceleration = np.array(model.getCOMLinearAcceleration(), dtype=float).reshape(3)
        support_force_target = mass_kg * (com_acceleration - gravity)
        support_force_local = ground_basis @ support_force_target
        prev_support_force_local = ground_basis @ (
            np.array(state["foot_forces"]["left"], dtype=float) + np.array(state["foot_forces"]["right"], dtype=float)
        )
        force_step_limit = np.array([0.25 * body_weight, 0.25 * body_weight, 0.50 * body_weight], dtype=float)
        support_force_local = prev_support_force_local + np.clip(
            support_force_local - prev_support_force_local,
            -force_step_limit,
            force_step_limit,
        )
        support_force_local[2] = np.clip(support_force_local[2], 0.0, 3.5 * body_weight)
        tangential_norm = float(np.linalg.norm(support_force_local[:2]))
        tangential_limit = float(mu) * support_force_local[2]
        if tangential_norm > tangential_limit > 0.0:
            support_force_local[:2] *= tangential_limit / tangential_norm
        if support_force_local[2] <= 0.03 * body_weight:
            support_force_local[:] = 0.0
        support_force_target = ground_basis.T @ support_force_local
        support_ratio = float(support_force_local[2] / max(body_weight, EPS))
        com_velocity_plane = com_velocity - float(np.dot(com_velocity, up)) * up
        com_speed_plane = float(np.linalg.norm(com_velocity_plane))
        quasi_static_support = float(np.clip(1.0 - com_speed_plane / 0.40, 0.0, 1.0))

        joint_confidence = np.mean(measurement_weights, axis=1)
        joint_conf_lookup = {}
        for i, joint in enumerate(measurement_joints):
            joint_conf_lookup[joint.getName()] = float(joint_confidence[i])

        raw_heights = []
        foot_cues = {}
        for side in FOOT_BODIES:
            heel_name, toe_name = FOOT_BODIES[side]
            heel_body = model.getBodyNode(heel_name)
            toe_body = model.getBodyNode(toe_name)
            heel_offset = np.zeros(3, dtype=float)
            toe_offset = np.zeros(3, dtype=float)

            heel_world = np.array(heel_body.getWorldTransform().matrix(), dtype=float) @ np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
            toe_world = np.array(toe_body.getWorldTransform().matrix(), dtype=float) @ np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
            heel_position = heel_world[:3]
            toe_position = toe_world[:3]

            heel_jacobian = np.array(model.getLinearJacobian(heel_body, heel_offset), dtype=float)
            toe_jacobian = np.array(model.getLinearJacobian(toe_body, toe_offset), dtype=float)
            heel_velocity = heel_jacobian @ dq_kin
            toe_velocity = toe_jacobian @ dq_kin
            heel_bias = np.array(model.getLinearJacobianDeriv(heel_body, heel_offset), dtype=float) @ dq_kin
            toe_bias = np.array(model.getLinearJacobianDeriv(toe_body, toe_offset), dtype=float) @ dq_kin

            raw_heights.append(float(np.dot(heel_position, up)))
            raw_heights.append(float(np.dot(toe_position, up)))

            foot_cues[side] = {
                "heel_body": heel_body,
                "toe_body": toe_body,
                "heel_offset": heel_offset,
                "toe_offset": toe_offset,
                "heel_position": heel_position,
                "toe_position": toe_position,
                "heel_jacobian": heel_jacobian,
                "toe_jacobian": toe_jacobian,
                "heel_velocity": heel_velocity,
                "toe_velocity": toe_velocity,
                "heel_bias": heel_bias,
                "toe_bias": toe_bias,
            }

        finite_heights = np.array([value for value in raw_heights if np.isfinite(value)], dtype=float)
        if finite_heights.size == 0:
            floor_height = 0.0 if not np.isfinite(state["floor_height"]) else float(state["floor_height"])
        else:
            floor_candidate = float(np.min(finite_heights))
            if not np.isfinite(state["floor_height"]):
                floor_height = floor_candidate
            elif floor_candidate < state["floor_height"]:
                floor_height = float(0.7 * state["floor_height"] + 0.3 * floor_candidate)
            else:
                floor_height = float(state["floor_height"] + 0.02 * (floor_candidate - state["floor_height"]))

        for side in FOOT_BODIES:
            cue = foot_cues[side]
            heel_height = float(np.dot(cue["heel_position"], up) - floor_height)
            toe_height = float(np.dot(cue["toe_position"], up) - floor_height)
            if heel_height <= toe_height:
                height = heel_height
                vertical_velocity = float(np.dot(cue["heel_velocity"], up))
            else:
                height = toe_height
                vertical_velocity = float(np.dot(cue["toe_velocity"], up))
            cue["height"] = height
            cue["vertical_velocity"] = vertical_velocity
            cue["anchor_position"] = 0.5 * (cue["heel_position"] + cue["toe_position"])
            cue["anchor_jacobian"] = 0.5 * (cue["heel_jacobian"] + cue["toe_jacobian"])
            cue["anchor_bias"] = 0.5 * (cue["heel_bias"] + cue["toe_bias"])
            cue["anchor_angular_jacobian"] = np.array(model.getAngularJacobian(cue["heel_body"]), dtype=float)
            cue["anchor_angular_bias"] = np.array(model.getAngularJacobianDeriv(cue["heel_body"]), dtype=float) @ dq_kin
            cue["anchor_angular_velocity"] = cue["anchor_angular_jacobian"] @ dq_kin
            foot_forward = cue["toe_position"] - cue["heel_position"]
            foot_forward -= float(np.dot(foot_forward, up)) * up
            foot_forward_norm = float(np.linalg.norm(foot_forward))
            if foot_forward_norm < 1e-6:
                foot_forward = tangent_1.copy()
            else:
                foot_forward /= foot_forward_norm
            foot_lateral = np.cross(up, foot_forward)
            foot_lateral_norm = float(np.linalg.norm(foot_lateral))
            if foot_lateral_norm < 1e-6:
                foot_lateral = tangent_2.copy()
            else:
                foot_lateral /= foot_lateral_norm
            cue["foot_basis"] = np.vstack([foot_forward, foot_lateral, up])
            foot_span = float(np.linalg.norm(cue["toe_position"] - cue["heel_position"] - np.dot(cue["toe_position"] - cue["heel_position"], up) * up))
            cue["cop_half_length"] = float(np.clip(0.5 * foot_span + 0.015, 0.05, 0.14))
            cue["cop_half_width"] = float(np.clip(0.35 * cue["cop_half_length"], 0.025, 0.06))
            cue["torsion_limit"] = float(0.5 * (cue["cop_half_length"] + cue["cop_half_width"]))
            ankle_name = "ankle_l" if side == "left" else "ankle_r"
            cue["joint_confidence"] = joint_conf_lookup.get(ankle_name, float(np.mean(joint_confidence)) if joint_confidence.size > 0 else 0.0)
            prev_vertical_load = float(np.dot(state["foot_forces"][side], up)) / max(body_weight, EPS)
            cue["prev_load_ratio"] = float(np.clip(prev_vertical_load, 0.0, 1.5))
            height_score = np.clip(1.0 - max(height, 0.0) / 0.16, 0.0, 1.0)
            velocity_score = np.clip(1.0 - abs(vertical_velocity) / 1.4, 0.0, 1.0)
            support_bonus = np.clip(support_ratio / 0.45, 0.0, 1.0)
            cue["confidence"] = float(
                np.clip(
                    0.40 * cue["joint_confidence"]
                    + 0.30 * height_score
                    + 0.15 * velocity_score
                    + 0.10 * float(state["contact_state"][side])
                    + 0.05 * support_bonus,
                    0.0,
                    1.0,
                )
            )
            cue["score"] = cue["confidence"]
            prev_prob = float(state.get("contact_prob", {}).get(side, 1.0 if state["contact_state"][side] else 0.0))
            prob = (
                0.50 * cue["score"]
                + 0.20 * height_score
                + 0.10 * velocity_score
                + 0.12 * prev_prob
                + 0.08 * support_bonus
            )
            if state["contact_state"][side]:
                prob += 0.05
            cue["contact_prob"] = float(np.clip(prob, 0.0, 1.0))
            if state["contact_state"][side]:
                cue["contact"] = bool(cue["score"] >= CONTACT_OFF_SCORE and height <= 0.18)
            else:
                cue["contact"] = bool(cue["score"] >= CONTACT_ON_SCORE and height <= 0.14)

        contact_state = {
            "left": bool(foot_cues["left"]["contact"]),
            "right": bool(foot_cues["right"]["contact"]),
        }
        contact_prob = {
            "left": float(foot_cues["left"]["contact_prob"]),
            "right": float(foot_cues["right"]["contact_prob"]),
        }
        if support_ratio >= 0.08 and not (contact_state["left"] or contact_state["right"]):
            best_side = min(
                ["left", "right"],
                key=lambda side: (
                    foot_cues[side]["height"] - 0.05 * foot_cues[side]["score"],
                    abs(foot_cues[side]["vertical_velocity"]),
                ),
            )
            contact_state[best_side] = True
            contact_prob[best_side] = max(contact_prob[best_side], 0.30)
        if support_ratio >= 0.25:
            for side in ["left", "right"]:
                if foot_cues[side]["score"] >= 0.40 and foot_cues[side]["height"] <= 0.18:
                    contact_state[side] = True
                    contact_prob[side] = max(contact_prob[side], 0.40)
        if support_ratio <= 0.02:
            for side in ["left", "right"]:
                if foot_cues[side]["score"] < 0.75:
                    contact_state[side] = False
                    contact_prob[side] = min(contact_prob[side], 0.10)
        if quasi_static_support >= 0.70:
            left_load = foot_cues["left"]["prev_load_ratio"]
            right_load = foot_cues["right"]["prev_load_ratio"]
            load_gap = abs(left_load - right_load)
            if load_gap >= 0.12:
                dominant = "left" if left_load > right_load else "right"
                weak = "right" if dominant == "left" else "left"
                if (
                    foot_cues[dominant]["height"] <= 0.12
                    and foot_cues[weak]["height"] >= foot_cues[dominant]["height"] + 0.01
                    and foot_cues[weak]["score"] < 0.75
                    and foot_cues[weak]["prev_load_ratio"] < 0.08
                ):
                    contact_prob[weak] = min(contact_prob[weak], 0.35)
                    if foot_cues[weak]["vertical_velocity"] > -0.05:
                        contact_state[weak] = False
        for side in ["left", "right"]:
            if contact_state[side]:
                contact_prob[side] = max(contact_prob[side], 0.55)
            else:
                contact_prob[side] = min(contact_prob[side], 0.45)

        support_wrench_split = np.zeros(12, dtype=float)
        active_support_sides = [side for side in ["left", "right"] if contact_prob[side] > 0.05]
        if support_force_local[2] > 0.0 and len(active_support_sides) > 0:
            com_plane = com_position - (float(np.dot(com_position, up)) - float(floor_height)) * up
            weights = []
            for side in active_support_sides:
                cue = foot_cues[side]
                anchor_plane = cue["anchor_position"] - (float(np.dot(cue["anchor_position"], up)) - float(floor_height)) * up
                planar_delta = anchor_plane - com_plane
                planar_delta -= float(np.dot(planar_delta, up)) * up
                distance = float(np.linalg.norm(planar_delta))
                weight = contact_prob[side] * (0.20 + cue["score"] + 0.15 * float(state["contact_state"][side])) / max(distance + 0.05, 1e-3)
                weights.append(weight)
            weights = np.array(weights, dtype=float)
            weights /= max(float(np.sum(weights)), EPS)
            for idx, side in enumerate(active_support_sides):
                side_force_local = weights[idx] * support_force_local
                side_tangential_norm = float(np.linalg.norm(side_force_local[:2]))
                side_tangential_limit = float(mu) * side_force_local[2]
                if side_tangential_norm > side_tangential_limit > 0.0:
                    side_force_local[:2] *= side_tangential_limit / side_tangential_norm
                side_force_world = ground_basis.T @ side_force_local
                cue = foot_cues[side]
                side_moment_world = np.zeros(3, dtype=float)
                wrench_slice = slice(0, 6) if side == "left" else slice(6, 12)
                support_wrench_split[wrench_slice] = np.concatenate([side_force_world, side_moment_world])

        model.setPositions(q_kin)
        model.setVelocities(dq_kin)

        M_full = np.array(model.getMassMatrix(), dtype=float)
        h_full = np.array(model.getCoriolisAndGravityForces(), dtype=float).reshape(-1)
        M = M_full[np.ix_(active_dofs, active_dofs)]
        h = h_full[active_dofs]

        n_dof = len(active_dofs)
        n_act = len(active_act_dofs)
        q_target = q_kin[active_dofs]
        dq_target = np.clip(dq_kin[active_dofs], -dq_cap[active_dofs], dq_cap[active_dofs])
        if use_stage1_kin_filter:
            ddq_target = (
                0.45 * ddq_kin[active_dofs]
                + 0.25 * ddq_hat[active_dofs]
                + 0.30 * np.array(state.get("ddq_kin", state["ddq"]), dtype=float)[active_dofs]
            )
        else:
            ddq_target = 0.60 * ddq_hat[active_dofs] + 0.40 * state["ddq"][active_dofs]
        rel = dof_reliability[active_dofs]

        w_q = 40.0 * (0.6 + 0.8 * rel)
        w_dq = 8.0 * (0.4 + 0.6 * rel)
        w_ddq = 1.8 * (0.35 + 0.65 * rel)
        if n_dof >= 6:
            w_q[:6] *= 4.0
            w_dq[:6] *= 2.0
            w_ddq[:6] *= 1.5

        J_wrench = np.zeros((n_dof, 12), dtype=float)
        for foot_idx, side in enumerate(["left", "right"]):
            J_linear = foot_cues[side]["anchor_jacobian"][:, active_dofs]
            J_angular = foot_cues[side]["anchor_angular_jacobian"][:, active_dofs]
            J_wrench[:, 6 * foot_idx : 6 * foot_idx + 3] = J_linear.T
            J_wrench[:, 6 * foot_idx + 3 : 6 * foot_idx + 6] = J_angular.T
        active_root_rows = np.where(active_dofs < 6)[0]
        active_act_rows = np.where(active_dofs >= 6)[0]
        n_root = len(active_root_rows)
        root_select = np.zeros((n_root, 6), dtype=float)
        for row_idx, active_row in enumerate(active_root_rows):
            root_select[row_idx, active_dofs[active_row]] = 1.0
        M_root = M[active_root_rows, :] if n_root > 0 else np.zeros((0, n_dof), dtype=float)
        h_root = h[active_root_rows] if n_root > 0 else np.zeros(0, dtype=float)
        J_root = J_wrench[active_root_rows, :] if n_root > 0 else np.zeros((0, 12), dtype=float)
        M_act = M[active_act_rows, :] if n_act > 0 else np.zeros((0, n_dof), dtype=float)
        h_act = h[active_act_rows] if n_act > 0 else np.zeros(0, dtype=float)
        J_act = J_wrench[active_act_rows, :] if n_act > 0 else np.zeros((0, 12), dtype=float)
        active_dof_names = [model.getDofByIndex(int(dof)).getName() for dof in active_dofs]
        active_act_dof_names = [model.getDofByIndex(int(dof)).getName() for dof in active_act_dofs]
        tau_abs_limits = np.ones(n_act, dtype=float) * 2500.0
        tau_rate_limits = np.ones(n_act, dtype=float) * 1200.0
        for i, name in enumerate(active_act_dof_names):
            if "hip_" in name or "knee_" in name:
                tau_abs_limits[i] = 5000.0
                tau_rate_limits[i] = 2200.0
            elif "lumbar_" in name or "thorax_" in name:
                tau_abs_limits[i] = 2500.0
                tau_rate_limits[i] = 1200.0
            elif "shoulder_" in name or "scapula_" in name:
                tau_abs_limits[i] = 1600.0
                tau_rate_limits[i] = 800.0
            elif "elbow_" in name:
                tau_abs_limits[i] = 900.0
                tau_rate_limits[i] = 500.0
            elif "ankle_" in name or "subtalar_" in name or "mtp_" in name:
                tau_abs_limits[i] = 1400.0
                tau_rate_limits[i] = 700.0

        q_margin = 1e-3
        ddq_pos_upper = 2.0 * (q_upper[active_dofs] - q_margin - q_prev[active_dofs] - dt_sim * dq_prev[active_dofs]) / max(dt_sim * dt_sim, EPS)
        ddq_pos_lower = 2.0 * (q_lower[active_dofs] + q_margin - q_prev[active_dofs] - dt_sim * dq_prev[active_dofs]) / max(dt_sim * dt_sim, EPS)
        ddq_vel_upper = (dq_cap[active_dofs] - dq_prev[active_dofs]) / max(dt_sim, EPS)
        ddq_vel_lower = (-dq_cap[active_dofs] - dq_prev[active_dofs]) / max(dt_sim, EPS)
        ddq_abs = np.ones(n_dof, dtype=float) * 90.0
        if n_dof >= 6:
            ddq_abs[:3] = 50.0
            ddq_abs[3:6] = 40.0
        ddq_upper = np.minimum.reduce([ddq_pos_upper, ddq_vel_upper, ddq_abs])
        ddq_lower = np.maximum.reduce([ddq_pos_lower, ddq_vel_lower, -ddq_abs])
        bad = ddq_lower > ddq_upper
        if np.any(bad):
            mid = 0.5 * (ddq_lower + ddq_upper)
            ddq_lower[bad] = mid[bad] - 1e-3
            ddq_upper[bad] = mid[bad] + 1e-3

        valid = measurement_weights.reshape(-1) > 0.0
        J_meas = np.zeros((int(np.sum(valid)), n_dof), dtype=float)
        meas_b = np.zeros(int(np.sum(valid)), dtype=float)
        meas_w = np.zeros(int(np.sum(valid)), dtype=float)
        if np.any(valid):
            x_hat = np.array(model.getJointWorldPositions(measurement_joints), dtype=float).reshape(-1)
            J_meas_full = np.array(model.getJointWorldPositionsJacobianWrtJointPositions(measurement_joints), dtype=float)[:, active_dofs]
            J_meas = J_meas_full[valid, :]
            meas_b = x_step.reshape(-1)[valid] - x_hat[valid] - J_meas @ (
                q_prev[active_dofs] + dt_sim * dq_prev[active_dofs] - q_kin[active_dofs]
            )
            meas_w = 40.0 * np.maximum(measurement_weights.reshape(-1)[valid], 1e-6)

        dyn_key = (n_dof, n_act, n_root, len(measurement_joints), round(float(mu), 6))
        if dyn_key not in DYN_QP_CACHE:
            ddq_var = cp.Variable(n_dof)
            wrench_var = cp.Variable(12)
            root_var = cp.Variable(6)
            M_root_param = cp.Parameter((n_root, n_dof))
            h_root_param = cp.Parameter(n_root)
            J_root_param = cp.Parameter((n_root, 12))
            M_act_param = cp.Parameter((n_act, n_dof))
            h_act_param = cp.Parameter(n_act)
            J_act_param = cp.Parameter((n_act, 12))
            ddq_lower_param = cp.Parameter(n_dof)
            ddq_upper_param = cp.Parameter(n_dof)
            A_q_param = cp.Parameter((n_dof, n_dof))
            b_q_param = cp.Parameter(n_dof)
            A_dq_param = cp.Parameter((n_dof, n_dof))
            b_dq_param = cp.Parameter(n_dof)
            A_ddq_param = cp.Parameter((n_dof, n_dof))
            b_ddq_param = cp.Parameter(n_dof)
            A_ddq_s_param = cp.Parameter((n_dof, n_dof))
            b_ddq_s_param = cp.Parameter(n_dof)
            tau_prev_param = cp.Parameter(n_act)
            wrench_prev_param = cp.Parameter(12)
            root_prev_param = cp.Parameter(6)
            A_wrench_net_param = cp.Parameter((3, 12))
            b_wrench_net_param = cp.Parameter(3)
            A_wrench_split_param = cp.Parameter((12, 12))
            b_wrench_split_param = cp.Parameter(12)
            tau_smooth_const = np.diag(np.ones(n_act, dtype=float) * 0.14)
            tau_reg_const = np.diag(np.ones(n_act, dtype=float) * 0.008)
            wrench_smooth_weights = np.array([0.08, 0.08, 0.10, 0.16, 0.16, 0.08] * 2, dtype=float)
            wrench_reg_weights = np.array([0.006, 0.006, 0.008, 0.016, 0.016, 0.008] * 2, dtype=float)
            wrench_smooth_const = np.diag(wrench_smooth_weights)
            wrench_reg_const = np.diag(wrench_reg_weights)
            root_smooth_const = np.diag(np.array([0.08, 0.08, 0.08, 0.05, 0.05, 0.05], dtype=float))
            root_const = np.diag(np.array([0.95, 0.95, 0.95, 0.45, 0.45, 0.45], dtype=float))
            tau_aff = M_act_param @ ddq_var + h_act_param - J_act_param @ wrench_var
            objective = cp.sum_squares(A_q_param @ ddq_var - b_q_param)
            objective += cp.sum_squares(A_dq_param @ ddq_var - b_dq_param)
            objective += cp.sum_squares(A_ddq_param @ ddq_var - b_ddq_param)
            objective += cp.sum_squares(A_ddq_s_param @ ddq_var - b_ddq_s_param)
            objective += cp.sum_squares(tau_smooth_const @ tau_aff - tau_smooth_const @ tau_prev_param)
            objective += cp.sum_squares(tau_reg_const @ tau_aff)
            objective += cp.sum_squares(wrench_smooth_const @ wrench_var - wrench_smooth_const @ wrench_prev_param)
            objective += cp.sum_squares(wrench_reg_const @ wrench_var)
            wrench_net_expr = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=float,
            ) @ wrench_var
            objective += cp.sum_squares(A_wrench_net_param @ wrench_var - b_wrench_net_param)
            objective += cp.sum_squares(A_wrench_split_param @ wrench_var - b_wrench_split_param)
            objective += cp.sum_squares(root_smooth_const @ root_var - root_smooth_const @ root_prev_param)
            objective += cp.sum_squares(root_const @ root_var)
            constraints = [ddq_var >= ddq_lower_param, ddq_var <= ddq_upper_param]
            if n_root > 0:
                constraints += [M_root_param @ ddq_var + h_root_param == J_root_param @ wrench_var + root_select @ root_var]

            meas_rows = 3 * len(measurement_joints)
            J_meas_param = cp.Parameter((meas_rows, n_dof))
            meas_b_param = cp.Parameter(meas_rows)
            objective += cp.sum_squares(J_meas_param @ ddq_var - meas_b_param)

            side_params = {}
            for side, wrench_slice in [("left", slice(0, 6)), ("right", slice(6, 12))]:
                A_acc_param = cp.Parameter((3, n_dof))
                b_acc_param = cp.Parameter(3)
                A_vel_param = cp.Parameter((3, n_dof))
                b_vel_param = cp.Parameter(3)
                A_ang_acc_param = cp.Parameter((3, n_dof))
                b_ang_acc_param = cp.Parameter(3)
                A_ang_vel_param = cp.Parameter((3, n_dof))
                b_ang_vel_param = cp.Parameter(3)
                fz_upper_param = cp.Parameter(nonneg=True)
                objective += cp.sum_squares(A_acc_param @ ddq_var - b_acc_param)
                objective += cp.sum_squares(A_vel_param @ ddq_var - b_vel_param)
                objective += cp.sum_squares(A_ang_acc_param @ ddq_var - b_ang_acc_param)
                objective += cp.sum_squares(A_ang_vel_param @ ddq_var - b_ang_vel_param)
                local_force = wrench_var[wrench_slice.start : wrench_slice.start + 3]
                local_moment = wrench_var[wrench_slice.start + 3 : wrench_slice.stop]
                constraints += [
                    local_force[2] >= 0.0,
                    local_force[2] <= fz_upper_param + 1e-6,
                    local_force[0] <= float(mu) * local_force[2],
                    -local_force[0] <= float(mu) * local_force[2],
                    local_force[1] <= float(mu) * local_force[2],
                    -local_force[1] <= float(mu) * local_force[2],
                    local_moment[0] <= FOOT_COP_HALF_WIDTH * local_force[2],
                    -local_moment[0] <= FOOT_COP_HALF_WIDTH * local_force[2],
                    local_moment[1] <= FOOT_COP_HALF_LENGTH * local_force[2],
                    -local_moment[1] <= FOOT_COP_HALF_LENGTH * local_force[2],
                    local_moment[2] <= FOOT_TORSION_RADIUS * local_force[2],
                    -local_moment[2] <= FOOT_TORSION_RADIUS * local_force[2],
                ]
                side_params[side] = {
                    "A_acc": A_acc_param,
                    "b_acc": b_acc_param,
                    "A_vel": A_vel_param,
                    "b_vel": b_vel_param,
                    "A_ang_acc": A_ang_acc_param,
                    "b_ang_acc": b_ang_acc_param,
                    "A_ang_vel": A_ang_vel_param,
                    "b_ang_vel": b_ang_vel_param,
                    "fz_upper": fz_upper_param,
                }

            DYN_QP_CACHE[dyn_key] = {
                "problem": cp.Problem(cp.Minimize(objective), constraints),
                "ddq": ddq_var,
                "wrench": wrench_var,
                "root": root_var,
                "M_root": M_root_param,
                "h_root": h_root_param,
                "J_root": J_root_param,
                "M_act": M_act_param,
                "h_act": h_act_param,
                "J_act": J_act_param,
                "ddq_lower": ddq_lower_param,
                "ddq_upper": ddq_upper_param,
                "A_q": A_q_param,
                "b_q": b_q_param,
                "A_dq": A_dq_param,
                "b_dq": b_dq_param,
                "A_ddq": A_ddq_param,
                "b_ddq": b_ddq_param,
                "A_ddq_s": A_ddq_s_param,
                "b_ddq_s": b_ddq_s_param,
                "tau_prev": tau_prev_param,
                "wrench_prev": wrench_prev_param,
                "root_prev": root_prev_param,
                "A_wrench_net": A_wrench_net_param,
                "b_wrench_net": b_wrench_net_param,
                "A_wrench_split": A_wrench_split_param,
                "b_wrench_split": b_wrench_split_param,
                "J_meas": J_meas_param,
                "meas_b": meas_b_param,
                "side_params": side_params,
            }

        problem = DYN_QP_CACHE[dyn_key]
        problem["M_root"].value = M_root + PARAM_DENSE_EPS
        problem["h_root"].value = h_root
        problem["J_root"].value = J_root + PARAM_DENSE_EPS
        problem["M_act"].value = M_act + PARAM_DENSE_EPS
        problem["h_act"].value = h_act
        problem["J_act"].value = J_act + PARAM_DENSE_EPS
        problem["ddq_lower"].value = ddq_lower
        problem["ddq_upper"].value = ddq_upper
        problem["A_q"].value = np.diag(w_q * (0.5 * dt_sim * dt_sim))
        problem["b_q"].value = w_q * (q_target - q_prev[active_dofs] - dt_sim * dq_prev[active_dofs])
        problem["A_dq"].value = np.diag(w_dq * dt_sim)
        problem["b_dq"].value = w_dq * (dq_target - dq_prev[active_dofs])
        problem["A_ddq"].value = np.diag(w_ddq)
        problem["b_ddq"].value = w_ddq * ddq_target
        ddq_prev_target = np.array(state.get("ddq_kin", state["ddq"]), dtype=float)[active_dofs]
        ddq_smooth_w = 0.35 * (1.3 - 0.7 * rel)
        ddq_smooth_w[: min(6, n_dof)] *= 0.8
        problem["A_ddq_s"].value = np.diag(ddq_smooth_w)
        problem["b_ddq_s"].value = ddq_smooth_w * ddq_prev_target
        if n_act > 0:
            tau_prev_active = state["tau"][active_act_dofs - 6]
            problem["tau_prev"].value = tau_prev_active
        else:
            problem["tau_prev"].value = np.zeros(0, dtype=float)
        wrench_prev = np.concatenate([state["foot_wrenches"]["left"], state["foot_wrenches"]["right"]])
        problem["wrench_prev"].value = wrench_prev
        problem["root_prev"].value = state["root_residual"]
        net_force_w = np.array([0.006, 0.006, 0.024], dtype=float) * (0.2 + min(support_ratio, 1.5))
        split_wrench_w = np.zeros(12, dtype=float)
        split_wrench_w[:6] = np.array([0.010, 0.010, 0.012, 0.012, 0.012, 0.008], dtype=float) * contact_prob["left"] * (0.2 + foot_cues["left"]["score"])
        split_wrench_w[6:12] = np.array([0.010, 0.010, 0.012, 0.012, 0.012, 0.008], dtype=float) * contact_prob["right"] * (0.2 + foot_cues["right"]["score"])
        for side in ["left", "right"]:
            wrench_slice = slice(0, 6) if side == "left" else slice(6, 12)
            prev_moment = state["foot_wrenches"][side][3:6]
            support_wrench_split[wrench_slice.start + 3 : wrench_slice.stop] = 0.8 * prev_moment
        problem["A_wrench_net"].value = np.diag(net_force_w) @ np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        problem["b_wrench_net"].value = net_force_w * support_force_target
        problem["A_wrench_split"].value = np.diag(split_wrench_w)
        problem["b_wrench_split"].value = split_wrench_w * support_wrench_split
        J_meas_full = np.zeros((3 * len(measurement_joints), n_dof), dtype=float)
        meas_b_full = np.zeros(3 * len(measurement_joints), dtype=float)
        if np.any(valid):
            J_meas_full[valid, :] = J_meas
            meas_b_full[valid] = meas_b
        meas_w_full = np.zeros(3 * len(measurement_joints), dtype=float)
        if np.any(valid):
            meas_w_full[valid] = meas_w
        problem["J_meas"].value = np.diag(0.5 * dt_sim * dt_sim * meas_w_full) @ (J_meas_full + PARAM_DENSE_EPS)
        problem["meas_b"].value = meas_w_full * meas_b_full

        for side in ["left", "right"]:
            side_params = problem["side_params"][side]
            J_side = foot_cues[side]["anchor_jacobian"][:, active_dofs]
            J_ang = foot_cues[side]["anchor_angular_jacobian"][:, active_dofs]
            prob = contact_prob[side]
            active_scale = 1.0 if prob > 0.05 else 0.0
            w_acc = np.array([0.04, 0.04, 0.16], dtype=float) * (0.15 + prob)
            w_vel = np.array([0.015, 0.015, 0.06], dtype=float) * (0.15 + prob)
            w_ang_acc = np.array([0.06, 0.06, 0.05], dtype=float) * (0.15 + prob)
            w_ang_vel = np.array([0.05, 0.05, 0.04], dtype=float) * (0.15 + prob)
            side_params["A_acc"].value = np.diag(active_scale * w_acc) @ (J_side + PARAM_DENSE_EPS)
            side_params["b_acc"].value = -(active_scale * w_acc) * foot_cues[side]["anchor_bias"]
            side_params["A_vel"].value = np.diag(active_scale * w_vel) @ (dt_sim * (J_side + PARAM_DENSE_EPS))
            side_params["b_vel"].value = -(active_scale * w_vel) * (J_side @ dq_kin[active_dofs])
            side_params["A_ang_acc"].value = np.diag(active_scale * w_ang_acc) @ (J_ang + PARAM_DENSE_EPS)
            side_params["b_ang_acc"].value = -(active_scale * w_ang_acc) * foot_cues[side]["anchor_angular_bias"]
            side_params["A_ang_vel"].value = np.diag(active_scale * w_ang_vel) @ (dt_sim * (J_ang + PARAM_DENSE_EPS))
            side_params["b_ang_vel"].value = -(active_scale * w_ang_vel) * (J_ang @ dq_kin[active_dofs])
            side_params["fz_upper"].value = active_scale * max(0.35 * body_weight, (1.5 + 8.0 * prob) * body_weight)

        try:
            problem["problem"].solve(
                solver=cp.OSQP,
                warm_start=True,
                ignore_dpp=False,
                verbose=False,
                max_iter=STAGE2_MAX_OSQP_ITERS,
                eps_abs=STAGE2_EPS_ABS,
                eps_rel=STAGE2_EPS_REL,
                polish=False,
            )
        except cp.SolverError:
            try:
                problem["problem"].solve(
                    solver=cp.SCS,
                    warm_start=True,
                    ignore_dpp=False,
                    verbose=False,
                    max_iters=10000,
                    eps=1e-5,
                )
            except cp.SolverError:
                return None

        ddq_active = None if problem["ddq"].value is None else np.array(problem["ddq"].value, dtype=float).reshape(-1)
        wrench_proj = None if problem["wrench"].value is None else np.array(problem["wrench"].value, dtype=float).reshape(-1)
        root_residual = None if problem["root"].value is None else np.array(problem["root"].value, dtype=float).reshape(-1)
        if ddq_active is None or wrench_proj is None or root_residual is None:
            return None

        ddq_full = np.zeros(n_dof_full, dtype=float)
        ddq_full[active_dofs] = ddq_active
        dq_dyn = dq_prev + dt_sim * ddq_full
        q_dyn = q_prev + dt_sim * dq_prev + 0.5 * dt_sim * dt_sim * ddq_full
        q_dyn = np.clip(q_dyn, q_lower, q_upper)
        q_full = q_hat.copy()
        q_full = q_kin.copy()
        dq_full = dq_kin.copy()

        tau_active = M_act @ ddq_active + h_act - J_act @ wrench_proj
        tau_out = np.zeros(n_act_full, dtype=float)
        tau_full = np.zeros(n_dof_full, dtype=float)
        tau_full[:6] = root_residual
        for i, dof_idx in enumerate(active_act_dofs):
            tau_out[dof_idx - 6] = tau_active[i]
            tau_full[dof_idx] = tau_active[i]

        foot_wrenches = {
            "left": wrench_proj[:6].copy(),
            "right": wrench_proj[6:12].copy(),
        }
        foot_forces = {
            "left": foot_wrenches["left"][:3].copy(),
            "right": foot_wrenches["right"][:3].copy(),
        }

        model.setPositions(q_full)
        model.setVelocities(dq_full)
        model.setAccelerations(ddq_full)
        model.setControlForces(np.concatenate([np.zeros(6, dtype=float), tau_out]))

        contact_info = []
        fc_out = []
        for side in ["left", "right"]:
            if not contact_state[side]:
                continue
            cue = foot_cues[side]
            contact_info.append((cue["heel_body"], cue["heel_offset"], float(floor_height)))
            contact_info.append((cue["toe_body"], cue["toe_offset"], float(floor_height)))
            local_force = cue["foot_basis"] @ foot_forces[side]
            local_moment = cue["foot_basis"] @ foot_wrenches[side][3:6]
            heel_force = 0.5 * local_force
            toe_force = 0.5 * local_force
            if local_force[2] > 1e-6:
                cop_x = float(np.clip(-local_moment[1] / local_force[2], -cue["cop_half_length"], cue["cop_half_length"]))
                toe_share = np.clip(0.5 + cop_x / max(2.0 * cue["cop_half_length"], 1e-6), 0.0, 1.0)
                heel_share = 1.0 - toe_share
                heel_force = heel_share * local_force
                toe_force = toe_share * local_force
            fc_out.append(cue["foot_basis"].T @ heel_force)
            fc_out.append(cue["foot_basis"].T @ toe_force)
        if len(fc_out) == 0:
            fc_out = np.zeros(0, dtype=float)
        else:
            fc_out = np.concatenate(fc_out)

        contact_generalized = np.zeros(n_dof_full, dtype=float)
        for side in ["left", "right"]:
            contact_generalized += foot_cues[side]["anchor_jacobian"].T @ foot_forces[side]
            contact_generalized += foot_cues[side]["anchor_angular_jacobian"].T @ foot_wrenches[side][3:6]
        measurement_delta = x_step - np.array(model.getJointWorldPositions(measurement_joints), dtype=float).reshape(-1, 3)
        measurement_delta = measurement_delta[joint_weights > 0.0]
        dynamics_residual = M_full @ ddq_full + h_full - tau_full - contact_generalized

        state = {
            "q": q_full.copy(),
            "dq": dq_full.copy(),
            "ddq": ddq_full.copy(),
            "q_kin": q_kin.copy(),
            "dq_kin": dq_kin.copy(),
            "ddq_kin": ddq_kin.copy(),
            "tau": tau_out.copy(),
            "tau_full": tau_full.copy(),
            "root_residual": root_residual.copy(),
            "foot_forces": {
                "left": foot_forces["left"].copy(),
                "right": foot_forces["right"].copy(),
            },
            "foot_wrenches": {
                "left": foot_wrenches["left"].copy(),
                "right": foot_wrenches["right"].copy(),
            },
            "contact_state": {
                "left": bool(contact_state["left"]),
                "right": bool(contact_state["right"]),
            },
            "contact_prob": {
                "left": float(contact_prob["left"]),
                "right": float(contact_prob["right"]),
            },
            "floor_height": float(floor_height),
            "step_index": state["step_index"] + 1,
        }

        result = {
            "q": q_full,
            "dq": dq_full,
            "ddq": ddq_full,
            "q_dyn": q_dyn,
            "dq_dyn": dq_dyn,
            "q_hat": q_hat,
            "dq_hat": dq_hat,
            "ddq_hat": ddq_hat,
            "q_kin": q_kin,
            "dq_kin": dq_kin,
            "ddq_kin": ddq_kin,
            "measurement_weights": measurement_weights,
            "joint_confidence": joint_confidence,
            "dof_reliability": dof_reliability,
            "foot_cues": foot_cues,
            "floor_height": floor_height,
            "tau": tau_out,
            "tau_full": tau_full,
            "root_residual": root_residual,
            "foot_forces": foot_forces,
            "foot_wrenches": foot_wrenches,
            "contact_state": contact_state,
            "contact_prob": contact_prob,
            "contact_info": contact_info,
            "fc": fc_out,
            "delta": measurement_delta,
            "state": state,
            "dynamics_residual": dynamics_residual,
            "dynamics_residual_norm": float(np.linalg.norm(dynamics_residual)),
        }

    return result


def rmse(estimate, reference):
    error = np.array(estimate, dtype=float) - np.array(reference, dtype=float)
    return float(np.sqrt(np.mean(error * error)))


def mae(estimate, reference):
    error = np.array(estimate, dtype=float) - np.array(reference, dtype=float)
    return float(np.mean(np.abs(error)))


def per_column_rmse(estimate, reference, names):
    error = np.array(estimate, dtype=float) - np.array(reference, dtype=float)
    values = np.sqrt(np.mean(error * error, axis=0))
    return [(str(name), float(value)) for name, value in zip(names, values)]


def top_k_rmse(estimate, reference, names, k=8):
    ranked = per_column_rmse(estimate, reference, names)
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def binary_classification_metrics(prediction, reference):
    pred = np.array(prediction, dtype=bool).reshape(-1)
    ref = np.array(reference, dtype=bool).reshape(-1)
    tp = float(np.sum(pred & ref))
    tn = float(np.sum(~pred & ~ref))
    fp = float(np.sum(pred & ~ref))
    fn = float(np.sum(~pred & ref))
    precision = tp / max(tp + fp, EPS)
    recall = tp / max(tp + fn, EPS)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, EPS)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

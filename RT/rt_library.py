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
    contact_state=None,
    floor_height=np.nan,
):
    model = skeleton._nimble if hasattr(skeleton, "_nimble") else skeleton
    n_dof = model.getNumDofs()
    n_act = n_dof - 6
    state = {
        "q": np.array(model.getPositions(), dtype=float) if q is None else np.array(q, dtype=float).copy(),
        "dq": np.array(model.getVelocities(), dtype=float) if dq is None else np.array(dq, dtype=float).copy(),
        "ddq": np.zeros(n_dof, dtype=float) if ddq is None else np.array(ddq, dtype=float).copy(),
        "tau": np.zeros(n_act, dtype=float) if tau is None else np.array(tau, dtype=float).copy(),
        "tau_full": np.zeros(n_dof, dtype=float) if tau_full is None else np.array(tau_full, dtype=float).copy(),
        "root_residual": np.zeros(6, dtype=float) if root_residual is None else np.array(root_residual, dtype=float).copy(),
        "foot_forces": {
            "left": np.zeros(3, dtype=float),
            "right": np.zeros(3, dtype=float),
        },
        "contact_state": {
            "left": False,
            "right": False,
        },
        "floor_height": float(floor_height),
        "step_index": 0,
    }
    if foot_forces is not None:
        for side in state["foot_forces"]:
            if side in foot_forces:
                state["foot_forces"][side] = np.array(foot_forces[side], dtype=float).copy()
    if contact_state is not None:
        for side in state["contact_state"]:
            if side in contact_state:
                state["contact_state"][side] = bool(contact_state[side])
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
    if state is None:
        state = initialize_rt_state(
            skeleton,
            ddq=np.zeros(n_dof_full, dtype=float) if ddq_prev is None else ddq_prev,
            tau=np.zeros(n_act_full, dtype=float) if tau_prev is None else tau_prev,
        )
    else:
        state = {
            "q": np.array(state["q"], dtype=float).copy(),
            "dq": np.array(state["dq"], dtype=float).copy(),
            "ddq": np.array(state["ddq"], dtype=float).copy(),
            "tau": np.array(state["tau"], dtype=float).copy(),
            "tau_full": np.array(state["tau_full"], dtype=float).copy(),
            "root_residual": np.array(state["root_residual"], dtype=float).copy(),
            "foot_forces": {
                "left": np.array(state["foot_forces"]["left"], dtype=float).copy(),
                "right": np.array(state["foot_forces"]["right"], dtype=float).copy(),
            },
            "contact_state": {
                "left": bool(state["contact_state"]["left"]),
                "right": bool(state["contact_state"]["right"]),
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
                valid = measurement_weights.reshape(-1) > 0.0
                if not np.any(valid):
                    break
                if np.max(np.abs(np.where(measurement_weights > 0.0, residual, 0.0))) < 1e-4:
                    dof_reliability = np.sum((measurement_weights.reshape(-1)[valid].reshape(-1, 1) * J_full[valid, :]) ** 2, axis=0)
                    break

                valid_key = np.packbits(valid.astype(np.uint8)).tobytes()
                cache_key = (n_dof_full, valid_key)
                if cache_key not in QPIK_CACHE:
                    dq_var = cp.Variable(n_dof_full)
                    q_param = cp.Parameter(n_dof_full)
                    qik_A_meas_param = cp.Parameter((int(np.sum(valid)), n_dof_full))
                    qik_b_meas_param = cp.Parameter(int(np.sum(valid)))
                    dq_accum_param = cp.Parameter(n_dof_full)
                    dq_l_param = cp.Parameter(n_dof_full)
                    dq_u_param = cp.Parameter(n_dof_full)
                    q_l_param = cp.Parameter(n_dof_full)
                    q_u_param = cp.Parameter(n_dof_full)
                    qik_A_step_param = cp.Parameter((n_dof_full, n_dof_full))
                    qik_A_prior_param = cp.Parameter((n_dof_full, n_dof_full))
                    qik_b_prior_param = cp.Parameter(n_dof_full)
                    qik_A_sensor_param = cp.Parameter((n_dof_full, n_dof_full))
                    qik_b_sensor_param = cp.Parameter(n_dof_full)
                    objective = cp.sum_squares(qik_A_meas_param @ dq_var - qik_b_meas_param)
                    objective += cp.sum_squares(qik_A_step_param @ dq_var)
                    objective += cp.sum_squares(qik_A_prior_param @ dq_var - qik_b_prior_param)
                    objective += cp.sum_squares(qik_A_sensor_param @ dq_var - qik_b_sensor_param)
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
                        "dq_accum": dq_accum_param,
                        "dq_l": dq_l_param,
                        "dq_u": dq_u_param,
                        "q_l": q_l_param,
                        "q_u": q_u_param,
                        "A_step": qik_A_step_param,
                        "A_prior": qik_A_prior_param,
                        "b_prior": qik_b_prior_param,
                        "A_sensor": qik_A_sensor_param,
                        "b_sensor": qik_b_sensor_param,
                    }

                problem = QPIK_CACHE[cache_key]
                problem["q"].value = q_hat
                meas_w = 30.0 * np.maximum(measurement_weights.reshape(-1)[valid], 1e-6)
                problem["A_meas"].value = np.diag(meas_w) @ (J_full[valid, :] + PARAM_DENSE_EPS)
                problem["b_meas"].value = meas_w * (x_step.reshape(-1)[valid] - x_current.reshape(-1)[valid])
                problem["dq_accum"].value = dq_cumulative
                problem["dq_l"].value = -dq_step_cap
                problem["dq_u"].value = dq_step_cap
                problem["q_l"].value = q_lower
                problem["q_u"].value = q_upper
                problem["A_step"].value = np.diag(step_w)
                problem["A_prior"].value = np.diag(prior_w)
                problem["b_prior"].value = prior_w * (q_pred - q_hat)
                if q_meas is not None and q_weights is not None:
                    q_sensor = np.where(np.isfinite(q_meas), q_meas, q_pred)
                    q_sensor_w = 1.5 * np.where(np.isfinite(q_meas), q_weights, 0.0)
                else:
                    q_sensor = q_pred
                    q_sensor_w = np.zeros(n_dof_full, dtype=float)
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

        dq_hat = dof_reliability * ((q_hat - q_prev) / max(dt_sim, 1e-3)) + (1.0 - dof_reliability) * dq_pred
        if dq_meas is not None and dq_weights is not None:
            dq_sensor = np.where(np.isfinite(dq_meas), dq_meas, dq_hat)
            blend = np.clip(dq_weights / np.maximum(dq_weights + 1.0, EPS), 0.0, 1.0)
            dq_hat = blend * dq_sensor + (1.0 - blend) * dq_hat
        ddq_hat = (dq_hat - dq_prev) / max(dt_sim, 1e-3)

        model.setPositions(q_hat)
        model.setVelocities(dq_hat)

        gravity = np.array(model.getGravity(), dtype=float).reshape(3)
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
            heel_velocity = heel_jacobian @ dq_hat
            toe_velocity = toe_jacobian @ dq_hat
            heel_bias = np.array(model.getLinearJacobianDeriv(heel_body, heel_offset), dtype=float) @ dq_hat
            toe_bias = np.array(model.getLinearJacobianDeriv(toe_body, toe_offset), dtype=float) @ dq_hat

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
            ankle_name = "ankle_l" if side == "left" else "ankle_r"
            cue["joint_confidence"] = joint_conf_lookup.get(ankle_name, float(np.mean(joint_confidence)) if joint_confidence.size > 0 else 0.0)
            cue["confidence"] = float(
                np.clip(
                    0.5 * cue["joint_confidence"]
                    + 0.25 * np.clip(1.0 - max(height, 0.0) / 0.08, 0.0, 1.0)
                    + 0.25 * np.clip(1.0 - abs(vertical_velocity) / 0.8, 0.0, 1.0),
                    0.0,
                    1.0,
                )
            )
            if state["contact_state"][side]:
                cue["contact"] = bool(height <= 0.11 and abs(vertical_velocity) <= 1.2)
            else:
                cue["contact"] = bool(height <= 0.08 and abs(vertical_velocity) <= 0.8)

        model.setPositions(q_hat)
        model.setVelocities(dq_hat)

        M_full = np.array(model.getMassMatrix(), dtype=float)
        h_full = np.array(model.getCoriolisAndGravityForces(), dtype=float).reshape(-1)
        M = M_full[np.ix_(active_dofs, active_dofs)]
        h = h_full[active_dofs]

        n_dof = len(active_dofs)
        n_act = len(active_act_dofs)
        q_target = q_hat[active_dofs]
        dq_target = np.clip(dq_hat[active_dofs], -dq_cap[active_dofs], dq_cap[active_dofs])
        ddq_target = ddq_hat[active_dofs]
        rel = dof_reliability[active_dofs]

        w_q = 35.0 * (0.2 + rel)
        w_dq = 4.0 * (0.2 + rel)
        w_ddq = 0.30 * (0.1 + rel)
        if n_dof >= 6:
            w_q[:6] *= 4.0
            w_dq[:6] *= 2.0
            w_ddq[:6] *= 1.5

        contact_state = {
            "left": bool(foot_cues["left"]["contact"]),
            "right": bool(foot_cues["right"]["contact"]),
        }

        J_force = np.zeros((n_dof, 6), dtype=float)
        for foot_idx, side in enumerate(["left", "right"]):
            J_side = foot_cues[side]["anchor_jacobian"][:, active_dofs]
            J_force[:, 3 * foot_idx : 3 * foot_idx + 3] = J_side.T

        S_T = np.zeros((n_dof, n_act), dtype=float)
        for i, dof in enumerate(active_act_dofs):
            S_T[np.where(active_dofs == dof)[0][0], i] = 1.0

        S_root = np.zeros((n_dof, 6), dtype=float)
        for i, dof in enumerate(active_dofs):
            if dof < 6:
                S_root[i, dof] = 1.0

        q_margin = 1e-3
        ddq_pos_upper = 2.0 * (q_upper[active_dofs] - q_margin - q_prev[active_dofs] - dt_sim * dq_prev[active_dofs]) / max(dt_sim * dt_sim, EPS)
        ddq_pos_lower = 2.0 * (q_lower[active_dofs] + q_margin - q_prev[active_dofs] - dt_sim * dq_prev[active_dofs]) / max(dt_sim * dt_sim, EPS)
        ddq_vel_upper = (dq_cap[active_dofs] - dq_prev[active_dofs]) / max(dt_sim, EPS)
        ddq_vel_lower = (-dq_cap[active_dofs] - dq_prev[active_dofs]) / max(dt_sim, EPS)
        ddq_abs = np.ones(n_dof, dtype=float) * 120.0
        if n_dof >= 6:
            ddq_abs[:3] = 80.0
            ddq_abs[3:6] = 60.0
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
                q_prev[active_dofs] + dt_sim * dq_prev[active_dofs] - q_hat[active_dofs]
            )
            meas_w = 40.0 * np.maximum(measurement_weights.reshape(-1)[valid], 1e-6)

        valid_key = np.packbits(valid.astype(np.uint8)).tobytes()
        basis_key = np.round(ground_basis.reshape(-1), 8).tobytes()
        dyn_key = (n_dof, n_act, valid_key, contact_state["left"], contact_state["right"], round(float(mu), 6), basis_key)
        if dyn_key not in DYN_QP_CACHE:
            ddq_var = cp.Variable(n_dof)
            tau_var = cp.Variable(n_act)
            force_var = cp.Variable(6)
            root_var = cp.Variable(6)
            M_param = cp.Parameter((n_dof, n_dof))
            h_param = cp.Parameter(n_dof)
            J_force_param = cp.Parameter((n_dof, 6))
            ddq_lower_param = cp.Parameter(n_dof)
            ddq_upper_param = cp.Parameter(n_dof)
            A_q_param = cp.Parameter((n_dof, n_dof))
            b_q_param = cp.Parameter(n_dof)
            A_dq_param = cp.Parameter((n_dof, n_dof))
            b_dq_param = cp.Parameter(n_dof)
            A_ddq_param = cp.Parameter((n_dof, n_dof))
            b_ddq_param = cp.Parameter(n_dof)
            A_tau_s_param = cp.Parameter((n_act, n_act))
            b_tau_s_param = cp.Parameter(n_act)
            A_tau_r_param = cp.Parameter((n_act, n_act))
            A_force_s_param = cp.Parameter((6, 6))
            b_force_s_param = cp.Parameter(6)
            A_force_r_param = cp.Parameter((6, 6))
            A_root_param = cp.Parameter((6, 6))
            objective = cp.sum_squares(A_q_param @ ddq_var - b_q_param)
            objective += cp.sum_squares(A_dq_param @ ddq_var - b_dq_param)
            objective += cp.sum_squares(A_ddq_param @ ddq_var - b_ddq_param)
            objective += cp.sum_squares(A_tau_s_param @ tau_var - b_tau_s_param)
            objective += cp.sum_squares(A_tau_r_param @ tau_var)
            objective += cp.sum_squares(A_force_s_param @ force_var - b_force_s_param)
            objective += cp.sum_squares(A_force_r_param @ force_var)
            objective += cp.sum_squares(A_root_param @ root_var)
            constraints = [
                M_param @ ddq_var + h_param == S_T @ tau_var + J_force_param @ force_var + S_root @ root_var,
                ddq_var >= ddq_lower_param,
                ddq_var <= ddq_upper_param,
            ]

            if int(np.sum(valid)) > 0:
                J_meas_param = cp.Parameter((int(np.sum(valid)), n_dof))
                meas_b_param = cp.Parameter(int(np.sum(valid)))
                objective += cp.sum_squares(J_meas_param @ ddq_var - meas_b_param)
            else:
                J_meas_param = None
                meas_b_param = None

            side_params = {}
            for side, force_slice in [("left", slice(0, 3)), ("right", slice(3, 6))]:
                if dyn_key[3] if side == "left" else dyn_key[4]:
                    A_acc_param = cp.Parameter((3, n_dof))
                    b_acc_param = cp.Parameter(3)
                    A_vel_param = cp.Parameter((3, n_dof))
                    b_vel_param = cp.Parameter(3)
                    objective += cp.sum_squares(A_acc_param @ ddq_var - b_acc_param)
                    objective += cp.sum_squares(A_vel_param @ ddq_var - b_vel_param)
                    local_force = ground_basis @ force_var[force_slice]
                    constraints += [
                        local_force[2] >= 0.0,
                        local_force[0] <= float(mu) * local_force[2],
                        -local_force[0] <= float(mu) * local_force[2],
                        local_force[1] <= float(mu) * local_force[2],
                        -local_force[1] <= float(mu) * local_force[2],
                    ]
                    side_params[side] = {
                        "A_acc": A_acc_param,
                        "b_acc": b_acc_param,
                        "A_vel": A_vel_param,
                        "b_vel": b_vel_param,
                    }
                else:
                    constraints += [force_var[force_slice] == 0.0]
                    side_params[side] = None

            DYN_QP_CACHE[dyn_key] = {
                "problem": cp.Problem(cp.Minimize(objective), constraints),
                "ddq": ddq_var,
                "tau": tau_var,
                "force": force_var,
                "root": root_var,
                "M": M_param,
                "h": h_param,
                "J_force": J_force_param,
                "ddq_lower": ddq_lower_param,
                "ddq_upper": ddq_upper_param,
                "A_q": A_q_param,
                "b_q": b_q_param,
                "A_dq": A_dq_param,
                "b_dq": b_dq_param,
                "A_ddq": A_ddq_param,
                "b_ddq": b_ddq_param,
                "A_tau_s": A_tau_s_param,
                "b_tau_s": b_tau_s_param,
                "A_tau_r": A_tau_r_param,
                "A_force_s": A_force_s_param,
                "b_force_s": b_force_s_param,
                "A_force_r": A_force_r_param,
                "A_root": A_root_param,
                "J_meas": J_meas_param,
                "meas_b": meas_b_param,
                "side_params": side_params,
            }

        problem = DYN_QP_CACHE[dyn_key]
        problem["M"].value = M + PARAM_DENSE_EPS
        problem["h"].value = h
        problem["J_force"].value = J_force + PARAM_DENSE_EPS
        problem["ddq_lower"].value = ddq_lower
        problem["ddq_upper"].value = ddq_upper
        problem["A_q"].value = np.diag(w_q * (0.5 * dt_sim * dt_sim))
        problem["b_q"].value = w_q * (q_target - q_prev[active_dofs] - dt_sim * dq_prev[active_dofs])
        problem["A_dq"].value = np.diag(w_dq * dt_sim)
        problem["b_dq"].value = w_dq * (dq_target - dq_prev[active_dofs])
        problem["A_ddq"].value = np.diag(w_ddq)
        problem["b_ddq"].value = w_ddq * ddq_target
        if n_act > 0:
            tau_prev_active = state["tau"][active_act_dofs - 6]
            tau_smooth_w = np.ones(n_act, dtype=float) * 0.05
            tau_reg_w = np.ones(n_act, dtype=float) * 0.004
            problem["A_tau_s"].value = np.diag(tau_smooth_w)
            problem["b_tau_s"].value = tau_smooth_w * tau_prev_active
            problem["A_tau_r"].value = np.diag(tau_reg_w)
        else:
            problem["A_tau_s"].value = np.zeros((0, 0), dtype=float)
            problem["b_tau_s"].value = np.zeros(0, dtype=float)
            problem["A_tau_r"].value = np.zeros((0, 0), dtype=float)
        force_prev = np.concatenate([state["foot_forces"]["left"], state["foot_forces"]["right"]])
        force_smooth_w = np.ones(6, dtype=float) * 0.06
        force_reg_w = np.ones(6, dtype=float) * 0.004
        root_w = np.array([0.40, 0.40, 0.40, 0.20, 0.20, 0.20], dtype=float)
        problem["A_force_s"].value = np.diag(force_smooth_w)
        problem["b_force_s"].value = force_smooth_w * force_prev
        problem["A_force_r"].value = np.diag(force_reg_w)
        problem["A_root"].value = np.diag(root_w)
        if problem["J_meas"] is not None:
            problem["J_meas"].value = np.diag(meas_w) @ (0.5 * dt_sim * dt_sim * (J_meas + PARAM_DENSE_EPS))
            problem["meas_b"].value = meas_w * meas_b

        for side, force_slice in [("left", slice(0, 3)), ("right", slice(3, 6))]:
            side_params = problem["side_params"][side]
            if side_params is None:
                continue
            J_side = foot_cues[side]["anchor_jacobian"][:, active_dofs]
            w_acc = np.array([0.10, 0.10, 0.30], dtype=float) * (0.3 + foot_cues[side]["confidence"])
            w_vel = np.array([0.03, 0.03, 0.08], dtype=float) * (0.3 + foot_cues[side]["confidence"])
            side_params["A_acc"].value = np.diag(w_acc) @ (J_side + PARAM_DENSE_EPS)
            side_params["b_acc"].value = -w_acc * foot_cues[side]["anchor_bias"]
            side_params["A_vel"].value = np.diag(w_vel) @ (dt_sim * (J_side + PARAM_DENSE_EPS))
            side_params["b_vel"].value = -w_vel * (J_side @ dq_prev)

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
        tau_active = None if problem["tau"].value is None else np.array(problem["tau"].value, dtype=float).reshape(-1)
        force_proj = None if problem["force"].value is None else np.array(problem["force"].value, dtype=float).reshape(-1)
        root_residual = None if problem["root"].value is None else np.array(problem["root"].value, dtype=float).reshape(-1)
        if ddq_active is None or tau_active is None or force_proj is None or root_residual is None:
            return None

        ddq_full = np.zeros(n_dof_full, dtype=float)
        ddq_full[active_dofs] = ddq_active
        dq_dyn = dq_prev + dt_sim * ddq_full
        q_dyn = q_prev + dt_sim * dq_prev + 0.5 * dt_sim * dt_sim * ddq_full
        q_dyn = np.clip(q_dyn, q_lower, q_upper)
        q_full = q_hat.copy()
        dq_full = dq_hat.copy()

        tau_out = np.zeros(n_act_full, dtype=float)
        tau_full = np.zeros(n_dof_full, dtype=float)
        tau_full[:6] = root_residual
        for i, dof_idx in enumerate(active_act_dofs):
            tau_out[dof_idx - 6] = tau_active[i]
            tau_full[dof_idx] = tau_active[i]

        foot_forces = {
            "left": force_proj[:3].copy(),
            "right": force_proj[3:6].copy(),
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
            point_force = 0.5 * foot_forces[side]
            contact_info.append((cue["heel_body"], cue["heel_offset"], float(floor_height)))
            contact_info.append((cue["toe_body"], cue["toe_offset"], float(floor_height)))
            fc_out.append(point_force)
            fc_out.append(point_force)
        if len(fc_out) == 0:
            fc_out = np.zeros(0, dtype=float)
        else:
            fc_out = np.concatenate(fc_out)

        contact_generalized = np.zeros(n_dof_full, dtype=float)
        for side in ["left", "right"]:
            contact_generalized += foot_cues[side]["anchor_jacobian"].T @ foot_forces[side]
        measurement_delta = x_step - np.array(model.getJointWorldPositions(measurement_joints), dtype=float).reshape(-1, 3)
        measurement_delta = measurement_delta[joint_weights > 0.0]
        dynamics_residual = M_full @ ddq_full + h_full - tau_full - contact_generalized

        state = {
            "q": q_full.copy(),
            "dq": dq_full.copy(),
            "ddq": ddq_full.copy(),
            "tau": tau_out.copy(),
            "tau_full": tau_full.copy(),
            "root_residual": root_residual.copy(),
            "foot_forces": {
                "left": foot_forces["left"].copy(),
                "right": foot_forces["right"].copy(),
            },
            "contact_state": {
                "left": bool(contact_state["left"]),
                "right": bool(contact_state["right"]),
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
            "measurement_weights": measurement_weights,
            "joint_confidence": joint_confidence,
            "dof_reliability": dof_reliability,
            "foot_cues": foot_cues,
            "floor_height": floor_height,
            "tau": tau_out,
            "tau_full": tau_full,
            "root_residual": root_residual,
            "foot_forces": foot_forces,
            "contact_state": contact_state,
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

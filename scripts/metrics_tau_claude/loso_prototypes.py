#!/usr/bin/env python3
"""Fit direct-GRF prototypes in LOSO and prepare selected full-ID inputs.

All predictors use pipeline estimates and kinematics at inference. Measured
forces/CoP are read only as training labels or evaluation targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diagnose_modes import (
    SUBJECTS,
    TARGET_DIRNAME,
    TIME_TOLERANCE_S,
    TRANSITION_WINDOW_S,
    align_indices,
    map_source_path,
    read_cache,
    read_contact_wrenches,
    read_storage,
    stance_progress,
)
from opensim_batch_dynamics.inverse_dynamics_no_grf import (
    _horizontal_symmetric_targets,
    _phase_cop_progress,
)
from groundlink import (
    DEFAULT_SOURCE_FORCE_ROOT,
    DEFAULT_SOURCE_TORQUE_ROOT,
    EvaluationConfig,
    _force_path,
    _load_force_arrays,
    _lowpass_force,
)
sys.path.insert(0, str(Path(__file__).with_name("refid")))
from prepare_hybrid_ids import make_external_xml, read_mot, write_mot


COMPONENTS = {
    "BASELINE": (),
    "A": ("A",),
    "H": ("H",),
    "Hsym": ("Hsym",),
    "Hsym+P": ("Hsym", "P"),
    "S": ("S",),
    "P": ("P",),
    "Pphase": ("Pphase",),
    "Hsym+Pphase": ("Hsym", "Pphase"),
    "Pmix": ("Pmix",),
    "Hsym+Pmix": ("Hsym", "Pmix"),
    "H+P": ("H", "P"),
    "H+S+P": ("H", "S", "P"),
    "A+S": ("A", "S"),
    "A+P": ("A", "P"),
    "S+P": ("S", "P"),
    "A+S+P": ("A", "S", "P"),
}
HEIGHT_GRID_M = (0.01, 0.02, 0.03, 0.04, 0.05)
SPEED_GRID_MPS = (0.1, 0.2, 0.3, 0.5, 0.7)
HYSTERESIS_GRID_M = (0.0, 0.01)
RIDGE_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
S_CONTACT_GRID = (0.5, 0.65, 0.8, 0.9, 0.95)
PHASES = ("total", "single_stance", "double_stance", "flight", "transition", "other")
BODY_NAMES = ("calcn_l", "toes_l", "calcn_r", "toes_r")
BODY_INDEX = {name: index for index, name in enumerate(BODY_NAMES)}
GRAVITY = 9.81
FALSE_CONTACT_N = 20.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def classify_phases(times: np.ndarray, measured_force: np.ndarray, measured_raw: np.ndarray, bw_n: float) -> tuple[np.ndarray, np.ndarray]:
    contact = measured_force[:, :, 2] > 0.05 * bw_n
    flight = np.all(measured_raw[:, :, 2] <= 1.0, axis=1)
    phase = np.full(len(times), "other", dtype=object)
    count = contact.sum(axis=1)
    phase[count == 1] = "single_stance"
    phase[count == 2] = "double_stance"
    phase[flight] = "flight"
    transition = np.zeros(len(times), dtype=bool)
    for side in range(2):
        states = contact[:, side]
        edges = np.flatnonzero(states[1:] != states[:-1]) + 1
        if states.size and states[0]:
            edges = np.r_[0, edges]
        for edge in edges:
            transition |= np.abs(times - times[edge]) <= TRANSITION_WINDOW_S
    phase[transition] = "transition"
    return phase, contact


def fit_ridge(features: list[np.ndarray], targets: list[np.ndarray], ridge: float) -> np.ndarray:
    x = np.concatenate(features, axis=0)
    y = np.concatenate(targets, axis=0)
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[finite], y[finite]
    if len(y) < x.shape[1] + 2:
        raise ValueError(f"Too few finite samples to fit model: n={len(y)}, p={x.shape[1]}")
    design = np.column_stack([np.ones(len(x)), x])
    if ridge <= 0.0:
        return np.linalg.lstsq(design, y, rcond=None)[0]
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def predict_linear(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(features)), features]) @ coefficients


def hysteretic_contact(height: np.ndarray, speed: np.ndarray, height_cut: float, speed_cut: float, margin: float) -> np.ndarray:
    onset = (height <= height_cut) & (speed <= speed_cut)
    hold = (height <= height_cut + margin) & (speed <= speed_cut)
    index = np.arange(len(height))
    last_on = np.maximum.accumulate(np.where(onset, index, -1))
    last_break = np.maximum.accumulate(np.where(~hold, index, -1))
    return last_on > last_break


def estimated_point_contacts(
    data: dict[str, Any], a_params: dict[str, Any] | None = None, activation_threshold: float = 0.5
) -> np.ndarray:
    if a_params is None:
        return data["activation"].reshape(len(data["times"]), 4) > activation_threshold
    point_contact = np.zeros((len(data["times"]), 4), dtype=bool)
    for body_index in range(4):
        point_contact[:, body_index] = hysteretic_contact(
            data["body_height"][:, body_index], data["body_speed"][:, body_index],
            float(a_params["height_threshold_m"]), float(a_params["speed_threshold_mps"]),
            float(a_params["hysteresis_margin_m"]),
        )
    return point_contact


def estimated_contacts(
    data: dict[str, Any], a_params: dict[str, Any] | None = None, activation_threshold: float = 0.5
) -> np.ndarray:
    point_contact = estimated_point_contacts(data, a_params, activation_threshold)
    return np.column_stack([point_contact[:, :2].any(axis=1), point_contact[:, 2:].any(axis=1)])


def elapsed_since_state_change(times: np.ndarray, contact: np.ndarray) -> np.ndarray:
    edges = np.r_[0, np.flatnonzero(contact[1:] != contact[:-1]) + 1]
    latest = edges[np.searchsorted(edges, np.arange(len(times)), side="right") - 1]
    return times - times[latest]


def split_features(data: dict[str, Any], contact: np.ndarray) -> np.ndarray:
    left = data["body_positions"][:, BODY_INDEX["calcn_l"], :2]
    right = data["body_positions"][:, BODY_INDEX["calcn_r"], :2]
    com = data["com"][:, :2]
    line = right - left
    length2 = np.sum(line * line, axis=1)
    lam = np.divide(np.sum((com - left) * line, axis=1), length2,
                    out=np.full(len(com), 0.5), where=length2 > 1e-8)
    elapsed = np.column_stack([
        elapsed_since_state_change(data["times"], contact[:, side]) for side in range(2)
    ])
    return np.column_stack([lam, elapsed])


def cop_features(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    heel = data["body_positions"][:, BODY_INDEX["calcn_l"]]
    toes = data["body_positions"][:, BODY_INDEX["toes_l"]]
    axis = toes[:, :2] - heel[:, :2]
    length = np.linalg.norm(axis, axis=1)
    incline_l = np.arctan2(toes[:, 2] - heel[:, 2], np.maximum(length, 1e-4))
    heel_r = data["body_positions"][:, BODY_INDEX["calcn_r"]]
    toes_r = data["body_positions"][:, BODY_INDEX["toes_r"]]
    axis_r = toes_r[:, :2] - heel_r[:, :2]
    length_r = np.linalg.norm(axis_r, axis=1)
    incline_r = np.arctan2(toes_r[:, 2] - heel_r[:, 2], np.maximum(length_r, 1e-4))
    progress = np.column_stack([
        np.nan_to_num(stance_progress(data["times"], data["base_contact"][:, side]), nan=0.0)
        for side in range(2)
    ])
    activation = data["activation"]
    result = []
    for side in range(2):
        result.append(np.column_stack([progress[:, side], (incline_l, incline_r)[side],
                                       activation[:, side, 0], activation[:, side, 1]]))
    return np.stack(result, axis=1), np.stack([incline_l, incline_r], axis=1)


def cop_phase_features(data: dict[str, Any]) -> np.ndarray:
    """CoP features aligned to the production phase-model schema."""
    contact = estimated_contacts(data)
    progress = np.column_stack([
        np.nan_to_num(stance_progress(data["times"], contact[:, side]), nan=0.0)
        for side in range(2)
    ])
    cop, _ = cop_features(data)
    incline = cop[:, :, 1]
    active = contact.astype(float)
    return np.stack([progress, incline, active, active], axis=2)


def horizontal_features(data: dict[str, Any]) -> np.ndarray:
    if "_horizontal_features" in data:
        return data["_horizontal_features"]
    times, body, com = data["times"], data["body_positions"], data["com"]
    feet = body[:, [BODY_INDEX["calcn_l"], BODY_INDEX["calcn_r"]]]
    height = np.maximum(com[:, 2] - data["floor_height"], 0.1)
    com_velocity = np.gradient(com[:, :2], times, axis=0)
    foot_velocity = np.gradient(feet[:, :, :2], times, axis=0)
    relative = (feet[:, :, :2] - com[:, None, :2]) / height[:, None, None]
    relative_velocity = (foot_velocity - com_velocity[:, None, :]) / height[:, None, None]
    force_z = np.stack([data["body_force"][:, :2, 2].sum(axis=1),
                        data["body_force"][:, 2:, 2].sum(axis=1)], axis=1) / data["bw_n"]
    contact = estimated_contacts(data).astype(float)
    progress = np.column_stack([
        np.nan_to_num(stance_progress(times, data["base_contact"][:, side]), nan=0.0)
        for side in range(2)
    ])
    data["_horizontal_features"] = np.column_stack([
        progress, contact, force_z, relative.reshape(len(times), -1),
        relative_velocity.reshape(len(times), -1), data["net_force"][:, :2] / data["bw_n"]
    ])
    return data["_horizontal_features"]


def horizontal_local_features(data: dict[str, Any]) -> np.ndarray:
    if "_horizontal_local_features" in data:
        return data["_horizontal_local_features"]
    times, body, com = data["times"], data["body_positions"], data["com"]
    feet = body[:, [BODY_INDEX["calcn_l"], BODY_INDEX["calcn_r"]]]
    height = np.maximum(com[:, 2] - data["floor_height"], 0.1)
    com_velocity = np.gradient(com[:, :2], times, axis=0)
    foot_velocity = np.gradient(feet[:, :, :2], times, axis=0)
    relative = (feet[:, :, :2] - com[:, None, :2]) / height[:, None, None]
    relative_velocity = (foot_velocity - com_velocity[:, None, :]) / height[:, None, None]
    # Reflect the right foot's mediolateral coordinate into the left-foot frame.
    mirror = np.array([1.0, -1.0])[None, :, None]
    relative *= mirror
    relative_velocity *= mirror
    vertical = np.stack([
        data["body_force"][:, :2, 2].sum(axis=1),
        data["body_force"][:, 2:, 2].sum(axis=1),
    ], axis=1) / data["bw_n"]
    contact = estimated_contacts(data).astype(float)
    progress = np.column_stack([
        np.nan_to_num(stance_progress(times, data["base_contact"][:, side]), nan=0.0)
        for side in range(2)
    ])
    net = np.broadcast_to(data["net_force"][:, None, :2] / data["bw_n"], (len(times), 2, 2)).copy()
    net[:, 1, 1] *= -1.0
    data["_horizontal_local_features"] = np.concatenate([
        progress[:, :, None], contact[:, :, None], vertical[:, :, None],
        relative, relative_velocity, net,
    ], axis=2)
    return data["_horizontal_local_features"]


def fit_horizontal_symmetric_model(
    segments: list[dict[str, Any]], ridge: float, residual_allocation: str,
) -> dict[str, Any]:
    features, targets = [], []
    for data in segments:
        mask = data["valid_target"] & data["force_valid"] & data["true_contact"].all(axis=1)
        local = horizontal_local_features(data)
        for side in range(2):
            features.append(local[mask, side])
            target = data["measured_force"][mask, side, :2] / data["bw_n"]
            if side == 1:
                target = target.copy()
                target[:, 1] *= -1.0
            targets.append(target)
    x, y = np.concatenate(features), np.concatenate(targets)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y = x[finite], y[finite]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = (np.linalg.lstsq(design, y, rcond=None)[0] if ridge <= 0
                    else np.linalg.solve(design.T @ design + penalty, design.T @ y))
    return {
        "architecture": "shared_per_foot_mirrored_y", "residual_allocation": residual_allocation,
        "mean": mean.tolist(), "scale": scale.tolist(), "coefficients": coefficients.tolist(),
    }


def fit_horizontal_model(segments: list[dict[str, Any]], ridge: float) -> dict[str, Any]:
    features, targets = [], []
    for data in segments:
        mask = data["valid_target"] & data["force_valid"] & data["true_contact"].all(axis=1)
        features.append(horizontal_features(data)[mask])
        targets.append(data["measured_force"][mask, 0, :2] / data["bw_n"])
    x, y = np.concatenate(features), np.concatenate(targets)
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x, y = x[finite], y[finite]
    if len(y) < x.shape[1] + 2:
        raise ValueError(f"Too few double-stance samples to fit horizontal model: n={len(y)}")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = (np.linalg.lstsq(design, y, rcond=None)[0] if ridge <= 0
                    else np.linalg.solve(design.T @ design + penalty, design.T @ y))
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coefficients": coefficients.tolist()}


def model_features(data: dict[str, Any], component: str, a_params: dict[str, Any] | None = None) -> np.ndarray:
    if component == "H":
        return horizontal_features(data)
    if component == "S":
        return split_features(data, estimated_contacts(data, a_params))
    if component == "P":
        return data["cop_features"].reshape(-1, data["cop_features"].shape[-1])
    if component == "Pphase":
        return cop_phase_features(data).reshape(-1, 4)
    raise ValueError(component)


def fit_split_model(
    segments: list[dict[str, Any]], ridge: float, a_params: dict[str, Any] | None = None,
    activation_threshold: float = 0.5,
) -> np.ndarray:
    features, targets = [], []
    for data in segments:
        measured_contact = data["true_contact"]
        double = measured_contact[:, 0] & measured_contact[:, 1]
        mask = data["valid_target"] & data["force_valid"] & double
        total = data["measured_force"][:, :, 2].sum(axis=1)
        share = np.divide(data["measured_force"][:, 0, 2], total,
                          out=np.full(len(total), np.nan), where=total > 1e-6)
        features.append(split_features(data, estimated_contacts(data, a_params, activation_threshold))[mask])
        targets.append(share[mask])
    return fit_ridge(features, targets, ridge)


def fit_cop_model(
    segments: list[dict[str, Any]], ridge: float, stance_phase: str | None = None,
    phase_features: bool = False,
) -> np.ndarray:
    features, targets = [], []
    use_phase_features = phase_features or stance_phase is not None
    if stance_phase not in (None, "single_stance", "double_stance"):
        raise ValueError(f"Unsupported CoP training phase: {stance_phase}")
    for data in segments:
        contact_count = data["true_contact"].sum(axis=1)
        for side in range(2):
            heel = data["body_positions"][:, BODY_INDEX[("calcn_l", "calcn_r")[side]], :2]
            toes = data["body_positions"][:, BODY_INDEX[("toes_l", "toes_r")[side]], :2]
            axis = toes - heel
            length2 = np.sum(axis * axis, axis=1)
            delta = data["measured_cop"][:, side] - heel
            progress = np.divide(np.sum(delta * axis, axis=1), length2,
                                 out=np.full(len(axis), np.nan), where=length2 > 1e-8)
            mask = (data["valid_target"] & data["force_valid"] & ~data["cop_invalid"]
                    & data["true_contact"][:, side] & np.isfinite(progress))
            if stance_phase is not None:
                required_contacts = 1 if stance_phase == "single_stance" else 2
                mask &= contact_count == required_contacts
            feature_array = cop_phase_features(data) if use_phase_features else data["cop_features"]
            features.append(feature_array[:, side, :][mask])
            targets.append(progress[mask])
    return fit_ridge(features, targets, ridge)


def body_shares(body_force: np.ndarray, side: int) -> np.ndarray:
    start = 0 if side == 0 else 2
    fz = np.maximum(body_force[:, start:start + 2, 2], 0.0)
    total = fz.sum(axis=1, keepdims=True)
    magnitudes = np.linalg.norm(body_force[:, start:start + 2], axis=2)
    fallback = magnitudes / np.maximum(magnitudes.sum(axis=1, keepdims=True), 1e-9)
    return np.divide(fz, total, out=fallback, where=total > 1e-9)


def components_for(variant: str) -> tuple[str, ...]:
    if variant not in COMPONENTS:
        raise ValueError(f"Unknown prototype variant: {variant}")
    return COMPONENTS[variant]


def apply_variant(data: dict[str, Any], variant: str, params: dict[str, Any]) -> dict[str, np.ndarray]:
    components = components_for(variant)
    body_force = data["body_force"].copy()
    body_point = data["body_point"].copy()
    contact = estimated_contacts(data, params.get("A") if "A" in components else None)

    if "A" in components:
        original = body_force.copy()
        total = original.sum(axis=1)
        body_force[:] = 0.0
        counts = contact.sum(axis=1)
        point_contact = estimated_point_contacts(data, params["A"])
        for side in range(2):
            active_side = contact[:, side]
            if not active_side.any():
                continue
            indices = slice(0, 2) if side == 0 else slice(2, 4)
            side_total = original[:, indices].sum(axis=1)
            desired = np.where((counts == 1)[:, None], total, side_total)
            point_mask = point_contact[:, indices].astype(float)
            shares = body_shares(original, side) * point_mask
            share_sum = shares.sum(axis=1, keepdims=True)
            fallback = point_mask / np.maximum(point_mask.sum(axis=1, keepdims=True), 1.0)
            shares = np.divide(shares, share_sum, out=fallback, where=share_sum > 1e-9)
            body_force[active_side, indices] = desired[active_side, None, :] * shares[active_side, :, None]

    if "H" in components:
        model = params["H_model"]
        features = horizontal_features(data)
        design = np.column_stack([np.ones(len(features)),
                                  (features - np.asarray(model["mean"])) / np.asarray(model["scale"])])
        left_force = design @ np.asarray(model["coefficients"]) * data["bw_n"]
        force_contact = np.column_stack([
            np.any(body_force[:, :2, :] != 0.0, axis=(1, 2)),
            np.any(body_force[:, 2:, :] != 0.0, axis=(1, 2)),
        ])
        double = force_contact.all(axis=1)
        targets = np.zeros((len(contact), 2, 2), dtype=float)
        targets[double, 0] = left_force[double]
        targets[double, 1] = data["net_force"][double, :2] - left_force[double]
        single_left = force_contact[:, 0] & ~force_contact[:, 1]
        single_right = force_contact[:, 1] & ~force_contact[:, 0]
        targets[single_left, 0] = data["net_force"][single_left, :2]
        targets[single_right, 1] = data["net_force"][single_right, :2]
        for side in range(2):
            start = 0 if side == 0 else 2
            shares = body_shares(body_force, side)
            for axis in range(2):
                body_force[:, start:start + 2, axis] = targets[:, side, axis, None] * shares

    if "Hsym" in components:
        model = params["H_model"]
        local = horizontal_local_features(data)
        mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
        coefficients = np.asarray(model["coefficients"])
        design = np.concatenate([
            np.ones((*local.shape[:2], 1)), (local - mean) / scale,
        ], axis=2)
        predictions = design @ coefficients * data["bw_n"]
        predictions[:, 1, 1] *= -1.0
        force_contact = np.column_stack([
            np.any(body_force[:, :2, :] != 0.0, axis=(1, 2)),
            np.any(body_force[:, 2:, :] != 0.0, axis=(1, 2)),
        ])
        targets = _horizontal_symmetric_targets(
            data["net_force"][:, :2], force_contact, predictions,
            np.stack([body_force[:, :2, 2].sum(axis=1), body_force[:, 2:, 2].sum(axis=1)], axis=1),
            model["residual_allocation"],
        )
        for side in range(2):
            start = 0 if side == 0 else 2
            shares = body_shares(body_force, side)
            for axis in range(2):
                body_force[:, start:start + 2, axis] = targets[:, side, axis, None] * shares

    if "S" in components:
        split_contact = estimated_contacts(data, activation_threshold=float(params["S_contact_threshold"]))
        split = predict_linear(split_features(data, split_contact), np.asarray(params["S_coefficients"]))
        left_share = np.clip(split, 0.0, 1.0)
        active_count = split_contact.sum(axis=1)
        total_z = np.maximum(np.nan_to_num(data["net_force"][:, 2]), 0.0)
        desired = np.zeros((len(contact), 2), dtype=float)
        double = active_count == 2
        left_only = (active_count == 1) & split_contact[:, 0]
        right_only = (active_count == 1) & split_contact[:, 1]
        desired[double, 0] = total_z[double] * left_share[double]
        desired[double, 1] = total_z[double] - desired[double, 0]
        desired[left_only, 0] = total_z[left_only]
        desired[right_only, 1] = total_z[right_only]
        double = active_count == 2
        for side in range(2):
            frames = double
            if not frames.any():
                continue
            shares = body_shares(body_force, side)
            start = 0 if side == 0 else 2
            body_force[frames, start:start + 2, 2] = desired[frames, side, None] * shares[frames]

    if "P" in components:
        coefficients = np.asarray(params["P_coefficients"])
        for side in range(2):
            progress = np.clip(predict_linear(data["cop_features"][:, side, :], coefficients), 0.0, 1.0)
            heel = data["body_positions"][:, BODY_INDEX[("calcn_l", "calcn_r")[side]], :]
            toes = data["body_positions"][:, BODY_INDEX[("toes_l", "toes_r")[side]], :]
            point = heel + progress[:, None] * (toes - heel)
            start = 0 if side == 0 else 2
            body_point[:, start:start + 2, :] = point[:, None, :]

    for phase_variant in ("Pphase", "Pmix"):
        if phase_variant not in components:
            continue
        contact = estimated_contacts(data)
        features = cop_phase_features(data)
        coefficients = {
            name: np.asarray(value)
            for name, value in params[f"{phase_variant}_coefficients"].items()
        }
        for side in range(2):
            progress = _phase_cop_progress(features[:, side, :], contact, coefficients)
            active = contact[:, side] & np.isfinite(progress)
            if not active.any():
                continue
            heel = data["body_positions"][:, BODY_INDEX[("calcn_l", "calcn_r")[side]], :]
            toes = data["body_positions"][:, BODY_INDEX[("toes_l", "toes_r")[side]], :]
            point = heel + progress[:, None] * (toes - heel)
            start = 0 if side == 0 else 2
            body_point[active, start:start + 2, :] = point[active, None, :]

    foot_force = np.stack([body_force[:, :2].sum(axis=1), body_force[:, 2:].sum(axis=1)], axis=1)
    foot_cop = np.zeros((len(body_force), 2, 3), dtype=float)
    for side in range(2):
        start = 0 if side == 0 else 2
        force = body_force[:, start:start + 2]
        points = body_point[:, start:start + 2]
        weights = np.maximum(force[:, :, 2], 0.0)
        total = weights.sum(axis=1)
        midpoint = points.mean(axis=1)
        weighted = np.divide(
            np.sum(points * weights[:, :, None], axis=1), total[:, None],
            out=midpoint.copy(), where=total[:, None] > 1e-8,
        )
        foot_cop[:, side] = weighted
    return {"body_force": body_force, "body_point": body_point, "foot_force": foot_force,
            "foot_cop": foot_cop, "contact": contact}


def metric_bundle(segments: list[dict[str, Any]], variant: str, params: dict[str, Any], phase_name: str = "total") -> dict[str, float]:
    # Match Table I: compute each metric per trial, average the two feet, then
    # average trials within subject. False contacts use the separately defined
    # pooled per-subject flight denominator.
    trials: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {key: [] for key in ("fz", "horizontal", "f1_true", "f1_pred", "cop")}
    )
    false_num = false_den = 0
    for data in segments:
        candidate = apply_variant(data, variant, params)
        selected = data["force_valid"] & (np.ones(len(data["times"]), dtype=bool)
                    if phase_name == "total" else data["phase"] == phase_name)
        trial = trials[data["trial"]]
        truth = data["true_contact"]
        for side in range(2):
            stance = selected & truth[:, side]
            if stance.any():
                error_z = (candidate["foot_force"][stance, side, 2]
                           - data["measured_force"][stance, side, 2]) / data["bw_n"]
                error_xy = (candidate["foot_force"][stance, side, :2]
                            - data["measured_force"][stance, side, :2]) / data["bw_n"]
                trial["fz"].append(np.column_stack([np.full(error_z.size, side), error_z]))
                trial["horizontal"].append(np.column_stack([np.full(error_xy.shape[0], side),
                                                               np.linalg.norm(error_xy, axis=1)]))
            f1_mask = selected & np.isfinite(data["measured_force"][:, side, 2])
            if f1_mask.any():
                trial["f1_true"].append(np.column_stack([np.full(f1_mask.sum(), side),
                                                          truth[f1_mask, side].astype(float)]))
                trial["f1_pred"].append(np.column_stack([np.full(f1_mask.sum(), side),
                                                          candidate["contact"][f1_mask, side].astype(float)]))
            cop_mask = selected & ~data["cop_invalid"] & truth[:, side]
            if cop_mask.any():
                distances = np.linalg.norm(candidate["foot_cop"][cop_mask, side, :2]
                                           - data["measured_cop"][cop_mask, side], axis=1) * 100.0
                trial["cop"].append(np.column_stack([np.full(distances.size, side), distances]))
            flight = data["valid_target"] & (data["measured_raw"][:, side, 2] <= 1.0)
            if phase_name != "total":
                flight &= data["phase"] == phase_name
            false_num += int(np.sum(candidate["foot_force"][flight, side, 2] > FALSE_CONTACT_N))
            false_den += int(flight.sum())

    per_trial: dict[str, list[float]] = {key: [] for key in ("fz", "horizontal", "f1", "cop")}
    for metrics in trials.values():
        for key in ("fz", "horizontal", "cop"):
            values = metrics[key]
            if not values:
                continue
            arr = np.concatenate(values)
            per_side = []
            for side in (0, 1):
                x = arr[arr[:, 0] == side, 1]
                if len(x):
                    per_side.append(float(np.sqrt(np.mean(x ** 2))) if key in ("fz", "horizontal")
                                    else float(np.mean(x)))
            if per_side:
                per_trial[key].append(float(np.mean(per_side)))
        if metrics["f1_true"]:
            truth = np.concatenate(metrics["f1_true"])
            pred = np.concatenate(metrics["f1_pred"])
            side_scores = []
            for side in (0, 1):
                mask = truth[:, 0] == side
                y, yp = truth[mask, 1].astype(bool), pred[mask, 1].astype(bool)
                denom = 2 * np.sum(y & yp) + np.sum(~y & yp) + np.sum(y & ~yp)
                if denom:
                    side_scores.append(float(2 * np.sum(y & yp) / denom))
            if side_scores:
                per_trial["f1"].append(float(np.mean(side_scores)))

    mean = lambda values: float(np.mean(values)) if values else math.nan
    return {
        "fz_rmse_contact_bw": mean(per_trial["fz"]),
        "contact_f1": mean(per_trial["f1"]),
        "cop_error_cm": mean(per_trial["cop"]),
        "horizontal_rmse_bw": mean(per_trial["horizontal"]),
        "false_contact_pooled": false_num / false_den if false_den else math.nan,
        "false_contact_numerator": false_num,
        "false_contact_denominator": false_den,
    }


def objective(metrics: dict[str, float], baseline: dict[str, float]) -> float:
    ratios = []
    for name in ("fz_rmse_contact_bw", "cop_error_cm", "horizontal_rmse_bw", "false_contact_pooled"):
        value, base = metrics[name], baseline[name]
        if math.isfinite(value) and math.isfinite(base) and base > 1e-12:
            ratios.append(value / base)
    if math.isfinite(metrics["contact_f1"]) and math.isfinite(baseline["contact_f1"]):
        base_error = max(1.0 - baseline["contact_f1"], 1e-6)
        ratios.append((1.0 - metrics["contact_f1"]) / base_error)
    return float(np.mean(ratios)) if ratios else math.nan


def laterality_diagnostics(
    segments: list[dict[str, Any]], fold: str, variant: str, params: dict[str, Any],
) -> list[dict[str, Any]]:
    errors: dict[tuple[int, int], list[float]] = defaultdict(list)
    measured_means: dict[tuple[int, int], list[float]] = defaultdict(list)
    estimated_means: dict[tuple[int, int], list[float]] = defaultdict(list)
    for data in segments:
        candidate = apply_variant(data, variant, params)
        for side in range(2):
            mask = data["valid_target"] & data["force_valid"] & data["true_contact"][:, side]
            for axis in range(2):
                selected = mask & np.isfinite(candidate["foot_force"][:, side, axis])
                if not selected.any():
                    continue
                key = (side, axis)
                measured = data["measured_force"][selected, side, axis] / data["bw_n"]
                estimated = candidate["foot_force"][selected, side, axis] / data["bw_n"]
                errors[key].extend((estimated - measured).tolist())
                measured_means[key].extend(measured.tolist())
                estimated_means[key].extend(estimated.tolist())
    rows = []
    for side in range(2):
        for axis in range(2):
            key = (side, axis)
            error = np.asarray(errors[key], dtype=float)
            measured = np.asarray(measured_means[key], dtype=float)
            estimated = np.asarray(estimated_means[key], dtype=float)
            rows.append({
                "fold": fold, "variant": variant, "side": ("left", "right")[side],
                "axis": ("global_x", "global_y")[axis], "n": int(len(error)),
                "rmse_bw": float(np.sqrt(np.mean(error * error))) if len(error) else math.nan,
                "measured_mean_bw": float(np.mean(measured)) if len(measured) else math.nan,
                "estimated_mean_bw": float(np.mean(estimated)) if len(estimated) else math.nan,
            })
    return rows


def subject_segments(all_data: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for data in all_data:
        grouped[data["subject"]].append(data)
    return dict(grouped)


def score_subjects(segments_by_subject: dict[str, list[dict[str, Any]]], variant: str, params: dict[str, Any], subjects: list[str]) -> dict[str, dict[str, float]]:
    return {subject: metric_bundle(segments_by_subject[subject], variant, params)
            for subject in subjects if subject in segments_by_subject}


def mean_training_objective(
    segments_by_subject: dict[str, list[dict[str, Any]]], variant: str, params: dict[str, Any],
    baseline_by_subject: dict[str, dict[str, float]], subjects: list[str],
) -> float:
    scores = []
    for subject in subjects:
        if subject not in segments_by_subject:
            continue
        score = objective(metric_bundle(segments_by_subject[subject], variant, params), baseline_by_subject[subject])
        if math.isfinite(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else math.inf


def _source_path(path: str, root: str, local_root: Path) -> Path:
    normalized, normalized_root = str(Path(path)), str(Path(root))
    if normalized == normalized_root:
        return local_root
    if normalized.startswith(normalized_root + "/"):
        return local_root / normalized[len(normalized_root) + 1:]
    return Path(path)


def load_segment(cache_row: dict[str, str], args: argparse.Namespace, eval_config: EvaluationConfig) -> dict[str, Any]:
    subject, trial, segment = cache_row["subject"], cache_row["trial"], int(cache_row["segment"])
    target_dir = args.torque_root / TARGET_DIRNAME / subject / trial
    target_path = target_dir / "torque_targets.csv"
    metadata = json.loads((target_dir / "run_metadata.json").read_text())
    target_all = pd.read_csv(target_path)
    target = target_all[target_all.segment.astype(int) == segment].reset_index(drop=True)
    times = target.time.to_numpy(dtype=float)
    segment_dir = target_dir / "segments" / f"segment_{segment}"
    measured_path = next(segment_dir.glob("*_grf_measured.mot"))
    estimated_dir = segment_dir / "estimated_id"
    estimated_path = next(estimated_dir.glob("*_grf_estimated.mot"))
    measured_all, estimated_all = read_storage(measured_path), read_storage(estimated_path)
    measured_idx = align_indices(measured_all.time.to_numpy(dtype=float), times, measured_path)
    estimated_idx = align_indices(estimated_all.time.to_numpy(dtype=float), times, estimated_path)
    measured = measured_all.iloc[measured_idx].reset_index(drop=True)
    estimated = estimated_all.iloc[estimated_idx].reset_index(drop=True)

    pipeline_path = map_source_path(metadata["source_paths"]["pipeline_csv"], eval_config.source_torque_root, args.torque_root)
    pipeline_all = pd.read_csv(pipeline_path)
    pipeline_idx = align_indices(pipeline_all.time.to_numpy(dtype=float), times, pipeline_path)
    pipeline = pipeline_all.iloc[pipeline_idx].reset_index(drop=True)
    model_path = map_source_path(metadata["source_paths"]["model"], eval_config.source_torque_root, args.torque_root)
    force_path = _force_path(metadata["source_paths"]["measured_force"], eval_config)
    force_raw, cop_raw = _load_force_arrays(force_path)
    force_filtered = _lowpass_force(force_raw, 250.0, 12.0)
    raw_index = np.rint(times * 250.0).astype(int)
    in_range = (raw_index >= 0) & (raw_index < len(force_filtered))
    safe_index = np.clip(raw_index, 0, len(force_filtered) - 1)
    measured_force = force_filtered[safe_index].copy()
    measured_raw = force_raw[safe_index].copy()
    measured_cop = cop_raw[safe_index, :, :2].copy()
    measured_force[~in_range] = np.nan
    measured_raw[~in_range] = np.nan
    measured_cop[~in_range] = np.nan
    shift = np.asarray(metadata.get("cop_shift_m", [0.0, 0.0]), dtype=float).reshape(-1)
    if shift.size >= 2:
        if metadata.get("cop_shift_source") == "tcode":
            measured_cop += shift[:2][None, None, :]
        else:
            measured_cop -= shift[:2][None, None, :]

    cache = read_cache(Path(cache_row["cache_file"]), cache_row["cache_prefix"])
    cache_idx = align_indices(cache["times"].astype(float), times, Path(cache_row["cache_file"]))
    body_positions = cache["body_positions"][cache_idx]
    body_names = [str(name) for name in cache["body_names"].tolist()]
    body_index = {name: index for index, name in enumerate(body_names)}
    body_force = np.stack([
        np.column_stack([estimated[f"ground_force_{name}_v{axis}"].to_numpy(dtype=float) for axis in "xyz"])
        for name in BODY_NAMES
    ], axis=1)
    body_point = np.stack([
        np.column_stack([estimated[f"ground_force_{name}_p{axis}"].to_numpy(dtype=float) for axis in "xyz"])
        for name in BODY_NAMES
    ], axis=1)
    activation = np.stack([
        np.column_stack([
            pipeline.get(f"calcn_{side}_contact", pd.Series(np.zeros(len(times)))).to_numpy(dtype=float),
            pipeline.get(f"toes_{side}_contact", pd.Series(np.zeros(len(times)))).to_numpy(dtype=float),
        ])
        for side in "lr"
    ], axis=1)
    base_contact = np.column_stack([((activation[:, side, :] > 0.05).any(axis=1)) for side in range(2)])

    cw = read_contact_wrenches(model_path, pipeline_path, segment, times)
    net_force = cw[[f"target_force_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
    measured_fz = measured_force[:, :, 2]
    true_contact = measured_fz > 0.05 * (float(pipeline.subject_mass_kg.iloc[0]) * GRAVITY)
    bw_n = float(pipeline.subject_mass_kg.iloc[0]) * GRAVITY
    phase, _ = classify_phases(times, measured_force, measured_raw, bw_n)
    force_valid = (
        in_range & np.isfinite(measured_force[:, :, 2]).all(axis=1) & np.isfinite(net_force[:, 2])
        & (np.abs(measured_force[:, :, 2].sum(axis=1) - net_force[:, 2]) <= 0.25 * bw_n)
    )
    valid_target = target.valid.to_numpy(dtype=int) == 1
    cop_invalid = target.cop_invalid.to_numpy(dtype=int) != 0
    cop_feat, incline = cop_features({
        "times": times, "body_positions": body_positions, "base_contact": base_contact, "activation": activation,
    })
    return {
        "subject": subject, "trial": trial, "segment": segment, "times": times,
        "target_path": target_path, "target": target, "metadata": metadata,
        "segment_dir": segment_dir, "estimated_dir": estimated_dir,
        "estimated_path": estimated_path, "measured_path": measured_path,
        "model_path": model_path, "pipeline_path": pipeline_path,
        "measured": measured, "estimated": estimated, "pipeline": pipeline,
        "force_raw": force_raw, "measured_raw": measured_raw, "measured_force": measured_force,
        "measured_cop": measured_cop, "body_positions": body_positions,
        "body_height": cache["body_heights_above_floor"][cache_idx],
        "body_speed": cache["body_speeds"][cache_idx], "com": cache["com_positions"][cache_idx],
        "floor_height": float(cache["floor_height"][0]), "body_index": body_index,
        "body_force": body_force, "body_point": body_point,
        "activation": activation, "base_contact": base_contact,
        "true_contact": true_contact, "phase": phase, "net_force": net_force,
        "force_valid": force_valid, "valid_target": valid_target,
        "cop_invalid": cop_invalid, "bw_n": bw_n,
        "cop_features": cop_feat, "foot_incline": incline,
        "measured_cop_raw": cop_raw, "in_range": in_range,
    }


def mean_training_horizontal_rmse(
    by_subject: dict[str, list[dict[str, Any]]], params: dict[str, Any], subjects: list[str],
) -> float:
    scores = []
    for subject in subjects:
        squared_errors = []
        for data in by_subject.get(subject, []):
            candidate = apply_variant(data, "Hsym", params)
            for side in range(2):
                mask = data["valid_target"] & data["force_valid"] & data["true_contact"][:, side]
                error = (candidate["foot_force"][mask, side, :2]
                         - data["measured_force"][mask, side, :2]) / data["bw_n"]
                if len(error):
                    squared_errors.extend(np.sum(error * error, axis=1).tolist())
        if squared_errors:
            scores.append(float(np.sqrt(np.mean(squared_errors))))
    return float(np.mean(scores)) if scores else math.inf


def fit_variant_params(train_segments: list[dict[str, Any]], train_subjects: list[str], by_subject: dict[str, list[dict[str, Any]]], baseline: dict[str, dict[str, float]], selected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {"BASELINE": {}}

    h_best: tuple[float, dict[str, Any]] | None = None
    for ridge in RIDGE_GRID:
        params = {"H_model": fit_horizontal_model(train_segments, ridge), "H_ridge": ridge}
        score = mean_training_objective(by_subject, "H", params, baseline, train_subjects)
        if h_best is None or score < h_best[0]:
            h_best = (score, params)
    assert h_best is not None
    candidates["H"] = {**h_best[1], "training_J": h_best[0], "budget_configs": len(RIDGE_GRID)}

    # Five predeclared activation cutoffs gate only the estimated double-stance split.
    s_best: tuple[float, dict[str, Any]] | None = None
    for threshold in S_CONTACT_GRID:
        coeff = fit_split_model(train_segments, 0.0, activation_threshold=threshold)
        params = {"S_coefficients": coeff.tolist(), "S_ridge": 0.0,
                  "S_contact_threshold": threshold}
        score = mean_training_objective(by_subject, "S", params, baseline, train_subjects)
        if s_best is None or score < s_best[0]:
            s_best = (score, params)
    assert s_best is not None
    candidates["S"] = {**s_best[1], "training_J": s_best[0], "budget_configs": len(S_CONTACT_GRID)}

    # P uses five declared ridge settings; labels are measured only on training subjects.
    p_best: tuple[float, dict[str, Any]] | None = None
    for ridge in RIDGE_GRID:
        coeff = fit_cop_model(train_segments, ridge)
        params = {"P_coefficients": coeff.tolist(), "P_ridge": ridge}
        score = mean_training_objective(by_subject, "P", params, baseline, train_subjects)
        if p_best is None or score < p_best[0]:
            p_best = (score, params)
    assert p_best is not None
    candidates["P"] = {**p_best[1], "training_J": p_best[0], "budget_configs": len(RIDGE_GRID)}

    # Declared Pphase search: 5 single-stance × 5 double-stance ridge pairs per fold.
    phase_models = {
        phase: {ridge: fit_cop_model(train_segments, ridge, phase) for ridge in RIDGE_GRID}
        for phase in ("single_stance", "double_stance")
    }
    pphase_best: tuple[float, dict[str, Any]] | None = None
    for single_ridge in RIDGE_GRID:
        for double_ridge in RIDGE_GRID:
            params = {
                "Pphase_coefficients": {
                    phase: phase_models[phase][ridge].tolist()
                    for phase, ridge in (("single_stance", single_ridge), ("double_stance", double_ridge))
                },
                "Pphase_ridges": {"single_stance": single_ridge, "double_stance": double_ridge},
            }
            score = mean_training_objective(by_subject, "Pphase", params, baseline, train_subjects)
            if pphase_best is None or score < pphase_best[0]:
                pphase_best = (score, params)
    assert pphase_best is not None
    candidates["Pphase"] = {**pphase_best[1], "training_J": pphase_best[0], "budget_configs": 25}

    # Pmix pools measured single- and double-stance targets for the single model.
    pmix_single = {ridge: fit_cop_model(train_segments, ridge, phase_features=True)
                   for ridge in RIDGE_GRID}
    pmix_best: tuple[float, dict[str, Any]] | None = None
    for single_ridge in RIDGE_GRID:
        for double_ridge in RIDGE_GRID:
            params = {
                "Pmix_coefficients": {
                    "single_stance": pmix_single[single_ridge].tolist(),
                    "double_stance": phase_models["double_stance"][double_ridge].tolist(),
                },
                "Pmix_ridges": {"single_stance": single_ridge, "double_stance": double_ridge},
            }
            score = mean_training_objective(by_subject, "Pmix", params, baseline, train_subjects)
            if pmix_best is None or score < pmix_best[0]:
                pmix_best = (score, params)
    assert pmix_best is not None
    candidates["Pmix"] = {**pmix_best[1], "training_J": pmix_best[0], "budget_configs": 25}

    # Declared Hsym search: 5 ridge values × 2 residual-allocation rules per fold.
    hsym_best: tuple[float, dict[str, Any]] | None = None
    for ridge in RIDGE_GRID:
        for allocation in ("equal", "vertical"):
            model = fit_horizontal_symmetric_model(train_segments, ridge, allocation)
            params = {"H_model": model, "H_ridge": ridge}
            score = mean_training_horizontal_rmse(by_subject, params, train_subjects)
            if hsym_best is None or score < hsym_best[0]:
                hsym_best = (score, params)
    assert hsym_best is not None
    candidates["Hsym"] = {**hsym_best[1], "training_horizontal_rmse": hsym_best[0],
                          "budget_configs": len(RIDGE_GRID) * 2}

    s_ridge = float(candidates["S"]["S_ridge"])
    s_threshold = float(candidates["S"]["S_contact_threshold"])
    p_params = candidates["P"]
    candidates["H+P"] = {"H_model": candidates["H"]["H_model"], "H_ridge": candidates["H"]["H_ridge"],
                         "P_coefficients": p_params["P_coefficients"], "P_ridge": p_params["P_ridge"]}
    candidates["S+P"] = {"S_coefficients": candidates["S"]["S_coefficients"], "S_ridge": s_ridge,
                         "S_contact_threshold": s_threshold,
                         "P_coefficients": p_params["P_coefficients"], "P_ridge": p_params["P_ridge"]}
    candidates["H+S+P"] = {"H_model": candidates["H"]["H_model"], "H_ridge": candidates["H"]["H_ridge"],
                           "S_coefficients": candidates["S"]["S_coefficients"], "S_ridge": s_ridge,
                           "S_contact_threshold": s_threshold,
                           "P_coefficients": p_params["P_coefficients"], "P_ridge": p_params["P_ridge"]}
    candidates["Hsym+P"] = {"H_model": candidates["Hsym"]["H_model"], "H_ridge": candidates["Hsym"]["H_ridge"],
                             "P_coefficients": p_params["P_coefficients"], "P_ridge": p_params["P_ridge"]}
    candidates["Hsym+Pphase"] = {
        "H_model": candidates["Hsym"]["H_model"], "H_ridge": candidates["Hsym"]["H_ridge"],
        "Pphase_coefficients": candidates["Pphase"]["Pphase_coefficients"],
        "Pphase_ridges": candidates["Pphase"]["Pphase_ridges"],
    }
    candidates["Hsym+Pmix"] = {
        "H_model": candidates["Hsym"]["H_model"], "H_ridge": candidates["Hsym"]["H_ridge"],
        "Pmix_coefficients": candidates["Pmix"]["Pmix_coefficients"],
        "Pmix_ridges": candidates["Pmix"]["Pmix_ridges"],
    }
    for name in ("H+P", "S+P", "H+S+P", "Hsym+P", "Hsym+Pphase", "Hsym+Pmix"):
        score = mean_training_objective(by_subject, name, candidates[name], baseline, train_subjects)
        candidates[name]["training_J"] = score
        candidates[name]["budget_configs"] = 1
    return candidates


def parameter_rows(fold: str, variant: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    if variant == "BASELINE":
        return [{"fold": fold, "variant": variant, "parameter": "default_wrench_max_abs_delta",
                 "value": 0.0, "training_J": 1.0, "budget_configs": 1}]
    for component in components_for(variant):
        if component == "A":
            for key, value in params["A"].items():
                output.append({"fold": fold, "variant": variant, "parameter": f"A.{key}",
                               "value": value, "training_J": params.get("training_J"),
                               "budget_configs": params.get("budget_configs")})
        if component == "Hsym":
            output.append({"fold": fold, "variant": variant, "parameter": "Hsym.ridge",
                           "value": params["H_ridge"], "training_J": params.get("training_J"),
                           "budget_configs": params.get("budget_configs")})
            output.append({"fold": fold, "variant": variant, "parameter": "Hsym.residual_allocation",
                           "value": params["H_model"]["residual_allocation"],
                           "training_J": params.get("training_J"),
                           "budget_configs": params.get("budget_configs")})
            for name in ("mean", "scale"):
                for index, value in enumerate(params["H_model"][name]):
                    output.append({"fold": fold, "variant": variant,
                                   "parameter": f"Hsym.{name}_{index}", "value": value,
                                   "training_J": params.get("training_J"),
                                   "budget_configs": params.get("budget_configs")})
            for index, pair in enumerate(params["H_model"]["coefficients"]):
                for axis, value in zip(("x", "y"), pair):
                    output.append({"fold": fold, "variant": variant,
                                   "parameter": f"Hsym.coef_{index}_{axis}", "value": value,
                                   "training_J": params.get("training_J"),
                                   "budget_configs": params.get("budget_configs")})
        if component == "H":
            output.append({"fold": fold, "variant": variant, "parameter": "H.ridge",
                           "value": params["H_ridge"], "training_J": params.get("training_J"),
                           "budget_configs": params.get("budget_configs")})
            for name in ("mean", "scale"):
                for index, value in enumerate(params["H_model"][name]):
                    output.append({"fold": fold, "variant": variant,
                                   "parameter": f"H.{name}_{index}", "value": value,
                                   "training_J": params.get("training_J"),
                                   "budget_configs": params.get("budget_configs")})
            for index, pair in enumerate(params["H_model"]["coefficients"]):
                for axis, value in zip(("x", "y"), pair):
                    output.append({"fold": fold, "variant": variant,
                                   "parameter": f"H.coef_{index}_{axis}", "value": value,
                                   "training_J": params.get("training_J"),
                                   "budget_configs": params.get("budget_configs")})
        if component == "S":
            output.append({"fold": fold, "variant": variant, "parameter": "S.contact_threshold",
                           "value": params["S_contact_threshold"], "training_J": params.get("training_J"),
                           "budget_configs": params.get("budget_configs")})
            for index, value in enumerate(params["S_coefficients"]):
                output.append({"fold": fold, "variant": variant, "parameter": f"S.coef_{index}",
                               "value": value, "training_J": params.get("training_J"),
                               "budget_configs": params.get("budget_configs")})
            output.append({"fold": fold, "variant": variant, "parameter": "S.ridge",
                           "value": params["S_ridge"], "training_J": params.get("training_J"),
                           "budget_configs": params.get("budget_configs")})
        if component in ("Pphase", "Pmix"):
            for phase, coefficients in params[f"{component}_coefficients"].items():
                output.append({"fold": fold, "variant": variant,
                               "parameter": f"{component}.{phase}.ridge",
                               "value": params[f"{component}_ridges"][phase],
                               "training_J": params.get("training_J"),
                               "budget_configs": params.get("budget_configs")})
                for index, value in enumerate(coefficients):
                    output.append({"fold": fold, "variant": variant,
                                   "parameter": f"{component}.{phase}.coef_{index}", "value": value,
                                   "training_J": params.get("training_J"),
                                   "budget_configs": params.get("budget_configs")})
        if component == "P":
            for index, value in enumerate(params["P_coefficients"]):
                output.append({"fold": fold, "variant": variant, "parameter": f"P.coef_{index}",
                               "value": value, "training_J": params.get("training_J"),
                               "budget_configs": params.get("budget_configs")})
            output.append({"fold": fold, "variant": variant, "parameter": "P.ridge",
                           "value": params["P_ridge"], "training_J": params.get("training_J"),
                           "budget_configs": params.get("budget_configs")})
    return output


def write_selected_id_inputs(data: dict[str, Any], variant: str, params: dict[str, Any], study_root: Path) -> dict[str, str]:
    from prepare_hybrid_ids import find_one

    candidate = apply_variant(data, variant, params)
    source_columns, source_rows, _ = read_mot(data["estimated_path"])
    source_map = {name: index for index, name in enumerate(source_columns)}
    source_times = np.asarray([row[source_map["time"]] for row in source_rows], dtype=float)
    target_to_source = align_indices(source_times, data["times"], data["estimated_path"])
    for name, index in source_map.items():
        if name.startswith("ground_force_") and ("_v" in name or "_p" in name):
            body = next((body for body in BODY_NAMES if name.startswith(f"ground_force_{body}_")), None)
            if body is None:
                continue
            body_index = BODY_INDEX[body]
            kind = name.rsplit("_", 1)[-1]
            if len(kind) != 2 or kind[0] not in "vp":
                continue
            axis_index = "xyz".index(kind[1])
            array = candidate["body_force"] if kind[0] == "v" else candidate["body_point"]
            for target_index, source_index in enumerate(target_to_source):
                source_rows[source_index][index] = float(array[target_index, body_index, axis_index])

    segment_dir = data["segment_dir"]
    setup = find_one(list(data["estimated_dir"].glob("Setup_ID_estimatedGRF_*.xml")), "estimated ID setup")
    estimated_xml = find_one(list(data["estimated_dir"].glob("*_external_forces_estimated.xml")), "estimated ExternalLoads XML")
    target_sto = find_one(list(data["segment_dir"].glob("*_id_measuredGRF.sto")), "measured target ID STO")
    estimated_sto = find_one(list(data["estimated_dir"].glob("*_id_estimatedGRF.sto")), "estimated baseline ID STO")
    output_mot_dir = study_root / "phase_d" / "id_inputs" / variant / data["subject"] / data["trial"] / f"segment_{data['segment']}"
    mot_name = f"{data['subject']}_{data['trial']}_segment_{data['segment']}_grf_{variant.lower().replace('+', '_')}.mot"
    mot_path = output_mot_dir / mot_name
    write_mot(mot_path, source_columns, source_rows, mot_name.removesuffix(".mot"))
    external_xml = mot_path.with_name(mot_name.replace(".mot", "_external.xml"))
    make_external_xml(estimated_xml, external_xml, mot_name, variant.replace("+", "_"))
    output_dir = study_root / "phase_d" / "id_outputs" / variant / data["subject"] / data["trial"] / f"segment_{data['segment']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "variant": variant, "subject": data["subject"], "trial": data["trial"], "segment": str(data["segment"]),
        "setup": str(setup.resolve()), "external_loads_file": str(external_xml.resolve()),
        "output_dir": str(output_dir.resolve()), "output_name": f"id_{variant.lower().replace('+', '_')}.sto",
        "forces_to_exclude": "Actuators", "target_sto": str(target_sto.resolve()),
        "estimated_sto": str(estimated_sto.resolve()), "target_csv": str(data["target_path"].resolve()),
        "measured_mot": str(data["measured_path"].resolve()), "estimated_mot": str(data["estimated_path"].resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-root", type=Path, required=True)
    parser.add_argument("--force-root", type=Path, required=True)
    parser.add_argument("--source-torque-root", default=DEFAULT_SOURCE_TORQUE_ROOT)
    parser.add_argument("--source-force-root", default=DEFAULT_SOURCE_FORCE_ROOT)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--cache-subdir", default="kinematics_cache_h2_unwrapped")
    parser.add_argument("--phase-d-variants", help="Comma-separated variants to prepare for held-out ID")
    args = parser.parse_args()

    eval_config = EvaluationConfig(
        torque_root=args.torque_root, force_root=args.force_root,
        output_dir=args.study_root / "phase_c", source_torque_root=args.source_torque_root,
        source_force_root=args.source_force_root,
    )
    cache_rows: list[dict[str, str]] = []
    for subject in SUBJECTS:
        manifest = args.study_root / "phase_b" / args.cache_subdir / subject / "cache_manifest.csv"
        with manifest.open(newline="") as handle:
            cache_rows.extend(csv.DictReader(handle))
    all_data = []
    for index, row in enumerate(cache_rows, start=1):
        all_data.append(load_segment(row, args, eval_config))
        if index % 50 == 0 or index == len(cache_rows):
            print(f"loaded_segments={index}/{len(cache_rows)}", flush=True)
    by_subject = subject_segments(all_data)
    default_wrench_max_abs_delta = 0.0
    for data in all_data:
        default = apply_variant(data, "BASELINE", {})
        force_delta = float(np.max(np.abs(default["body_force"] - data["body_force"])))
        point_delta = float(np.max(np.abs(default["body_point"] - data["body_point"])))
        default_wrench_max_abs_delta = max(default_wrench_max_abs_delta, force_delta, point_delta)
    if default_wrench_max_abs_delta > 1e-6:
        raise RuntimeError(f"Default no-op changed an estimated wrench by {default_wrench_max_abs_delta:.9g}")
    baseline_by_subject = {
        subject: metric_bundle(segments, "BASELINE", {}) for subject, segments in by_subject.items()
    }
    for subject, values in baseline_by_subject.items():
        values["J"] = objective(values, values)

    ablation_rows: list[dict[str, Any]] = []
    params_rows: list[dict[str, Any]] = []
    all_fold_params: dict[str, dict[str, dict[str, Any]]] = {}
    test_scores: dict[str, list[float]] = defaultdict(list)
    laterality_rows: list[dict[str, Any]] = []
    for heldout in SUBJECTS:
        train_subjects = [subject for subject in SUBJECTS if subject != heldout]
        train_segments = [data for data in all_data if data["subject"] in train_subjects]
        chosen = fit_variant_params(train_segments, train_subjects, by_subject, baseline_by_subject, {})
        chosen["BASELINE"] = {"training_J": 1.0, "budget_configs": 1}
        all_fold_params[heldout] = chosen
        test_segments = by_subject[heldout]
        for variant in ("BASELINE", "H+P"):
            laterality_rows.extend(laterality_diagnostics(
                test_segments, heldout, variant, chosen.get(variant, {}),
            ))
        for variant, params in chosen.items():
            params_rows.extend(parameter_rows(heldout, variant, params))
            for phase_name in PHASES:
                values = metric_bundle(test_segments, variant, params, phase_name)
                baseline = metric_bundle(test_segments, "BASELINE", {}, phase_name)
                score = objective(values, baseline)
                if phase_name == "total":
                    test_scores[variant].append(score)
                ablation_rows.append({
                    "variant": variant, "fold": heldout, "subject": heldout, "phase": phase_name,
                    **values, "J_vs_baseline": score,
                    "parameters": json.dumps(params, sort_keys=True, separators=(",", ":")),
                })
        print(f"fold={heldout} train={len(train_subjects)} test_segments={len(test_segments)} "
              f"best_H_ridge={chosen['H']['H_ridge']} best_S_ridge={chosen['S']['S_ridge']} "
              f"best_P_ridge={chosen['P']['P_ridge']}", flush=True)

    phase_c = args.study_root / "phase_c"
    phase_c.mkdir(parents=True, exist_ok=True)
    write_csv(phase_c / "ablation.csv", ablation_rows)
    write_csv(phase_c / "loso_params.csv", params_rows)
    write_csv(phase_c / "laterality.csv", laterality_rows)
    ranking = sorted(
        ((variant, float(np.nanmean(scores))) for variant, scores in test_scores.items() if variant != "BASELINE"),
        key=lambda item: item[1],
    )
    selected_variants = ([name.strip() for name in args.phase_d_variants.split(",") if name.strip()]
                         if args.phase_d_variants else
                         [variant for variant, score in ranking[:3] if math.isfinite(score)])
    unknown = set(selected_variants) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown Phase D variants: {sorted(unknown)}")
    summary = {
        "subjects": list(SUBJECTS), "segments": len(all_data), "budget": {
            "H_configs_per_fold": len(RIDGE_GRID), "Hsym_configs_per_fold": len(RIDGE_GRID) * 2,
            "S_configs_per_fold": len(S_CONTACT_GRID), "P_configs_per_fold": len(RIDGE_GRID),
            "Pphase_configs_per_fold": len(RIDGE_GRID) ** 2,
            "Pmix_configs_per_fold": len(RIDGE_GRID) ** 2,
            "combination_configs_per_fold": 6,
        },
        "variants": {variant: {"mean_loso_J": score, "fold_J": test_scores[variant]}
                     for variant, score in ranking},
        "selected_for_phase_d": selected_variants,
        "default_wrench_max_abs_delta": default_wrench_max_abs_delta,
        "measured_force_or_cop_used_at_inference": False,
    }
    model_dir = phase_c / "model_exports"
    model_dir.mkdir(parents=True, exist_ok=True)
    pphase_ridges = {
        phase: Counter(all_fold_params[subject]["Pphase"]["Pphase_ridges"][phase]
                       for subject in SUBJECTS).most_common(1)[0][0]
        for phase in ("single_stance", "double_stance")
    }
    pphase_model = {
        phase: fit_cop_model(all_data, ridge, phase).tolist()
        for phase, ridge in pphase_ridges.items()
    }
    pphase_path = model_dir / "groundlink_pphase_all_subjects.json"
    pphase_path.write_text(json.dumps({
        "schema_version": 1,
        "feature_schema": "estimated_contact_binary_v1",
        "ridge_selection": "modal_LOSO_training_fold",
        "ridges": pphase_ridges,
        "coefficients_by_phase": pphase_model,
    }, indent=2) + "\n", encoding="utf-8")
    pmix_ridges = {
        phase: Counter(all_fold_params[subject]["Pmix"]["Pmix_ridges"][phase]
                       for subject in SUBJECTS).most_common(1)[0][0]
        for phase in ("single_stance", "double_stance")
    }
    pmix_model = {
        "single_stance": fit_cop_model(all_data, pmix_ridges["single_stance"], phase_features=True).tolist(),
        "double_stance": fit_cop_model(all_data, pmix_ridges["double_stance"], "double_stance").tolist(),
    }
    pmix_path = model_dir / "groundlink_pmix_all_subjects.json"
    pmix_path.write_text(json.dumps({
        "schema_version": 1,
        "feature_schema": "estimated_contact_binary_v1",
        "single_stance_fit": "all_measured_contact_stance_targets",
        "ridge_selection": "modal_LOSO_training_fold",
        "ridges": pmix_ridges,
        "coefficients_by_phase": pmix_model,
    }, indent=2) + "\n", encoding="utf-8")
    hsym_models = [all_fold_params[subject]["Hsym"] for subject in SUBJECTS]
    hsym_ridge = Counter(params["H_ridge"] for params in hsym_models).most_common(1)[0][0]
    hsym_allocation = Counter(params["H_model"]["residual_allocation"]
                              for params in hsym_models).most_common(1)[0][0]
    hsym_model = fit_horizontal_symmetric_model(all_data, hsym_ridge, hsym_allocation)
    hsym_path = model_dir / "groundlink_hsym_all_subjects.json"
    hsym_path.write_text(json.dumps(hsym_model, indent=2) + "\n", encoding="utf-8")
    summary["model_exports"] = {
        "Hsym": str(hsym_path), "Pphase": str(pphase_path), "Pmix": str(pmix_path),
        "selection": "modal hyperparameters across LOSO training folds; coefficients refit on all subjects",
    }
    (phase_c / "selection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_rows: list[dict[str, str]] = []
    for subject in SUBJECTS:
        chosen = all_fold_params[subject]
        for variant in selected_variants:
            for data in by_subject[subject]:
                manifest_rows.append(write_selected_id_inputs(data, variant, chosen[variant], args.study_root))
    write_csv(args.study_root / "phase_d" / "id_manifest.csv", manifest_rows)
    metadata = {
        "selected_variants": selected_variants,
        "manifest_rows": len(manifest_rows),
        "source_cache": args.cache_subdir,
        "inference_inputs": ["pipeline estimated forces and activations", "kinematics cache"],
        "measured_data_use": "training labels and direct-GRF evaluation targets only",
        "default_wrench_max_abs_delta": default_wrench_max_abs_delta,
    }
    (args.study_root / "phase_d" / "id_manifest_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"segments={len(all_data)} selected={','.join(selected_variants)} id_rows={len(manifest_rows)} "
          f"ablation={phase_c / 'ablation.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

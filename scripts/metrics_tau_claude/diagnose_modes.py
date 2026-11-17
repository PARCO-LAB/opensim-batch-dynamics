#!/usr/bin/env python3
"""Compute the GroundLink B1-B5 GRF failure-mode diagnostics in LOSO."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SUBJECTS = ("s001", "s002", "s003", "s004", "s005", "s006")
TARGET_DIRNAME = "torque_targets_12hz_nomuscles_copfix_tcode"
TIME_TOLERANCE_S = 2e-4
TRANSITION_WINDOW_S = 0.050
LAG_LIMIT_FRAMES = 30
HEIGHT_GRID_M = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
SPEED_GRID_MPS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)


def read_storage(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    try:
        endheader = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as exc:
        raise ValueError(f"Missing endheader in {path}") from exc
    return pd.read_csv(path, sep=r"\s+", skiprows=endheader + 1, engine="python")


def align_indices(source_times: np.ndarray, target_times: np.ndarray, path: Path) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(source_times[left] - target_times) < np.abs(source_times[right] - target_times)
    indices = np.where(choose_left, left, right)
    error = np.abs(source_times[indices] - target_times)
    if np.any(error > TIME_TOLERANCE_S):
        raise ValueError(f"Cannot align {path}; max time difference={error.max():.8g}s")
    return indices


def map_source_path(path: str, source_root: str, torque_root: Path) -> Path:
    normalized, root = os.path.normpath(path), os.path.normpath(source_root)
    if normalized == root:
        return torque_root
    if not normalized.startswith(root + os.sep):
        raise ValueError(f"Path outside source root: {path}")
    return torque_root / os.path.relpath(normalized, root)


def read_contact_wrenches(model_path: Path, pipeline_csv: Path, segment: int, times: np.ndarray) -> pd.DataFrame:
    run_root = model_path.parents[2]
    exact_root = run_root / "results" / "ID_estimatedGRF"
    paths = sorted(exact_root.glob("**/*_contact_wrenches_estimated.csv"))
    if not paths:
        stem = pipeline_csv.stem
        for sibling in sorted(pipeline_csv.parent.glob(stem + "_*")):
            if sibling.is_dir() and sibling != run_root:
                paths.extend(sorted((sibling / "results" / "ID_estimatedGRF").glob("**/*_contact_wrenches_estimated.csv")))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No contact_wrenches_estimated.csv near {run_root}")
    parts = [pd.read_csv(path) for path in paths]
    joined = pd.concat(parts, ignore_index=True).sort_values("time").drop_duplicates("time", keep="first")
    idx = align_indices(joined.time.to_numpy(dtype=float), times, paths[0])
    return joined.iloc[idx].reset_index(drop=True)


def classify_phase(times: np.ndarray, measured_fz: np.ndarray, bw_n: float) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    contact = measured_fz > 0.05 * bw_n
    flight = np.all(measured_fz <= 1.0, axis=1)
    labels = np.full(times.shape, "other", dtype=object)
    count = contact.sum(axis=1)
    labels[count == 1] = "single_stance"
    labels[count == 2] = "double_stance"
    labels[flight] = "flight"
    transitions = np.zeros(times.shape, dtype=bool)
    onsets: dict[str, np.ndarray] = {}
    offsets: dict[str, np.ndarray] = {}
    for side_index, side in enumerate("lr"):
        states = contact[:, side_index]
        edges = np.flatnonzero(states[1:] != states[:-1]) + 1
        starts = list(edges[states[edges] if edges.size else np.zeros(0, dtype=bool)])
        ends = list(edges[~states[edges] if edges.size else np.zeros(0, dtype=bool)])
        if states.size and states[0]:
            starts.insert(0, 0)
        onsets[side] = times[np.asarray(starts, dtype=int)] if starts else np.asarray([], dtype=float)
        offsets[side] = times[np.asarray(ends, dtype=int)] if ends else np.asarray([], dtype=float)
        for edge in list(starts) + list(ends):
            transitions |= np.abs(times - times[edge]) <= TRANSITION_WINDOW_S
    labels[transitions] = "transition"
    return labels, contact, {"onsets_l": onsets["l"], "onsets_r": onsets["r"], "offsets_l": offsets["l"], "offsets_r": offsets["r"]}


def flight_periods(times: np.ndarray, contact: np.ndarray, events: dict[str, np.ndarray], side_index: int) -> np.ndarray:
    side = "l" if side_index == 0 else "r"
    result = np.full(times.shape, "mid_flight", dtype=object)
    is_flight = ~contact[:, side_index]
    onsets = events[f"onsets_{side}"]
    offsets = events[f"offsets_{side}"]
    for i, time in enumerate(times):
        if not is_flight[i]:
            result[i] = "contact"
        elif np.any((onsets >= time) & (onsets - time <= TRANSITION_WINDOW_S)):
            result[i] = "pre_strike_50ms"
        elif np.any((offsets <= time) & (time - offsets <= TRANSITION_WINDOW_S)):
            result[i] = "post_toeoff_50ms"
    return result


def stance_progress(times: np.ndarray, contact: np.ndarray) -> np.ndarray:
    progress = np.full(times.shape, np.nan, dtype=float)
    padded = np.r_[False, contact, False]
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for start, end in zip(changes[0::2], changes[1::2]):
        lo, hi = times[start], times[end - 1]
        if end - start == 1 or hi <= lo:
            progress[start:end] = 0.5
        else:
            progress[start:end] = np.linspace(0.0, 1.0, end - start)
    return progress


def f1_score(truth: np.ndarray, prediction: np.ndarray) -> float:
    tp = np.sum(truth & prediction)
    fp = np.sum(~truth & prediction)
    fn = np.sum(truth & ~prediction)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else math.nan


def regression_fit(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    test_design = np.column_stack([np.ones(len(test_x)), test_x])
    coefficients, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    return test_design @ coefficients


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float, int]:
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[finite], predicted[finite]
    if not len(observed):
        return math.nan, math.nan, math.nan, 0
    error = observed - predicted
    rmse = float(np.sqrt(np.mean(error ** 2)))
    ss_total = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - float(np.sum(error ** 2)) / ss_total if ss_total > 0 else math.nan
    bias = float(np.mean(error))
    return rmse, r2, bias, int(len(observed))


def best_lag(
    reference: np.ndarray,
    comparison: np.ndarray,
    valid: np.ndarray,
    max_lag: int = LAG_LIMIT_FRAMES,
) -> tuple[int, float]:
    best = (0, -math.inf)
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            ref, comp = reference[:lag], comparison[-lag:]
            selected = valid[:lag] & valid[-lag:]
        elif lag > 0:
            ref, comp = reference[lag:], comparison[:-lag]
            selected = valid[lag:] & valid[:-lag]
        else:
            ref, comp = reference, comparison
            selected = valid
        finite = selected & np.isfinite(ref) & np.isfinite(comp)
        ref, comp = ref[finite], comp[finite]
        if len(ref) < 3 or np.std(ref) < 1e-9 or np.std(comp) < 1e-9:
            continue
        corr = float(np.corrcoef(ref, comp)[0, 1])
        if corr > best[1]:
            best = (lag, corr)
    return best[0], best[1] if math.isfinite(best[1]) else math.nan


def read_cache(cache_file: Path, prefix: str) -> dict[str, np.ndarray]:
    with np.load(cache_file) as data:
        return {key[len(prefix):]: data[key] for key in data.files if key.startswith(prefix)}


def summarize_distribution(values: dict[tuple[str, str, str, str], list[tuple[float, float]]]) -> list[dict[str, object]]:
    rows = []
    for (subject, activity, label, period), pairs in sorted(values.items()):
        arr = np.asarray(pairs, dtype=float)
        rows.append({
            "subject": subject, "activity": activity, "label": label, "flight_period": period,
            "n": len(arr),
            "height_median_m": float(np.median(arr[:, 0])),
            "height_p10_m": float(np.quantile(arr[:, 0], 0.10)),
            "height_p90_m": float(np.quantile(arr[:, 0], 0.90)),
            "speed_median_mps": float(np.median(arr[:, 1])),
            "speed_p10_mps": float(np.quantile(arr[:, 1], 0.10)),
            "speed_p90_mps": float(np.quantile(arr[:, 1], 0.90)),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-root", type=Path, required=True)
    parser.add_argument("--source-torque-root", default="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque")
    parser.add_argument("--study-root", type=Path, required=True, help="Study root containing the kinematics cache")
    parser.add_argument("--cache-subdir", default="kinematics_cache", help="Versioned cache folder under phase_b")
    parser.add_argument("--output-dir", type=Path, required=True, help="Compact diagnostics output directory")
    args = parser.parse_args()

    b1_distributions: dict[tuple[str, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    b1_samples: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    b2_samples: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]] = defaultdict(list)
    b3_samples: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]] = defaultdict(list)
    b4_records: list[dict[str, object]] = []
    b5_rows: list[dict[str, object]] = []
    b1_floor_rows: list[dict[str, object]] = []
    b1_roc_rows: list[dict[str, object]] = []

    cache_rows = []
    for subject in SUBJECTS:
        manifest_path = args.study_root / "phase_b" / args.cache_subdir / subject / "cache_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing kinematics cache manifest: {manifest_path}")
        with manifest_path.open(newline="") as handle:
            cache_rows.extend(csv.DictReader(handle))

    for cache_row in cache_rows:
        subject, trial, segment = cache_row["subject"], cache_row["trial"], int(cache_row["segment"])
        target_path = args.torque_root / TARGET_DIRNAME / subject / trial / "torque_targets.csv"
        metadata_path = target_path.parent / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        target_all = pd.read_csv(target_path)
        target = target_all[target_all.segment.astype(int) == segment].copy()
        times = target.time.to_numpy(dtype=float)
        valid = target.valid.to_numpy(dtype=int) == 1
        measured_path = target_path.parent / "segments" / f"segment_{segment}"
        measured_path = next(measured_path.glob("*_grf_measured.mot"))
        estimated_path = target_path.parent / "segments" / f"segment_{segment}" / "estimated_id"
        estimated_path = next(estimated_path.glob("*_grf_estimated.mot"))
        measured_all, estimated_all = read_storage(measured_path), read_storage(estimated_path)
        measured_idx = align_indices(measured_all.time.to_numpy(dtype=float), times, measured_path)
        estimated_idx = align_indices(estimated_all.time.to_numpy(dtype=float), times, estimated_path)
        measured = measured_all.iloc[measured_idx].reset_index(drop=True)
        estimated = estimated_all.iloc[estimated_idx].reset_index(drop=True)

        pipeline_path = map_source_path(metadata["source_paths"]["pipeline_csv"], args.source_torque_root, args.torque_root)
        pipeline = pd.read_csv(pipeline_path)
        pipeline_idx = align_indices(pipeline.time.to_numpy(dtype=float), times, pipeline_path)
        pipeline = pipeline.iloc[pipeline_idx].reset_index(drop=True)
        mass = float(pipeline.subject_mass_kg.iloc[0])
        bw_n = mass * 9.81
        measured_fz = np.column_stack([measured[f"ground_force_calcn_{side}_vz"].to_numpy(dtype=float) for side in "lr"])
        estimated_fz = np.column_stack([
            estimated[f"ground_force_calcn_{side}_vz"].to_numpy(dtype=float)
            + estimated[f"ground_force_toes_{side}_vz"].to_numpy(dtype=float)
            for side in "lr"
        ])
        measured_force = np.stack([
            np.column_stack([measured[f"ground_force_calcn_{side}_v{axis}"].to_numpy(dtype=float) for axis in "xyz"])
            for side in "lr"
        ], axis=1)
        estimated_force = np.stack([
            np.column_stack([
                estimated[f"ground_force_calcn_{side}_v{axis}"].to_numpy(dtype=float)
                + estimated[f"ground_force_toes_{side}_v{axis}"].to_numpy(dtype=float)
                for axis in "xyz"
            ])
            for side in "lr"
        ], axis=1)
        phase, contact, events = classify_phase(times, measured_fz, bw_n)
        activity = trial.split("_")[0]
        cache = read_cache(Path(cache_row["cache_file"]), cache_row["cache_prefix"])
        cache_times = cache["times"].astype(float)
        cache_idx = align_indices(cache_times, times, Path(cache_row["cache_file"]))
        com = cache["com_positions"][cache_idx]
        body = cache["body_positions"][cache_idx]
        body_names = [str(name) for name in cache["body_names"].tolist()]
        body_index = {name: index for index, name in enumerate(body_names)}
        det_height = np.column_stack([cache[f"det_low_height_{side}"][cache_idx] for side in "lr"])
        det_speed = np.column_stack([cache[f"det_speed_{side}"][cache_idx] for side in "lr"])
        floor_height = float(cache["floor_height"][0])
        model_path = Path(metadata["source_paths"]["model"])
        model_path = map_source_path(str(model_path), args.source_torque_root, args.torque_root)
        cw = read_contact_wrenches(model_path, pipeline_path, segment, times)
        cw_target = cw[[f"target_force_{axis}" for axis in "xyz"]].to_numpy(dtype=float)

        # B1: contact point geometry, false-contact labels, floor estimate, and LOSO threshold samples.
        for side_index, side in enumerate("lr"):
            true_contact = contact[:, side_index]
            flight = measured_fz[:, side_index] <= 1.0
            false_contact = flight & (estimated_fz[:, side_index] > 20.0)
            no_estimated_contact = flight & ~false_contact
            period = flight_periods(times, contact, events, side_index)
            labels = np.full(times.shape, "other", dtype=object)
            labels[true_contact] = "true_contact"
            labels[false_contact] = "false_contact"
            labels[no_estimated_contact] = "true_flight_no_estimated_contact"
            for frame in np.flatnonzero(valid & (labels != "other")):
                this_period = str(period[frame]) if flight[frame] else "contact"
                b1_distributions[(subject, activity, str(labels[frame]), this_period)].append(
                    (float(det_height[frame, side_index]), float(det_speed[frame, side_index]))
                )
            usable = valid & (true_contact | flight)
            b1_samples[subject].append((
                det_height[usable, side_index], det_speed[usable, side_index], true_contact[usable]
            ))
            foot_indices = [body_index[f"calcn_{side}"], body_index[f"toes_{side}"]]
            true_height = np.min(cache["body_heights_above_floor"][cache_idx][:, foot_indices], axis=1)
            contact_point_z = np.min(body[:, foot_indices, 2], axis=1)
            selected = valid & true_contact
            b1_floor_rows.append({
                "subject": subject, "trial": trial, "segment": segment, "side": side,
                "activity": activity, "floor_z_m": floor_height,
                "median_contact_point_z_m": float(np.median(contact_point_z[selected])) if selected.any() else math.nan,
                "floor_error_vs_measured_contact_m": float(floor_height - np.median(contact_point_z[selected])) if selected.any() else math.nan,
                "median_body_height_above_floor_during_contact_m": float(np.median(true_height[selected])) if selected.any() else math.nan,
                "false_contact_rate_in_flight": float(np.mean(false_contact[valid & flight])) if np.any(valid & flight) else math.nan,
                "n_contact_frames": int(selected.sum()), "n_flight_frames": int((valid & flight).sum()),
            })

        # B2: vertical-force split in double stance.
        double = valid & contact[:, 0] & contact[:, 1]
        if double.any():
            lfoot = body[:, body_index["calcn_l"], :2]
            rfoot = body[:, body_index["calcn_r"], :2]
            com_xy = com[:, :2]
            line = rfoot - lfoot
            length2 = np.sum(line * line, axis=1)
            lambda_com = np.divide(np.sum((com_xy - lfoot) * line, axis=1), length2,
                                   out=np.full(len(times), 0.5), where=length2 > 1e-8)
            dl = np.linalg.norm(com_xy - lfoot, axis=1)
            dr = np.linalg.norm(com_xy - rfoot, axis=1)
            inverse_share = np.divide(dr, dl + dr, out=np.full(len(times), 0.5), where=(dl + dr) > 1e-8)
            lever_share = np.clip(1.0 - lambda_com, 0.0, 1.0)
            measured_share = measured_fz[:, 0] / np.maximum(measured_fz.sum(axis=1), 1e-8)
            estimated_contact = np.column_stack([
                ((pipeline.get(f"calcn_{side}_contact", pd.Series(np.zeros(len(times)))).to_numpy(dtype=float) > 0.05)
                 | (pipeline.get(f"toes_{side}_contact", pd.Series(np.zeros(len(times)))).to_numpy(dtype=float) > 0.05))
                for side in "lr"
            ])
            elapsed = np.zeros((len(times), 2), dtype=float)
            for side_idx, side in enumerate("lr"):
                state = estimated_contact[:, side_idx]
                event_frames = np.r_[0, np.flatnonzero(state[1:] != state[:-1]) + 1]
                latest = event_frames[np.searchsorted(event_frames, np.arange(len(times)), side="right") - 1]
                elapsed[:, side_idx] = times - times[latest]
            b2_samples[subject].append((
                lambda_com[double], elapsed[double, 0], elapsed[double, 1], measured_share[double],
                inverse_share[double], lever_share[double], activity,
            ))

        # B3 H1: diagnostic CoM-directed force, balanced to the measured net horizontal force.
        foot_xy = np.stack([
            body[:, body_index["calcn_l"], :2], body[:, body_index["calcn_r"], :2]
        ], axis=1)
        h = np.maximum(com[:, 2] - floor_height, 0.10)
        direction = (com[:, None, :2] - foot_xy) / h[:, None, None]
        raw_h = measured_fz[:, :, None] * direction
        net_h = measured_force[:, :, :2].sum(axis=1)
        raw_total = raw_h.sum(axis=1)
        positive_fz = np.maximum(measured_fz, 0.0)
        positive_total = positive_fz.sum(axis=1, keepdims=True)
        share = np.divide(
            positive_fz,
            positive_total,
            out=np.full_like(positive_fz, 0.5),
            where=positive_total > 1e-8,
        )
        predicted_h = raw_h + share[:, :, None] * (net_h - raw_total)[:, None, :]

        # B3 H2: use the measured net force and measured contact phase only to isolate
        # the foot-to-foot split; the moment target comes from segment kinematics.
        # This is a diagnostic least-squares model, not an inference variant.
        predicted_h2 = np.zeros_like(measured_force)
        net_force = measured_force.sum(axis=1)
        left_point = 0.5 * (body[:, body_index["calcn_l"], :] + body[:, body_index["toes_l"], :])
        right_point = 0.5 * (body[:, body_index["calcn_r"], :] + body[:, body_index["toes_r"], :])
        left_only = contact[:, 0] & ~contact[:, 1]
        right_only = contact[:, 1] & ~contact[:, 0]
        predicted_h2[left_only, 0] = net_force[left_only]
        predicted_h2[right_only, 1] = net_force[right_only]
        double_idx = np.flatnonzero(contact[:, 0] & contact[:, 1])
        if double_idx.size:
            arms_l = left_point[double_idx] - com[double_idx]
            arms_r = right_point[double_idx] - com[double_idx]
            def skew_batch(vectors: np.ndarray) -> np.ndarray:
                out = np.zeros((len(vectors), 3, 3), dtype=float)
                out[:, 0, 1], out[:, 0, 2] = -vectors[:, 2], vectors[:, 1]
                out[:, 1, 0], out[:, 1, 2] = vectors[:, 2], -vectors[:, 0]
                out[:, 2, 0], out[:, 2, 1] = -vectors[:, 1], vectors[:, 0]
                return out
            matrices = np.zeros((len(double_idx), 6, 6), dtype=float)
            identity = np.eye(3)
            matrices[:, :3, :3] = identity
            matrices[:, :3, 3:] = identity
            matrices[:, 3:, :3] = skew_batch(arms_l)
            matrices[:, 3:, 3:] = skew_batch(arms_r)
            rhs = np.column_stack([
                net_force[double_idx],
                cache["centroidal_angular_momentum_rate"][cache_idx][double_idx],
            ])
            solutions = np.einsum("nij,nj->ni", np.linalg.pinv(matrices), rhs)
            predicted_h2[double_idx] = solutions.reshape(-1, 2, 3)
        b3_samples[subject].append((
            measured_force[:, :, :2][valid], predicted_h[valid], predicted_h2[:, :, :2][valid],
            phase[valid], np.full(int(valid.sum()), bw_n, dtype=float), activity,
        ))

        # B4: CoP progression along the heel-to-toe axis, inclination, and estimated contact activations.
        for side_index, side in enumerate("lr"):
            heel = body[:, body_index[f"calcn_{side}"], :]
            toes = body[:, body_index[f"toes_{side}"], :]
            axis = toes[:, :2] - heel[:, :2]
            length = np.linalg.norm(axis, axis=1)
            cop = np.column_stack([
                measured[f"ground_force_calcn_{side}_px"].to_numpy(dtype=float),
                measured[f"ground_force_calcn_{side}_py"].to_numpy(dtype=float),
            ])
            progress = np.divide(np.sum((cop - heel[:, :2]) * axis, axis=1), length ** 2,
                                 out=np.full(len(times), np.nan), where=length > 1e-4)
            angle = np.arctan2(toes[:, 2] - heel[:, 2], np.maximum(length, 1e-4))
            calcn_contact = pipeline[f"calcn_{side}_contact"].to_numpy(dtype=float) if f"calcn_{side}_contact" in pipeline else np.zeros(len(times))
            toes_contact = pipeline[f"toes_{side}_contact"].to_numpy(dtype=float) if f"toes_{side}_contact" in pipeline else np.zeros(len(times))
            estimated_stance = (calcn_contact > 0.05) | (toes_contact > 0.05)
            stance = np.nan_to_num(stance_progress(times, estimated_stance), nan=0.0)
            selected = valid & contact[:, side_index] & np.isfinite(progress)
            for frame in np.flatnonzero(selected):
                b4_records.append({
                    "subject": subject, "trial": trial, "segment": segment, "activity": activity,
                    "side": side, "time": float(times[frame]), "progress_measured": float(progress[frame]),
                    "time_progress": float(stance[frame]), "incline_rad": float(angle[frame]),
                    "foot_length_m": float(length[frame]),
                    "calcn_contact_estimated": float(calcn_contact[frame]),
                    "toes_contact_estimated": float(toes_contact[frame]),
                })

        # B5: lag is reported without shifting any signal.
        measured_total = measured_force.sum(axis=1)
        estimated_total = estimated_force.sum(axis=1)
        frame_ms = float(np.median(np.diff(times)) * 1000.0) if len(times) > 1 else math.nan
        phase_names = ("total", "single_stance", "double_stance", "flight", "transition", "other")
        for phase_name in phase_names:
            phase_valid = valid if phase_name == "total" else valid & (phase == phase_name)
            if phase_valid.sum() < 3:
                continue
            for axis_index, axis in enumerate("xyz"):
                comparisons = [
                    ("measured_vs_m_a_minus_g", cw_target[:, axis_index], measured_total[:, axis_index]),
                    ("estimated_vs_m_a_minus_g", cw_target[:, axis_index], estimated_total[:, axis_index]),
                ]
                comparisons.extend(
                    (f"measured_vs_estimated_foot_{side}", measured_force[:, side_index, axis_index],
                     estimated_force[:, side_index, axis_index])
                    for side_index, side in enumerate("lr")
                )
                for name, reference, comparison in comparisons:
                    lag, corr = best_lag(reference, comparison, phase_valid)
                    b5_rows.append({
                        "subject": subject, "trial": trial, "segment": segment, "activity": activity,
                        "phase": phase_name, "comparison": name, "axis": axis,
                        "lag_frames_reference_later": lag,
                        "lag_ms_reference_later": lag * frame_ms,
                        "correlation_at_best_lag": corr,
                    })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "B1_distribution.csv", summarize_distribution(b1_distributions))
    write_csv(args.output_dir / "B1_floor_by_trial.csv", b1_floor_rows)
    write_csv(args.output_dir / "B5_lag.csv", b5_rows)

    # B1 threshold grid selected on training subjects and scored once on each held-out subject.
    b1_loso = []
    loso_params: list[dict[str, object]] = []
    grid = [("height", h, math.nan) for h in HEIGHT_GRID_M]
    grid += [("speed", math.nan, s) for s in SPEED_GRID_MPS]
    grid += [("height_and_speed", h, s) for h in HEIGHT_GRID_M for s in SPEED_GRID_MPS]
    for heldout in SUBJECTS:
        train = [s for s in SUBJECTS if s != heldout]
        train_scores = []
        for method, height_cut, speed_cut in grid:
            subject_scores = []
            for subject in train:
                if subject not in b1_samples:
                    continue
                hs, ss, ys = [np.concatenate([chunk[i] for chunk in b1_samples[subject]]) for i in range(3)]
                pred = hs <= height_cut if method == "height" else ss <= speed_cut if method == "speed" else (hs <= height_cut) & (ss <= speed_cut)
                subject_scores.append(f1_score(ys.astype(bool), pred))
            score = float(np.nanmean(subject_scores)) if subject_scores else -math.inf
            train_scores.append((score, method, height_cut, speed_cut))
        _, method, height_cut, speed_cut = max(train_scores, key=lambda item: (item[0], item[1], item[2], item[3]))
        hs, ss, ys = [np.concatenate([chunk[i] for chunk in b1_samples[heldout]]) for i in range(3)]
        for roc_method, roc_height, roc_speed in grid:
            roc_pred = hs <= roc_height if roc_method == "height" else ss <= roc_speed if roc_method == "speed" else (hs <= roc_height) & (ss <= roc_speed)
            positives, negatives = ys.astype(bool), ~ys.astype(bool)
            b1_roc_rows.append({
                "heldout_subject": heldout, "method": roc_method,
                "height_threshold_m": roc_height, "speed_threshold_mps": roc_speed,
                "true_positive_rate": float(np.mean(roc_pred[positives])) if positives.any() else math.nan,
                "false_positive_rate": float(np.mean(roc_pred[negatives])) if negatives.any() else math.nan,
                "precision": float(np.mean(positives[roc_pred])) if roc_pred.any() else math.nan,
                "n_contact_frames": int(positives.sum()), "n_flight_frames": int(negatives.sum()),
            })
        pred = hs <= height_cut if method == "height" else ss <= speed_cut if method == "speed" else (hs <= height_cut) & (ss <= speed_cut)
        b1_loso.append({
            "heldout_subject": heldout, "selected_method": method,
            "height_threshold_m": height_cut, "speed_threshold_mps": speed_cut,
            "test_contact_f1": f1_score(ys.astype(bool), pred),
            "train_macro_f1": max(train_scores)[0], "n_test_frames": len(ys),
        })
        for parameter, value in (("selected_method", method), ("height_threshold_m", height_cut),
                                 ("speed_threshold_mps", speed_cut)):
            if isinstance(value, str) or math.isfinite(value):
                loso_params.append({"heldout_subject": heldout, "diagnostic": "B1",
                                    "parameter": parameter, "value": value,
                                    "selected_method": method})
    write_csv(args.output_dir / "B1_threshold_loso.csv", b1_loso)
    write_csv(args.output_dir / "B1_roc.csv", b1_roc_rows)

    # B2 LOSO: inverse-distance and lever baselines; a fitted lever+transition regression uses train subjects only.
    b2_loso = []
    b2_summary = []
    b2_by_subject: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for subject, chunks in b2_samples.items():
        values = [np.concatenate([chunk[i] for chunk in chunks]) for i in range(6)]
        b2_by_subject[subject] = tuple(values)
        b2_summary.append({"subject": subject, "n_double_stance_frames": len(values[0]),
                           "inverse_distance_rmse_share": regression_metrics(values[3], values[4])[0],
                           "lever_rmse_share": regression_metrics(values[3], values[5])[0]})
    for heldout in SUBJECTS:
        if heldout not in b2_by_subject:
            continue
        lam, tl, tr, observed, inverse, lever = b2_by_subject[heldout]
        features = np.column_stack([lam, tl, tr])
        train_features, train_values = [], []
        for subject in SUBJECTS:
            if subject == heldout or subject not in b2_by_subject:
                continue
            l, x, y, truth, _, _ = b2_by_subject[subject]
            train_features.append(np.column_stack([l, x, y]))
            train_values.append(truth)
        if train_features:
            train_x = np.concatenate(train_features)
            train_y = np.concatenate(train_values)
            coefficients, *_ = np.linalg.lstsq(
                np.column_stack([np.ones(len(train_x)), train_x]), train_y, rcond=None
            )
            fitted = np.column_stack([np.ones(len(features)), features]) @ coefficients
            for parameter, value in zip(
                ("intercept", "com_share", "time_since_left_strike_s", "time_since_right_strike_s"),
                coefficients,
            ):
                loso_params.append({"heldout_subject": heldout, "diagnostic": "B2",
                                    "parameter": parameter, "value": float(value),
                                    "selected_method": "lever_plus_transition_loso"})
        else:
            fitted = np.full(len(observed), math.nan)
        for model_name, pred in (("inverse_distance", inverse), ("lever", lever), ("lever_plus_transition_loso", fitted)):
            rmse, r2, bias, n = regression_metrics(observed, pred)
            b2_loso.append({"heldout_subject": heldout, "model": model_name, "rmse_left_share": rmse,
                            "r2": r2, "bias": bias, "n_double_stance_frames": n})
    write_csv(args.output_dir / "B2_double_stance_summary.csv", b2_summary)
    write_csv(args.output_dir / "B2_loso.csv", b2_loso)
    write_csv(args.output_dir / "loso_params.csv", loso_params)

    # B3 H1: measured Fz and CoM geometry define a diagnostic predictor; net measured Fx/Fy is only used to balance it.
    b3_rows = []
    b3_opposition_rows = []
    for subject, chunks in b3_samples.items():
        measured = np.concatenate([chunk[0] for chunk in chunks], axis=0)
        predicted_h1 = np.concatenate([chunk[1] for chunk in chunks], axis=0)
        predicted_h2 = np.concatenate([chunk[2] for chunk in chunks], axis=0)
        phases = np.concatenate([chunk[3] for chunk in chunks])
        bw_n = np.concatenate([chunk[4] for chunk in chunks])
        for model_name, predicted in (("H1_CoM_direction", predicted_h1), ("H2_angular_momentum_lstsq", predicted_h2)):
            double = phases == "double_stance"
            if double.any():
                truth_pair = measured[double]
                estimate_pair = predicted[double]
                true_opp = np.maximum(
                    0.0,
                    0.5 * (np.linalg.norm(truth_pair[:, 0], axis=1)
                           + np.linalg.norm(truth_pair[:, 1], axis=1)
                           - np.linalg.norm(truth_pair.sum(axis=1), axis=1)),
                )
                pred_opp = np.maximum(
                    0.0,
                    0.5 * (np.linalg.norm(estimate_pair[:, 0], axis=1)
                           + np.linalg.norm(estimate_pair[:, 1], axis=1)
                           - np.linalg.norm(estimate_pair.sum(axis=1), axis=1)),
                )
                true_opposed = np.sum(truth_pair[:, 0] * truth_pair[:, 1], axis=1) < 0
                pred_opposed = np.sum(estimate_pair[:, 0] * estimate_pair[:, 1], axis=1) < 0
                b3_opposition_rows.append({
                    "subject": subject, "model": model_name, "n_double_stance_frames": int(double.sum()),
                    "measured_opposing_frames": int(true_opposed.sum()),
                    "predicted_opposing_frames": int(pred_opposed.sum()),
                    "measured_opposing_force_n_sum": float(true_opp.sum()),
                    "predicted_opposing_force_n_sum": float(pred_opp.sum()),
                    "opposing_force_recovery_ratio": float(pred_opp.sum() / true_opp.sum()) if true_opp.sum() > 1e-8 else math.nan,
                    "opposed_frame_recall": float((pred_opposed & true_opposed).sum() / true_opposed.sum()) if true_opposed.any() else math.nan,
                })
        for phase_name in ("total", "single_stance", "double_stance", "flight", "transition", "other"):
            phase_mask = np.ones(len(phases), dtype=bool) if phase_name == "total" else phases == phase_name
            for model_name, predicted in (("H1_CoM_direction", predicted_h1), ("H2_angular_momentum_lstsq", predicted_h2)):
                for side_index, side in enumerate("lr"):
                    truth = measured[phase_mask, side_index, :]
                    estimate = predicted[phase_mask, side_index, :]
                    if not len(truth):
                        continue
                    error = truth - estimate
                    rmse = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
                    rmse_bw = float(np.sqrt(np.mean(np.sum((error / bw_n[phase_mask, None]) ** 2, axis=1))))
                    ss_total = float(np.sum((truth - truth.mean(axis=0)) ** 2))
                    r2 = 1.0 - float(np.sum((truth - estimate) ** 2)) / ss_total if ss_total > 0 else math.nan
                    truth_angle = np.arctan2(truth[:, 1], truth[:, 0])
                    estimate_angle = np.arctan2(estimate[:, 1], estimate[:, 0])
                    angle_error = np.abs(np.arctan2(np.sin(truth_angle - estimate_angle), np.cos(truth_angle - estimate_angle)))
                    b3_rows.append({"subject": subject, "model": model_name, "phase": phase_name, "foot": side, "rmse_n": rmse,
                                    "rmse_bw": rmse_bw, "r2": r2,
                                    "mean_angular_error_deg": float(np.degrees(np.mean(angle_error))), "n": len(truth)})
    write_csv(args.output_dir / "B3_horizontal_h1.csv", b3_rows)
    write_csv(args.output_dir / "B3_opposing_force_recovery.csv", b3_opposition_rows)

    # B4 LOSO models: midpoint, linear progression, inclination, and contact activation.
    b4_loso = []
    b4_frame_path = args.study_root / "phase_b" / "B4_cop_frames.csv"
    write_csv(b4_frame_path, b4_records)
    b4 = pd.DataFrame(b4_records)
    for heldout in SUBJECTS:
        test = b4[b4.subject == heldout]
        train = b4[b4.subject != heldout]
        if test.empty or train.empty:
            continue
        y_train = train.progress_measured.to_numpy(dtype=float)
        y_test = test.progress_measured.to_numpy(dtype=float)
        specs = {
            "midpoint": (None, None),
            "linear_time": (np.column_stack([train.time_progress]), np.column_stack([test.time_progress])),
            "inclination": (np.column_stack([train.incline_rad]), np.column_stack([test.incline_rad])),
            "activation_and_kinematics": (
                train[["time_progress", "incline_rad", "calcn_contact_estimated", "toes_contact_estimated"]].to_numpy(dtype=float),
                test[["time_progress", "incline_rad", "calcn_contact_estimated", "toes_contact_estimated"]].to_numpy(dtype=float),
            ),
        }
        for name, (train_x, test_x) in specs.items():
            prediction = np.full(len(test), 0.5) if train_x is None else regression_fit(train_x, y_train, test_x)
            error_cm = np.abs(prediction - y_test) * test.foot_length_m.to_numpy(dtype=float) * 100.0
            b4_loso.append({"heldout_subject": heldout, "model": name,
                            "mean_abs_cop_error_cm": float(np.mean(error_cm)),
                            "median_abs_cop_error_cm": float(np.median(error_cm)),
                            "n_contact_frames": len(test)})
    write_csv(args.output_dir / "B4_cop_loso.csv", b4_loso)
    print(f"B1={len(b1_loso)} B2={len(b2_loso)} B3={len(b3_rows)} B4={len(b4_loso)} B5={len(b5_rows)}")
    return 0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

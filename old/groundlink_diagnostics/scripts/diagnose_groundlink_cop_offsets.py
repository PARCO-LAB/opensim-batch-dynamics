#!/usr/bin/env python3
"""Measure GroundLink measured-CoP offsets against the saved estimated feet.

This is a read-only Phase A diagnostic. It reads the existing
``torque_targets_12hz_nomuscles`` tree plus the original GroundLink MoSh++
translations and writes new CSV reports to a separate output directory.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
from pathlib import Path
from typing import Any

import numpy as np


SUBJECTS = ("s001", "s002", "s003", "s004", "s005", "s006")
DETAILED_FIELDS = [
    "subject", "trial", "segment", "side", "stance_frames_fz_gt_100_n",
    "invalid_cop_frames", "median_cop_to_own_foot_cm", "p90_cop_to_own_foot_cm",
    "median_cop_to_other_foot_cm", "p90_cop_to_other_foot_cm",
    "own_foot_closer_percent", "median_offset_x_m", "iqr_offset_x_m",
    "median_offset_y_m", "iqr_offset_y_m",
    "median_cop_x_m", "median_cop_y_m",
    "median_own_foot_midpoint_x_m", "median_own_foot_midpoint_y_m",
    "candidate_cop_to_own_foot_cm", "candidate_cop_to_other_foot_cm",
    "candidate_own_foot_closer_percent",
    "frame_transform_cop_to_own_foot_cm", "frame_transform_cop_to_other_foot_cm",
    "frame_transform_own_foot_closer_percent",
    "root_transform_cop_to_own_foot_cm", "root_transform_cop_to_other_foot_cm",
    "root_transform_own_foot_closer_percent",
    "raw_moshpp_gender", "smplx_pelvis_offset_x_m", "smplx_pelvis_offset_y_m",
    "smplx_pelvis_offset_z_m", "pipeline_minus_raw_pelvis_x_m",
    "pipeline_minus_raw_pelvis_y_m", "pipeline_minus_raw_pelvis_z_m",
    "pipeline_minus_raw_pelvis_y_iqr_m",
    "raw_moshpp_trans_x_start_m", "raw_moshpp_trans_y_start_m",
    "raw_moshpp_trans_z_start_m", "raw_moshpp_trans_x_mean_m",
    "raw_moshpp_trans_y_mean_m", "raw_moshpp_trans_z_mean_m",
    "pipeline_pelvis_tx_start_m", "pipeline_pelvis_ty_start_m",
    "pipeline_pelvis_tz_start_m", "pipeline_pelvis_tx_mean_m",
    "pipeline_pelvis_ty_mean_m", "pipeline_pelvis_tz_mean_m",
]
TRIAL_FIELDS = [
    "subject", "trial", "segments", "stance_frames_fz_gt_100_n",
    "invalid_cop_frames", "valid_cop_frames", "median_offset_x_m", "median_offset_y_m",
    "median_abs_offset_x_m", "median_abs_offset_y_m",
    "median_cop_x_m", "median_cop_y_m",
    "median_foot_midpoint_x_m", "median_foot_midpoint_y_m",
    "median_cop_to_own_foot_cm", "p90_cop_to_own_foot_cm",
    "median_cop_to_other_foot_cm", "own_foot_closer_percent",
    "candidate_cop_shift_x_m", "candidate_cop_shift_y_m",
    "candidate_median_cop_to_own_foot_cm", "candidate_p90_cop_to_own_foot_cm",
    "candidate_median_cop_to_other_foot_cm", "candidate_own_foot_closer_percent",
    "frame_transform_median_offset_x_m", "frame_transform_median_offset_y_m",
    "frame_transform_median_cop_to_own_foot_cm", "frame_transform_p90_cop_to_own_foot_cm",
    "frame_transform_median_cop_to_other_foot_cm", "frame_transform_own_foot_closer_percent",
    "root_transform_shift_x_m", "root_transform_shift_y_m",
    "root_transform_residual_offset_y_m",
    "root_transform_median_cop_to_own_foot_cm", "root_transform_p90_cop_to_own_foot_cm",
    "root_transform_median_cop_to_other_foot_cm", "root_transform_own_foot_closer_percent",
    "root_transform_own_closer_over_95_percent",
    "raw_moshpp_gender", "smplx_pelvis_offset_x_m", "smplx_pelvis_offset_y_m",
    "smplx_pelvis_offset_z_m", "pipeline_minus_raw_pelvis_x_m",
    "pipeline_minus_raw_pelvis_y_m", "pipeline_minus_raw_pelvis_z_m",
    "pipeline_minus_raw_pelvis_y_iqr_m",
    "raw_moshpp_trans_x_start_m", "raw_moshpp_trans_y_start_m",
    "raw_moshpp_trans_z_start_m", "raw_moshpp_trans_x_mean_m",
    "raw_moshpp_trans_y_mean_m", "raw_moshpp_trans_z_mean_m",
    "pipeline_pelvis_tx_start_m", "pipeline_pelvis_ty_start_m",
    "pipeline_pelvis_tz_start_m", "pipeline_pelvis_tx_mean_m",
    "pipeline_pelvis_ty_mean_m", "pipeline_pelvis_tz_mean_m",
    "dy_minus_raw_trans_y_start_m", "dy_plus_raw_trans_y_start_m",
    "dy_minus_raw_trans_y_mean_m", "dy_plus_raw_trans_y_mean_m",
    "dy_minus_pipeline_ty_start_m", "dy_plus_pipeline_ty_start_m",
    "dy_minus_pipeline_ty_mean_m", "dy_plus_pipeline_ty_mean_m",
    "offset_plus_root_transform_y_m",
]


def _read_mot(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    endheader = next((i for i, line in enumerate(lines) if line.strip().lower() == "endheader"), None)
    if endheader is None or endheader + 1 >= len(lines):
        raise ValueError(f"Invalid OpenSim MOT header: {path}")
    columns = lines[endheader + 1].split()
    values = np.loadtxt(path, skiprows=endheader + 2, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.shape[1] != len(columns):
        raise ValueError(f"Column/data mismatch in {path}: {len(columns)} names, {values.shape[1]} values")
    return columns, values


def _column(columns: list[str], values: np.ndarray, name: str) -> np.ndarray:
    try:
        index = columns.index(name)
    except ValueError as exc:
        raise ValueError(f"Missing {name!r} in MOT columns") from exc
    return values[:, index]


def _point_to_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = b - a
    denominator = np.einsum("ij,ij->i", direction, direction)
    numerator = np.einsum("ij,ij->i", points - a, direction)
    fraction = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    closest = a + np.clip(fraction, 0.0, 1.0)[:, None] * direction
    return np.linalg.norm(points - closest, axis=1)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _summary(values: list[float] | np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return float("nan"), float("nan")
    return float(np.median(data)), float(np.percentile(data, 75) - np.percentile(data, 25))


@functools.lru_cache(maxsize=3)
def _smplx_pelvis_offset(model_path: str) -> tuple[float, float, float]:
    with np.load(model_path, allow_pickle=True) as model:
        root_joint = np.asarray(model["J"][0], dtype=float)
    return float(root_joint[0]), float(root_joint[1]), float(root_joint[2])


def _trial_roots(metadata: dict[str, Any], dataset_root: Path, model_dir: Path) -> dict[str, Any]:
    source_paths = metadata["source_paths"]
    measured_path = Path(source_paths["measured_force"])
    raw_motion_path = dataset_root / "moshpp" / metadata["subject"] / f"{measured_path.stem}_stageii.npz"
    with np.load(raw_motion_path, allow_pickle=True) as motion:
        translation = np.asarray(motion["trans"], dtype=float)
        gender = str(motion["gender"]).strip().upper()
    if translation.ndim != 2 or translation.shape[1] < 3 or not len(translation):
        raise ValueError(f"Invalid raw MoSh++ translation in {raw_motion_path}: {translation.shape}")
    if gender not in {"FEMALE", "MALE", "NEUTRAL"}:
        gender = "NEUTRAL"
    pelvis_offset = _smplx_pelvis_offset(str(model_dir / f"SMPLX_{gender}.npz"))

    pipeline_rows = _read_csv_rows(Path(source_paths["pipeline_csv"]))
    if not pipeline_rows:
        raise ValueError(f"No rows in pipeline CSV: {source_paths['pipeline_csv']}")
    pipeline_tx = np.asarray([_number(row.get("pelvis_tx")) for row in pipeline_rows])
    pipeline_ty = np.asarray([_number(row.get("pelvis_ty")) for row in pipeline_rows])
    pipeline_tz = np.asarray([_number(row.get("pelvis_tz")) for row in pipeline_rows])
    pipeline_time = np.asarray([_number(row.get("time")) for row in pipeline_rows])
    raw_time = np.arange(len(translation), dtype=float) / 250.0
    raw_pelvis = np.column_stack([
        np.interp(pipeline_time, raw_time, translation[:, axis] + pelvis_offset[axis])
        for axis in range(3)
    ])
    pipeline_pelvis = np.column_stack((pipeline_tx, pipeline_ty, pipeline_tz))
    frame_shift = pipeline_pelvis - raw_pelvis
    return {
        "raw_moshpp_gender": gender,
        "smplx_pelvis_offset_x_m": pelvis_offset[0],
        "smplx_pelvis_offset_y_m": pelvis_offset[1],
        "smplx_pelvis_offset_z_m": pelvis_offset[2],
        "raw_moshpp_trans_x_start_m": float(translation[0, 0]),
        "raw_moshpp_trans_y_start_m": float(translation[0, 1]),
        "raw_moshpp_trans_z_start_m": float(translation[0, 2]),
        "raw_moshpp_trans_x_mean_m": float(np.mean(translation[:, 0])),
        "raw_moshpp_trans_y_mean_m": float(np.mean(translation[:, 1])),
        "raw_moshpp_trans_z_mean_m": float(np.mean(translation[:, 2])),
        "pipeline_pelvis_tx_start_m": float(pipeline_tx[0]),
        "pipeline_pelvis_ty_start_m": float(pipeline_ty[0]),
        "pipeline_pelvis_tz_start_m": float(pipeline_tz[0]),
        "pipeline_pelvis_tx_mean_m": float(np.nanmean(pipeline_tx)),
        "pipeline_pelvis_ty_mean_m": float(np.nanmean(pipeline_ty)),
        "pipeline_pelvis_tz_mean_m": float(np.nanmean(pipeline_tz)),
        "pipeline_minus_raw_pelvis_x_m": float(np.median(frame_shift[:, 0])),
        "pipeline_minus_raw_pelvis_y_m": float(np.median(frame_shift[:, 1])),
        "pipeline_minus_raw_pelvis_z_m": float(np.median(frame_shift[:, 2])),
        "pipeline_minus_raw_pelvis_x_start_m": float(frame_shift[0, 0]),
        "pipeline_minus_raw_pelvis_y_start_m": float(frame_shift[0, 1]),
        "pipeline_minus_raw_pelvis_y_mean_m": float(np.mean(frame_shift[:, 1])),
        "pipeline_minus_raw_pelvis_y_iqr_m": float(np.percentile(frame_shift[:, 1], 75) - np.percentile(frame_shift[:, 1], 25)),
        "_pipeline_time": pipeline_time,
        "_frame_shift": frame_shift,
    }


def _analyze_trial(
    source_root: Path,
    dataset_root: Path,
    model_dir: Path,
    subject: str,
    trial: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trial_root = source_root / subject / trial
    metadata = json.loads((trial_root / "run_metadata.json").read_text(encoding="utf-8"))
    roots = _trial_roots(metadata, dataset_root, model_dir)
    detailed: list[dict[str, Any]] = []
    all_dx: list[np.ndarray] = []
    all_dy: list[np.ndarray] = []
    all_cop: list[np.ndarray] = []
    all_midpoint: list[np.ndarray] = []
    all_own_distance: list[np.ndarray] = []
    all_other_distance: list[np.ndarray] = []
    all_own_closer: list[np.ndarray] = []
    sample_groups: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    all_n_stance = 0
    all_n_invalid = 0
    segment_dirs = sorted((trial_root / "segments").glob("segment_*"))
    if not segment_dirs:
        raise FileNotFoundError(f"No saved ID segments under {trial_root}")

    for segment_dir in segment_dirs:
        segment_name = segment_dir.name[len("segment_"):]
        measured_files = sorted(segment_dir.glob("*_grf_measured.mot"))
        estimated_files = sorted((segment_dir / "estimated_id").glob("*_grf_estimated.mot"))
        if len(measured_files) != 1 or len(estimated_files) != 1:
            raise FileNotFoundError(
                f"Expected one measured and estimated MOT in {segment_dir}; "
                f"found {len(measured_files)} and {len(estimated_files)}"
            )
        measured_columns, measured = _read_mot(measured_files[0])
        estimated_columns, estimated = _read_mot(estimated_files[0])
        measured_time = _column(measured_columns, measured, "time")
        estimated_time = _column(estimated_columns, estimated, "time")
        if np.any(measured_time < estimated_time[0] - 1e-6) or np.any(measured_time > estimated_time[-1] + 1e-6):
            raise ValueError(f"MOT time ranges do not overlap in {segment_dir}")

        feet: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side in ("l", "r"):
            calc_x = np.interp(measured_time, estimated_time, _column(estimated_columns, estimated, f"ground_force_calcn_{side}_px"))
            calc_y = np.interp(measured_time, estimated_time, _column(estimated_columns, estimated, f"ground_force_calcn_{side}_py"))
            toes_x = np.interp(measured_time, estimated_time, _column(estimated_columns, estimated, f"ground_force_toes_{side}_px"))
            toes_y = np.interp(measured_time, estimated_time, _column(estimated_columns, estimated, f"ground_force_toes_{side}_py"))
            feet[side] = (np.column_stack((calc_x, calc_y)), np.column_stack((toes_x, toes_y)))

        for side, other in (("l", "r"), ("r", "l")):
            fz = _column(measured_columns, measured, f"ground_force_calcn_{side}_vz")
            cop = np.column_stack((
                _column(measured_columns, measured, f"ground_force_calcn_{side}_px"),
                _column(measured_columns, measured, f"ground_force_calcn_{side}_py"),
            ))
            stance = np.isfinite(fz) & (fz > 100.0)
            invalid_cop = stance & (~np.isfinite(cop).all(axis=1) | np.all(cop == 0.0, axis=1))
            valid = stance & ~invalid_cop
            all_n_stance += int(stance.sum())
            all_n_invalid += int(invalid_cop.sum())
            if not np.any(valid):
                continue

            own_a, own_b = feet[side]
            other_a, other_b = feet[other]
            own_distance = _point_to_segment_distance(cop[valid], own_a[valid], own_b[valid])
            other_distance = _point_to_segment_distance(cop[valid], other_a[valid], other_b[valid])
            midpoint = (own_a + own_b) * 0.5
            offset = cop[valid] - midpoint[valid]
            dx_med, dx_iqr = _summary(offset[:, 0])
            dy_med, dy_iqr = _summary(offset[:, 1])
            closer = own_distance < other_distance
            detail = {
                "subject": subject,
                "trial": trial,
                "segment": segment_name,
                "side": side,
                "stance_frames_fz_gt_100_n": int(stance.sum()),
                "invalid_cop_frames": int(invalid_cop.sum()),
                "median_cop_to_own_foot_cm": float(np.median(own_distance) * 100.0),
                "p90_cop_to_own_foot_cm": float(np.percentile(own_distance, 90) * 100.0),
                "median_cop_to_other_foot_cm": float(np.median(other_distance) * 100.0),
                "p90_cop_to_other_foot_cm": float(np.percentile(other_distance, 90) * 100.0),
                "own_foot_closer_percent": float(np.mean(closer) * 100.0),
                "median_offset_x_m": dx_med,
                "iqr_offset_x_m": dx_iqr,
                "median_offset_y_m": dy_med,
                "iqr_offset_y_m": dy_iqr,
                "median_cop_x_m": float(np.median(cop[valid, 0])),
                "median_cop_y_m": float(np.median(cop[valid, 1])),
                "median_own_foot_midpoint_x_m": float(np.median(midpoint[valid, 0])),
                "median_own_foot_midpoint_y_m": float(np.median(midpoint[valid, 1])),
                **roots,
            }
            detailed.append(detail)
            all_dx.append(offset[:, 0])
            all_dy.append(offset[:, 1])
            all_cop.append(cop[valid])
            all_midpoint.append(midpoint[valid])
            all_own_distance.append(own_distance)
            all_other_distance.append(other_distance)
            all_own_closer.append(closer)
            sample_groups.append((detail, cop[valid], own_a[valid], own_b[valid], other_a[valid], other_b[valid], measured_time[valid]))

    if not detailed:
        raise ValueError(f"No valid stance CoP samples for {subject}/{trial}")
    dx = np.concatenate(all_dx)
    dy = np.concatenate(all_dy)
    cop_points = np.concatenate(all_cop)
    midpoints = np.concatenate(all_midpoint)
    own_distance = np.concatenate(all_own_distance)
    other_distance = np.concatenate(all_other_distance)
    own_closer = np.concatenate(all_own_closer)
    shift = np.asarray([np.median(dx), np.median(dy)])
    candidate_own_distances: list[np.ndarray] = []
    candidate_other_distances: list[np.ndarray] = []
    candidate_own_closer: list[np.ndarray] = []
    root_transform_own_distances: list[np.ndarray] = []
    root_transform_other_distances: list[np.ndarray] = []
    root_transform_own_closer: list[np.ndarray] = []
    frame_transform_own_distances: list[np.ndarray] = []
    frame_transform_other_distances: list[np.ndarray] = []
    frame_transform_own_closer: list[np.ndarray] = []
    frame_transform_offsets: list[np.ndarray] = []
    root_transform_shift = np.asarray([
        roots["pipeline_minus_raw_pelvis_x_m"],
        roots["pipeline_minus_raw_pelvis_y_m"],
    ])
    for detail, points, own_a, own_b, other_a, other_b, sample_times in sample_groups:
        shifted_points = points - shift
        own_after = _point_to_segment_distance(shifted_points, own_a, own_b)
        other_after = _point_to_segment_distance(shifted_points, other_a, other_b)
        closer_after = own_after < other_after
        detail["candidate_cop_to_own_foot_cm"] = float(np.median(own_after) * 100.0)
        detail["candidate_cop_to_other_foot_cm"] = float(np.median(other_after) * 100.0)
        detail["candidate_own_foot_closer_percent"] = float(np.mean(closer_after) * 100.0)
        root_points = points + root_transform_shift
        root_own = _point_to_segment_distance(root_points, own_a, own_b)
        root_other = _point_to_segment_distance(root_points, other_a, other_b)
        root_closer = root_own < root_other
        detail["root_transform_cop_to_own_foot_cm"] = float(np.median(root_own) * 100.0)
        detail["root_transform_cop_to_other_foot_cm"] = float(np.median(root_other) * 100.0)
        detail["root_transform_own_foot_closer_percent"] = float(np.mean(root_closer) * 100.0)
        candidate_own_distances.append(own_after)
        candidate_other_distances.append(other_after)
        candidate_own_closer.append(closer_after)
        root_transform_own_distances.append(root_own)
        root_transform_other_distances.append(root_other)
        root_transform_own_closer.append(root_closer)
        frame_shift = np.column_stack([
            np.interp(sample_times, roots["_pipeline_time"], roots["_frame_shift"][:, axis])
            for axis in range(2)
        ])
        frame_points = points + frame_shift
        frame_own = _point_to_segment_distance(frame_points, own_a, own_b)
        frame_other = _point_to_segment_distance(frame_points, other_a, other_b)
        frame_closer = frame_own < frame_other
        frame_offset = frame_points - ((own_a + own_b) * 0.5)
        detail["frame_transform_cop_to_own_foot_cm"] = float(np.median(frame_own) * 100.0)
        detail["frame_transform_cop_to_other_foot_cm"] = float(np.median(frame_other) * 100.0)
        detail["frame_transform_own_foot_closer_percent"] = float(np.mean(frame_closer) * 100.0)
        frame_transform_own_distances.append(frame_own)
        frame_transform_other_distances.append(frame_other)
        frame_transform_own_closer.append(frame_closer)
        frame_transform_offsets.append(frame_offset)
    candidate_own = np.concatenate(candidate_own_distances)
    candidate_other = np.concatenate(candidate_other_distances)
    candidate_closer = np.concatenate(candidate_own_closer)
    root_transform_own = np.concatenate(root_transform_own_distances)
    root_transform_other = np.concatenate(root_transform_other_distances)
    root_transform_closer = np.concatenate(root_transform_own_closer)
    frame_transform_own = np.concatenate(frame_transform_own_distances)
    frame_transform_other = np.concatenate(frame_transform_other_distances)
    frame_transform_closer = np.concatenate(frame_transform_own_closer)
    frame_offsets = np.concatenate(frame_transform_offsets)
    median_cop = np.median(cop_points, axis=0)
    median_midpoint = np.median(midpoints, axis=0)
    trial_row = {
        "subject": subject,
        "trial": trial,
        "segments": len(segment_dirs),
        "stance_frames_fz_gt_100_n": all_n_stance,
        "invalid_cop_frames": all_n_invalid,
        "valid_cop_frames": int(len(dx)),
        "median_offset_x_m": float(np.median(dx)),
        "median_offset_y_m": float(np.median(dy)),
        "median_abs_offset_x_m": float(np.median(np.abs(dx))),
        "median_abs_offset_y_m": float(np.median(np.abs(dy))),
        "median_cop_x_m": float(median_cop[0]),
        "median_cop_y_m": float(median_cop[1]),
        "median_foot_midpoint_x_m": float(median_midpoint[0]),
        "median_foot_midpoint_y_m": float(median_midpoint[1]),
        "median_cop_to_own_foot_cm": float(np.median(own_distance) * 100.0),
        "p90_cop_to_own_foot_cm": float(np.percentile(own_distance, 90) * 100.0),
        "median_cop_to_other_foot_cm": float(np.median(other_distance) * 100.0),
        "own_foot_closer_percent": float(np.mean(own_closer) * 100.0),
        "candidate_cop_shift_x_m": float(shift[0]),
        "candidate_cop_shift_y_m": float(shift[1]),
        "candidate_median_cop_to_own_foot_cm": float(np.median(candidate_own) * 100.0),
        "candidate_p90_cop_to_own_foot_cm": float(np.percentile(candidate_own, 90) * 100.0),
        "candidate_median_cop_to_other_foot_cm": float(np.median(candidate_other) * 100.0),
        "candidate_own_foot_closer_percent": float(np.mean(candidate_closer) * 100.0),
        "frame_transform_median_offset_x_m": float(np.median(frame_offsets[:, 0])),
        "frame_transform_median_offset_y_m": float(np.median(frame_offsets[:, 1])),
        "frame_transform_median_cop_to_own_foot_cm": float(np.median(frame_transform_own) * 100.0),
        "frame_transform_p90_cop_to_own_foot_cm": float(np.percentile(frame_transform_own, 90) * 100.0),
        "frame_transform_median_cop_to_other_foot_cm": float(np.median(frame_transform_other) * 100.0),
        "frame_transform_own_foot_closer_percent": float(np.mean(frame_transform_closer) * 100.0),
        "root_transform_shift_x_m": float(root_transform_shift[0]),
        "root_transform_shift_y_m": float(root_transform_shift[1]),
        "root_transform_residual_offset_y_m": float(np.median(dy) + root_transform_shift[1]),
        "root_transform_median_cop_to_own_foot_cm": float(np.median(root_transform_own) * 100.0),
        "root_transform_p90_cop_to_own_foot_cm": float(np.percentile(root_transform_own, 90) * 100.0),
        "root_transform_median_cop_to_other_foot_cm": float(np.median(root_transform_other) * 100.0),
        "root_transform_own_foot_closer_percent": float(np.mean(root_transform_closer) * 100.0),
        "root_transform_own_closer_over_95_percent": float(np.mean(root_transform_closer) > 0.95),
        **roots,
        "dy_minus_raw_trans_y_start_m": float(np.median(dy) - roots["raw_moshpp_trans_y_start_m"]),
        "dy_plus_raw_trans_y_start_m": float(np.median(dy) + roots["raw_moshpp_trans_y_start_m"]),
        "dy_minus_raw_trans_y_mean_m": float(np.median(dy) - roots["raw_moshpp_trans_y_mean_m"]),
        "dy_plus_raw_trans_y_mean_m": float(np.median(dy) + roots["raw_moshpp_trans_y_mean_m"]),
        "dy_minus_pipeline_ty_start_m": float(np.median(dy) - roots["pipeline_pelvis_ty_start_m"]),
        "dy_plus_pipeline_ty_start_m": float(np.median(dy) + roots["pipeline_pelvis_ty_start_m"]),
        "dy_minus_pipeline_ty_mean_m": float(np.median(dy) - roots["pipeline_pelvis_ty_mean_m"]),
        "dy_plus_pipeline_ty_mean_m": float(np.median(dy) + roots["pipeline_pelvis_ty_mean_m"]),
        "offset_plus_root_transform_y_m": float(np.median(dy) + roots["pipeline_minus_raw_pelvis_y_m"]),
    }
    return detailed, trial_row


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze_subject(
    source_root: Path,
    dataset_root: Path,
    model_dir: Path,
    output_root: Path,
    subject: str,
) -> dict[str, int]:
    manifest = source_root / f"execution_manifest_{subject}.csv"
    included = [row for row in _read_csv_rows(manifest) if row.get("status") == "ok"]
    detail_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for item in included:
        detail, trial = _analyze_trial(source_root, dataset_root, model_dir, subject, item["trial"])
        detail_rows.extend(detail)
        trial_rows.append(trial)
    part_dir = output_root / "logs" / "parts"
    _write_rows(part_dir / f"cop_offset_before_{subject}.csv", DETAILED_FIELDS, detail_rows)
    _write_rows(part_dir / f"root_translation_vs_offset_{subject}.csv", TRIAL_FIELDS, trial_rows)
    return {"subject": subject, "trials": len(trial_rows), "segment_side_rows": len(detail_rows)}


def merge_parts(output_root: Path) -> dict[str, Any]:
    part_dir = output_root / "logs" / "parts"
    detail_rows: list[dict[str, str]] = []
    trial_rows: list[dict[str, str]] = []
    for subject in SUBJECTS:
        detail_rows.extend(_read_csv_rows(part_dir / f"cop_offset_before_{subject}.csv"))
        trial_rows.extend(_read_csv_rows(part_dir / f"root_translation_vs_offset_{subject}.csv"))
    trial_keys = {(row["subject"], row["trial"]) for row in trial_rows}
    if len(trial_rows) != 206 or len(trial_keys) != 206:
        raise ValueError(f"Expected 206 unique trial summaries; got {len(trial_rows)} rows, {len(trial_keys)} unique")
    _write_rows(output_root / "logs" / "cop_offset_before.csv", DETAILED_FIELDS, detail_rows)
    _write_rows(output_root / "logs" / "root_translation_vs_offset.csv", TRIAL_FIELDS, trial_rows)

    own_dist = np.asarray([float(row["median_cop_to_own_foot_cm"]) for row in detail_rows])
    abs_dx = np.asarray([abs(float(row["median_offset_x_m"])) * 100 for row in detail_rows])
    abs_dy = np.asarray([abs(float(row["median_offset_y_m"])) * 100 for row in detail_rows])
    trial_side_offsets: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in detail_rows:
        key = (row["subject"], row["trial"])
        trial_side_offsets.setdefault(key, {}).setdefault(row["side"], []).append(
            float(row["median_offset_y_m"])
        )
    trial_abs_dy_cm = np.asarray([
        float(np.median(np.abs([
            np.median(side_values) for side_values in sides.values()
        ]))) * 100.0
        for sides in trial_side_offsets.values()
    ])
    own_closer = np.asarray([float(row["own_foot_closer_percent"]) for row in detail_rows])
    candidate_own_dist = np.asarray([float(row["candidate_cop_to_own_foot_cm"]) for row in detail_rows])
    candidate_own_closer = np.asarray([float(row["candidate_own_foot_closer_percent"]) for row in detail_rows])
    candidate_trial_closer = np.asarray([float(row["candidate_own_foot_closer_percent"]) for row in trial_rows])
    root_transform_own_dist = np.asarray([float(row["root_transform_cop_to_own_foot_cm"]) for row in detail_rows])
    root_transform_own_closer = np.asarray([float(row["root_transform_own_foot_closer_percent"]) for row in detail_rows])
    root_transform_trial_closer = np.asarray([float(row["root_transform_own_closer_over_95_percent"]) for row in trial_rows])
    frame_transform_own_dist = np.asarray([float(row["frame_transform_cop_to_own_foot_cm"]) for row in detail_rows])
    frame_transform_own_closer = np.asarray([float(row["frame_transform_own_foot_closer_percent"]) for row in detail_rows])
    frame_transform_trial_median = np.asarray([float(row["frame_transform_median_cop_to_own_foot_cm"]) for row in trial_rows])
    frame_transform_trial_p90 = np.asarray([float(row["frame_transform_p90_cop_to_own_foot_cm"]) for row in trial_rows])
    frame_transform_trial_closer = np.asarray([float(row["frame_transform_own_foot_closer_percent"]) for row in trial_rows])
    frame_transform_trial_dx = np.asarray([float(row["frame_transform_median_offset_x_m"]) for row in trial_rows])
    frame_transform_trial_dy = np.asarray([float(row["frame_transform_median_offset_y_m"]) for row in trial_rows])
    rmse = {}
    for key in (
        "dy_minus_raw_trans_y_start_m", "dy_plus_raw_trans_y_start_m",
        "dy_minus_raw_trans_y_mean_m", "dy_plus_raw_trans_y_mean_m",
        "dy_minus_pipeline_ty_start_m", "dy_plus_pipeline_ty_start_m",
        "dy_minus_pipeline_ty_mean_m", "dy_plus_pipeline_ty_mean_m",
        "offset_plus_root_transform_y_m",
    ):
        values = np.asarray([float(row[key]) for row in trial_rows])
        rmse[key] = float(np.sqrt(np.mean(values ** 2)))
    summary = {
        "trials": len(trial_rows),
        "segment_side_rows": len(detail_rows),
        "median_segment_side_cop_to_own_foot_cm": float(np.median(own_dist)),
        "segment_side_fraction_own_distance_gt_8cm": float(np.mean(own_dist > 8.0)),
        "segment_side_fraction_own_distance_gt_15cm": float(np.mean(own_dist > 15.0)),
        "median_abs_segment_side_offset_x_cm": float(np.median(abs_dx)),
        "median_abs_segment_side_offset_y_cm": float(np.median(abs_dy)),
        "trial_median_abs_offset_y_under_5cm": int(np.sum(trial_abs_dy_cm < 5.0)),
        "trial_median_abs_offset_y_5_to_15cm": int(np.sum((trial_abs_dy_cm >= 5.0) & (trial_abs_dy_cm <= 15.0))),
        "trial_median_abs_offset_y_over_15cm": int(np.sum(trial_abs_dy_cm > 15.0)),
        "segment_side_median_own_foot_closer_percent": float(np.median(own_closer)),
        "candidate_median_segment_side_cop_to_own_foot_cm": float(np.median(candidate_own_dist)),
        "candidate_segment_side_fraction_own_closer_over_95_percent": float(np.mean(candidate_own_closer > 95.0)),
        "candidate_trials_own_closer_over_95_percent": int(np.sum(candidate_trial_closer > 95.0)),
        "root_transform_median_segment_side_cop_to_own_foot_cm": float(np.median(root_transform_own_dist)),
        "root_transform_segment_side_fraction_own_closer_over_95_percent": float(np.mean(root_transform_own_closer > 95.0)),
        "root_transform_trials_own_closer_over_95_percent": int(np.sum(root_transform_trial_closer > 0.5)),
        "root_transform_median_abs_offset_residual_y_cm": float(np.median([
            abs(float(row["root_transform_residual_offset_y_m"])) * 100.0 for row in trial_rows
        ])),
        "root_candidate_rmse_m": rmse,
        "frame_transform_median_segment_side_cop_to_own_foot_cm": float(np.median(frame_transform_own_dist)),
        "frame_transform_segment_side_fraction_own_closer_over_95_percent": float(np.mean(frame_transform_own_closer > 95.0)),
        "frame_transform_trials_distance_criteria_pass": int(np.sum((frame_transform_trial_median < 5.0) & (frame_transform_trial_p90 < 10.0))),
        "frame_transform_trials_offset_criteria_pass": int(np.sum((np.abs(frame_transform_trial_dx) < 0.03) & (np.abs(frame_transform_trial_dy) < 0.03))),
        "frame_transform_trials_side_criteria_pass": int(np.sum(frame_transform_trial_closer > 95.0)),
        "frame_transform_trials_all_criteria_pass": int(np.sum(
            (frame_transform_trial_median < 5.0)
            & (frame_transform_trial_p90 < 10.0)
            & (np.abs(frame_transform_trial_dx) < 0.03)
            & (np.abs(frame_transform_trial_dy) < 0.03)
            & (frame_transform_trial_closer > 95.0)
        )),
        "reports": [
            str(output_root / "logs" / "cop_offset_before.csv"),
            str(output_root / "logs" / "root_translation_vs_offset.csv"),
        ],
    }
    summary_path = output_root / "logs" / "cop_offset_before_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", type=Path, default=Path(__file__).resolve().parents[3] / "model" / "smpl")
    parser.add_argument("--subject", choices=SUBJECTS)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        print(json.dumps(merge_parts(args.output_dir), indent=2))
    elif args.subject:
        print(json.dumps(analyze_subject(
            args.source_dir, args.dataset_dir, args.smplx_model_dir, args.output_dir, args.subject
        ), indent=2))
    else:
        parser.error("choose --subject for a parallel worker or --merge after all workers finish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

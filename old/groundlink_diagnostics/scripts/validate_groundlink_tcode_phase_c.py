#!/usr/bin/env python3
"""Validate and apply the Phase C T_code candidate on a completed cohort.

The input target tree is read-only. The output tree is modified only to write
the approved ``side_invalid`` mask after its cohort-wide <1% gate passes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from aggregate_groundlink_torque_targets import EXPECTED_COUNTS, aggregate
from diagnose_groundlink_cop_offsets import _column, _point_to_segment_distance, _read_mot
from extract_groundlink_torque_targets import _verify_mot_add_tcode

SUBJECTS = tuple(EXPECTED_COUNTS)
SIDES = ("l", "r")
COORDS = (
    "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
)
EDGE_FRAMES = 75
FZ_DISTANCE_N = 100.0
FZ_VALIDATION_N = 20.0
NEAR_GROUND_M = 0.05
EXPECTED_FRAMES = 861_064


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): _sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file() and not file.is_symlink()
    }


def _resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _setup_paths(segment_dir: Path) -> tuple[Path, Path, Path, Path]:
    setup = next(iter(sorted(segment_dir.glob("Setup_ID_measuredGRF*.xml"))), None)
    if setup is None:
        raise FileNotFoundError(f"No measured ID setup in {segment_dir}")
    tool = ET.parse(setup).getroot().find("InverseDynamicsTool")
    if tool is None:
        raise ValueError(f"No InverseDynamicsTool in {setup}")
    model = _resolve(tool.findtext("model_file") or "", segment_dir)
    coordinates = _resolve(tool.findtext("coordinates_file") or "", segment_dir)
    external = _resolve(tool.findtext("external_loads_file") or "", segment_dir)
    datafile = ET.parse(external).getroot().find(".//datafile")
    if datafile is None or not datafile.text:
        raise ValueError(f"No force MOT in {external}")
    measured = _resolve(datafile.text.strip(), external.parent)
    if not all(path.is_file() for path in (model, coordinates, measured)):
        raise FileNotFoundError(f"Incomplete setup inputs for {segment_dir}")
    return setup, model, coordinates, measured


_SKELETON_HASH: str | None = None
_SKELETON: Any = None


def _foot_kinematics(model_path: Path, ik_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Return body-origin xyz for calcaneus/toes from the saved IK model."""
    global _SKELETON, _SKELETON_HASH
    digest = _sha256(model_path)
    if _SKELETON is None or _SKELETON_HASH != digest:
        try:
            import nimblephysics as nimble
        except ImportError as exc:  # pragma: no cover - available on HPC
            raise RuntimeError("nimblephysics is required for IK foot-height checks") from exc
        _SKELETON = nimble.biomechanics.OpenSimParser.parseOsim(
            str(model_path), "", True
        ).skeleton
        _SKELETON_HASH = digest
    import nimblephysics as nimble

    skeleton = _SKELETON
    motion = nimble.biomechanics.OpenSimParser.loadMot(skeleton, str(ik_path))
    poses = np.asarray(motion.poses, dtype=np.float64)
    times = np.asarray(motion.timestamps, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != times.size:
        raise ValueError(f"Unexpected IK shape in {ik_path}: {poses.shape}, {times.shape}")
    names = ("calcn_l", "toes_l", "calcn_r", "toes_r")
    bodies = [skeleton.getBodyNode(name) for name in names]
    xyz = np.empty((times.size, len(names), 3), dtype=np.float64)
    for frame in range(times.size):
        skeleton.setPositions(poses[:, frame])
        for body_index, body in enumerate(bodies):
            xyz[frame, body_index] = np.asarray(
                body.getWorldTransform().translation(), dtype=np.float64
            )
    return times, xyz, float(skeleton.getMass())


def _contiguous_intervals(frames: list[int]) -> list[tuple[int, int]]:
    if not frames:
        return []
    values = sorted(set(frames))
    result: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def _finite_metrics(target: np.ndarray, prediction: np.ndarray, denominator: float) -> tuple[float, float, float, int]:
    mask = np.isfinite(target) & np.isfinite(prediction)
    target, prediction = target[mask], prediction[mask]
    if not target.size:
        return float("nan"), float("nan"), float("nan"), 0
    rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if target.size > 1 and np.std(target) > 0 and np.std(prediction) > 0
        else float("nan")
    )
    return rmse, correlation, rmse / denominator if denominator > 0 else float("nan"), int(target.size)


def _lag_correlation(measured: np.ndarray, estimated: np.ndarray, lag: int) -> tuple[float, int]:
    if measured.size != estimated.size:
        raise ValueError("Measured and estimated force chunks have different lengths")
    if abs(lag) >= measured.size:
        return float("nan"), 0
    if lag >= 0:
        x, y = measured[: measured.size - lag or None], estimated[lag:]
    else:
        x, y = measured[-lag:], estimated[: estimated.size + lag]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def _trial_keys(index_path: Path) -> list[tuple[str, str]]:
    _, rows = _read_rows(index_path)
    selected = [(row["subject"], row["trial"]) for row in rows if row.get("included") == "1"]
    counts = {subject: sum(key[0] == subject for key in selected) for subject in SUBJECTS}
    if len(selected) != 206 or counts != EXPECTED_COUNTS or len(set(selected)) != 206:
        raise ValueError(f"H1 index mismatch: n={len(selected)}, counts={counts}")
    return selected


def validate(source_root: Path, output_root: Path, index_path: Path, tcode_audit: Path) -> dict[str, Any]:
    source_root, output_root = source_root.resolve(), output_root.resolve()
    trial_keys = _trial_keys(index_path)
    _, audit_rows = _read_rows(tcode_audit)
    if len(audit_rows) != 206:
        raise ValueError(f"T_code audit has {len(audit_rows)} rows, expected 206")
    tcode_max = max(float(row["max_abs_delta_m"]) for row in audit_rows)

    side_groups: dict[tuple[str, str, int, str], dict[str, np.ndarray]] = {}
    mask_candidates: dict[tuple[str, str, int, str], set[int]] = defaultdict(set)
    row_cache: dict[tuple[str, str], tuple[list[str], list[dict[str, str]], list[dict[str, str]]]] = {}
    distance_before: dict[tuple[str, str], list[float]] = defaultdict(list)
    distance_after: dict[tuple[str, str], list[float]] = defaultdict(list)
    distance_after_by_frame: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    offset_activity: dict[tuple[str, str], list[float]] = defaultdict(list)
    swing_deltas: list[float] = []
    hat_hash_failures: list[str] = []
    mot_segments_verified = 0
    valid_mot_points = 0
    total_frames = 0
    total_source_valid = 0
    subject_trial_arrays: dict[tuple[str, str], dict[str, Any]] = {}
    pelvis_rows: list[dict[str, Any]] = []
    magnitude_values: dict[str, list[float]] = defaultdict(list)
    trial_magnitude: dict[tuple[str, str], list[float]] = defaultdict(list)
    soccer_rows: list[dict[str, Any]] = []

    # Preserve a mask proposal if a previous validation run reached the <1% gate
    # and then stopped later while rebuilding a manifest or writing reports.
    prior_intervals = output_root / "logs" / "side_invalid_intervals.csv"
    if prior_intervals.is_file():
        _, prior_rows = _read_rows(prior_intervals)
        for row in prior_rows:
            key = (row["subject"], row["trial"], int(row["segment"]), row["side"])
            mask_candidates[key].update(range(int(row["frame_start"]), int(row["frame_end"]) + 1))

    for subject, trial in trial_keys:
        key = (subject, trial)
        src_trial, out_trial = source_root / subject / trial, output_root / subject / trial
        source_fields, source_rows = _read_rows(src_trial / "torque_targets.csv")
        output_fields, output_rows = _read_rows(out_trial / "torque_targets.csv")
        if len(source_rows) != len(output_rows):
            raise ValueError(f"Frame count mismatch for {subject}/{trial}")
        for src, out in zip(source_rows, output_rows):
            if any(src[name] != out[name] for name in ("frame", "time", "segment")):
                raise ValueError(f"frame/time/segment changed in {subject}/{trial}")
        row_cache[key] = (output_fields, source_rows, output_rows)
        total_frames += len(output_rows)
        total_source_valid += sum(int(float(row["valid"])) for row in source_rows)
        metadata = json.loads((out_trial / "run_metadata.json").read_text(encoding="utf-8"))
        shift = tuple(float(value) for value in metadata["cop_shift_m"])
        if len(shift) != 3 or metadata.get("cop_shift_source") != "tcode":
            raise ValueError(f"Invalid T_code metadata for {subject}/{trial}")

        target_idx = np.arange(len(output_rows), dtype=int)
        frames = np.asarray([int(float(row["frame"])) for row in output_rows], dtype=int)
        times_trial = np.asarray([float(row["time"]) for row in output_rows], dtype=float)
        segments_trial = np.asarray([int(float(row["segment"])) for row in output_rows], dtype=int)
        source_valid = np.asarray([int(float(row["valid"])) == 1 for row in source_rows])
        hat_columns_ok = True
        for name in COORDS:
            col = f"{name}_tau_hat"
            if [row[col] for row in source_rows] != [row[col] for row in output_rows]:
                hat_columns_ok = False
                hat_hash_failures.append(f"{subject}/{trial}/{col}")
        if not hat_columns_ok:
            pass

        trial_measured_total = np.full(len(output_rows), np.nan)
        trial_estimated_total = np.full(len(output_rows), np.nan)
        trial_mass: float | None = None
        trial_usable = np.zeros(len(output_rows), dtype=bool)
        trial_fz_by_side = {side: np.full(len(output_rows), np.nan) for side in SIDES}

        for segment in sorted(set(segments_trial.tolist())):
            ids = target_idx[segments_trial == segment]
            segment_dir = out_trial / "segments" / f"segment_{segment}"
            source_segment = src_trial / "segments" / f"segment_{segment}"
            setup, model_path, ik_path, measured_path = _setup_paths(segment_dir)
            source_setup = next(iter(sorted(source_segment.glob("Setup_ID_measuredGRF*.xml"))), None)
            if source_setup is None or _sha256(setup) != _sha256(source_setup):
                raise ValueError(f"Measured setup changed for {subject}/{trial}/segment_{segment}")
            source_measured = _setup_paths(source_segment)[3]
            point_count = _verify_mot_add_tcode(source_measured, measured_path, shift[0], shift[1])
            valid_mot_points += point_count
            mot_segments_verified += 1
            if _tree_hashes(source_segment / "estimated_id") != _tree_hashes(segment_dir / "estimated_id"):
                hat_hash_failures.append(f"{subject}/{trial}/segment_{segment}/estimated_id")

            measured_columns, measured = _read_mot(measured_path)
            source_columns, source_mot = _read_mot(source_measured)
            time_mot = _column(measured_columns, measured, "time")
            if measured_columns != source_columns:
                raise ValueError(f"Measured MOT columns changed: {measured_path}")
            target_times = times_trial[ids]
            if target_times.min() < time_mot[0] - 1e-5 or target_times.max() > time_mot[-1] + 1e-5:
                raise ValueError(f"MOT/target time mismatch in {measured_path}")
            ik_times, foot_xyz_ik, mass = _foot_kinematics(model_path, ik_path)
            trial_mass = mass if trial_mass is None else trial_mass
            if abs(trial_mass - mass) > 1e-6:
                raise ValueError(f"Model mass differs across segments in {subject}/{trial}")
            foot_xyz = np.empty((len(ids), 4, 3), dtype=float)
            for body_index in range(4):
                for axis in range(3):
                    foot_xyz[:, body_index, axis] = np.interp(
                        target_times, ik_times, foot_xyz_ik[:, body_index, axis]
                    )
            cop_out = np.column_stack([
                np.interp(target_times, time_mot, _column(measured_columns, measured, f"ground_force_calcn_{side}_p{axis}"))
                for side in SIDES for axis in ("x", "y")
            ])
            cop_in = np.column_stack([
                np.interp(target_times, time_mot, _column(source_columns, source_mot, f"ground_force_calcn_{side}_p{axis}"))
                for side in SIDES for axis in ("x", "y")
            ])
            fz = {
                side: np.interp(target_times, time_mot, _column(measured_columns, measured, f"ground_force_calcn_{side}_vz"))
                for side in SIDES
            }
            estimated_path = next(iter(sorted((segment_dir / "estimated_id").glob("*_grf_estimated.mot"))), None)
            if estimated_path is None:
                raise FileNotFoundError(f"Missing copied estimated GRF MOT in {segment_dir}")
            estimated_columns, estimated = _read_mot(estimated_path)
            estimated_time = _column(estimated_columns, estimated, "time")
            est_total = sum(
                np.interp(target_times, estimated_time, _column(estimated_columns, estimated, f"ground_force_{body}_{side}_vz"))
                for side in SIDES for body in ("calcn", "toes")
            )
            measured_total = fz["l"] + fz["r"]
            trial_measured_total[ids] = measured_total
            trial_estimated_total[ids] = est_total
            for side in SIDES:
                trial_fz_by_side[side][ids] = fz[side]

            length = len(ids)
            trim = np.zeros(length, dtype=bool)
            if length > 2 * EDGE_FRAMES:
                trim[EDGE_FRAMES:length - EDGE_FRAMES] = True
            else:
                trim[:] = False
            valid = source_valid[ids]
            trial_usable[ids] = valid & trim
            # Score the pre-mask criterion even when resuming after a prior run
            # applied side_invalid; retained frames are applied below.
            analysis_valid = valid & trim

            point_map = {"l": (0, 1), "r": (2, 3)}
            for side, other in (("l", "r"), ("r", "l")):
                side_idx = 0 if side == "l" else 2
                other_idx = 2 if other == "r" else 0
                cop_side = cop_out[:, side_idx:side_idx + 2]
                cop_source_side = cop_in[:, side_idx:side_idx + 2]
                own_a, own_b = foot_xyz[:, point_map[side][0], :2], foot_xyz[:, point_map[side][1], :2]
                other_a, other_b = foot_xyz[:, point_map[other][0], :2], foot_xyz[:, point_map[other][1], :2]
                d_after = _point_to_segment_distance(cop_side, own_a, own_b)
                d_before = _point_to_segment_distance(cop_source_side, own_a, own_b)
                d_other = _point_to_segment_distance(cop_side, other_a, other_b)
                cop_ok = np.isfinite(cop_side).all(axis=1) & ~np.all(cop_side == 0.0, axis=1)
                stance = np.isfinite(fz[side]) & (fz[side] > FZ_DISTANCE_N)
                usable = analysis_valid & stance & cop_ok
                distance_before[key].extend((d_before[usable] * 100.0).tolist())
                distance_after[key].extend((d_after[usable] * 100.0).tolist())
                distance_after_by_frame[key].extend(
                    (int(frame), float(distance * 100.0))
                    for frame, distance, keep in zip(frames[ids], d_after, usable)
                    if keep
                )
                midpoint = (own_a + own_b) * 0.5
                activity = trial.split("_")[0]
                offset_activity[(activity, side)].extend(
                    (cop_side[usable] - midpoint[usable]).tolist()
                )
                other_height = np.minimum(foot_xyz[:, other_idx, 2], foot_xyz[:, other_idx + 1, 2])
                near = np.isfinite(other_height) & (other_height < NEAR_GROUND_M)
                old_correct = d_after < d_other
                new_correct = (~near) | old_correct
                sample = usable
                sample_frames = frames[ids]
                side_key = (subject, trial, segment, side)
                side_groups[side_key] = {
                    "frames": sample_frames,
                    "usable": sample,
                    "old_correct": old_correct,
                    "new_correct": new_correct,
                    "wrong_near": sample & near & ~old_correct,
                    "d_after": d_after,
                    "d_other": d_other,
                    "near": near,
                }

            # Determine the stable-swing window on the complete ID segment time
            # series. H1 can split its trial rows at a boundary that is inside an
            # ID segment, so applying this window only to ``ids`` would treat the
            # first rows after that boundary as if they had no preceding samples.
            _, source_segment_rows = _read_rows(source_segment / "target_segment.csv")
            _, output_segment_rows = _read_rows(segment_dir / "target_segment.csv")
            if len(source_segment_rows) != len(output_segment_rows):
                raise ValueError(f"ID segment frame count changed for {subject}/{trial}/segment_{segment}")
            segment_times = np.asarray(
                [float(row["time"]) for row in output_segment_rows], dtype=float
            )
            for source_row, output_row in zip(source_segment_rows, output_segment_rows):
                if any(source_row[name] != output_row[name] for name in ("frame", "time", "segment")):
                    raise ValueError(f"ID segment frame/time/segment changed for {subject}/{trial}/segment_{segment}")
            segment_fz = {
                side: np.interp(
                    segment_times,
                    time_mot,
                    _column(measured_columns, measured, f"ground_force_calcn_{side}_vz"),
                )
                for side in SIDES
            }
            both_zero = (segment_fz["l"] == 0.0) & (segment_fz["r"] == 0.0)
            stable_zero = both_zero.copy()
            for offset in range(1, 11):
                stable_zero[offset:] &= both_zero[:-offset]
                stable_zero[:-offset] &= both_zero[offset:]
            # Compare only final trial rows, but use the surrounding samples from
            # the full ID segment when deciding whether a row is stable swing.
            global_by_time = {round(float(times_trial[i]), 7): int(i) for i in ids}
            for local, segment_row in enumerate(output_segment_rows):
                if not stable_zero[local]:
                    continue
                segment_time = float(segment_row["time"])
                global_index = global_by_time.get(round(segment_time, 7))
                if global_index is None:
                    continue
                for coord in COORDS:
                    column = f"{coord}_tau_target"
                    swing_deltas.append(
                        abs(float(output_rows[global_index][column]) - float(source_rows[global_index][column]))
                    )

            # Preserve per-frame signals for subject-level lag estimation and tau validation.
            trial_mass = float(mass)
            subject_trial_arrays[key] = {
                "frames": frames,
                "times": times_trial,
                "segments": segments_trial,
                "valid": source_valid,
                "usable": trial_usable,
                "measured_total": trial_measured_total,
                "estimated_total": trial_estimated_total,
                "fz": trial_fz_by_side,
                "mass": trial_mass,
                "source_rows": source_rows,
                "output_rows": output_rows,
                "side_samples_ready": True,
            }

            # Check pelvis generalized forces/moments from source and rerun ID STOs.
            src_sto = next(iter(sorted(source_segment.glob("*_id_measuredGRF.sto"))), None)
            out_sto = next(iter(sorted(segment_dir.glob("*_id_measuredGRF.sto"))), None)
            if src_sto and out_sto:
                src_labels, src_values = _read_mot(src_sto)
                out_labels, out_values = _read_mot(out_sto)
                if src_labels != out_labels or len(src_values) != len(out_values):
                    raise ValueError(f"ID STO schema changed: {out_sto}")
                force_cols = [c for c in src_labels if c.startswith("pelvis_") and c.endswith("_force")]
                moment_cols = [c for c in src_labels if c.startswith("pelvis_") and c.endswith("_moment")]
                force_delta = max((float(np.max(np.abs(_column(src_labels, src_values, c) - _column(out_labels, out_values, c)))) for c in force_cols), default=0.0)
                before_values = np.concatenate([_column(src_labels, src_values, c) for c in moment_cols]) if moment_cols else np.asarray([], dtype=float)
                after_values = np.concatenate([_column(out_labels, out_values, c) for c in moment_cols]) if moment_cols else np.asarray([], dtype=float)
                before_rms = float(np.sqrt(np.mean(before_values ** 2))) if before_values.size else float("nan")
                after_rms = float(np.sqrt(np.mean(after_values ** 2))) if after_values.size else float("nan")
                pelvis_rows.append({"subject": subject, "trial": trial, "segment": segment, "pelvis_force_max_abs_delta_nm": force_delta, "pelvis_moment_rms_before_nm": before_rms, "pelvis_moment_rms_after_nm": after_rms})

        if trial_mass is None:
            raise ValueError(f"No segments in {subject}/{trial}")
        for coord in COORDS:
            values = np.asarray([float(row[f"{coord}_tau_target"]) / trial_mass for row in output_rows if int(float(row["valid"])) == 1])
            p99 = float(np.percentile(np.abs(values), 99)) if values.size else float("nan")
            magnitude_values[(subject, coord)].extend(np.abs(values).tolist())
            trial_magnitude[key].append(p99)
            if trial.endswith("soccerkick_1") and subject == "s001" and coord == "ankle_angle_l":
                soccer_rows.append({"subject": subject, "trial": trial, "coord": coord, "before_min_nmkg": float(np.min([float(r[f"{coord}_tau_target"]) / trial_mass for r in source_rows])), "after_min_nmkg": float(np.min(values)) if values.size else float("nan"), "after_p99_abs_nmkg": p99})
            if trial.endswith("soccerkick_3") and subject == "s006" and coord == "ankle_angle_r":
                soccer_rows.append({"subject": subject, "trial": trial, "coord": coord, "before_min_nmkg": float(np.min([float(r[f"{coord}_tau_target"]) / trial_mass for r in source_rows])), "after_min_nmkg": float(np.min(values)) if values.size else float("nan"), "after_p99_abs_nmkg": p99})

    # Derive candidate side masks only for failing segment-side groups.
    for side_key, sample in side_groups.items():
        usable = sample["usable"]
        n = int(np.count_nonzero(usable))
        if n < 50:
            continue
        score = 100.0 * float(np.count_nonzero(sample["new_correct"] & usable)) / n
        if score <= 95.0:
            for frame in sample["frames"][sample["wrong_near"]]:
                mask_candidates[side_key].add(int(frame))

    mandatory_key = ("s001", "lambadadance_3")
    mandatory_segment = 3
    mandatory_fields, mandatory_source_rows, _ = row_cache[mandatory_key]
    mandatory_frames = [
        int(float(row["frame"])) for row in mandatory_source_rows
        if int(float(row["segment"])) == mandatory_segment
    ]
    if len(mandatory_frames) != 17:
        raise ValueError(f"Mandatory IK-root mask expected 17 rows, found {len(mandatory_frames)}")
    for side in SIDES:
        for frame in mandatory_frames:
            mask_candidates[("s001", "lambadadance_3", mandatory_segment, side)].add(frame)

    # The mask is a frame-level union because torque_targets.csv has one valid column.
    mask_by_trial: dict[tuple[str, str], set[int]] = defaultdict(set)
    for (subject, trial, _segment, _side), frames in mask_candidates.items():
        mask_by_trial[(subject, trial)].update(frames)
    proposed_valid = sum(
        1 for key, frames in mask_by_trial.items()
        for row in row_cache[key][1]
        if int(float(row["frame"])) in frames and int(float(row["valid"])) == 1
    )
    mask_fraction = proposed_valid / max(1, total_source_valid)

    interval_rows: list[dict[str, Any]] = []
    for (subject, trial, segment, side), frames_for_side in sorted(mask_candidates.items()):
        source_rows = row_cache[(subject, trial)][1]
        valid_by_frame = {int(float(row["frame"])): int(float(row["valid"])) for row in source_rows}
        reason = "mandatory_root_ik_zero" if (subject, trial, segment) == ("s001", "lambadadance_3", 3) else "opposite_foot_nearer_and_grounded"
        for start, end in _contiguous_intervals(list(frames_for_side)):
            interval_rows.append({
                "subject": subject, "trial": trial, "segment": segment, "side": side,
                "frame_start": start, "frame_end": end,
                "n_valid_frames": sum(valid_by_frame.get(frame, 0) for frame in range(start, end + 1)),
                "reason": reason,
            })
    logs = output_root / "logs"
    _write_rows(logs / "side_invalid_intervals.csv", ["subject", "trial", "segment", "side", "frame_start", "frame_end", "n_valid_frames", "reason"], interval_rows)
    if not mask_fraction < 0.01:
        raise RuntimeError(f"Proposed side_invalid mask is {proposed_valid}/{total_frames} frames ({mask_fraction:.3%}); must be <1%. No mask was applied.")

    # Apply the gated mask and update per-trial metadata/manifests.
    for key, (fields, source_rows, output_rows) in row_cache.items():
        flagged = mask_by_trial.get(key, set())
        if "side_invalid" not in fields:
            fields.append("side_invalid")
        for source, output in zip(source_rows, output_rows):
            invalid = int(float(source["frame"])) in flagged
            output["side_invalid"] = "1" if invalid else "0"
            output["valid"] = "1" if int(float(source["valid"])) == 1 and not invalid else "0"
        target_path = output_root / key[0] / key[1] / "torque_targets.csv"
        _write_rows(target_path, fields, output_rows)
        metadata_path = target_path.with_name("run_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update({
            "side_invalid_frames": len(flagged),
            "side_invalid_valid_frames": sum(1 for row in source_rows if int(float(row["frame"])) in flagged and int(float(row["valid"])) == 1),
            "side_invalid_mask_applied": True,
            "output_csv": str(target_path),
        })
        temp = metadata_path.with_suffix(".json.partial")
        temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temp.replace(metadata_path)

    # Recalculate side percentages after the accepted frame mask.
    side_after_rows: list[dict[str, Any]] = []
    side_before_rows: list[dict[str, Any]] = []
    for side_key, sample in sorted(side_groups.items()):
        subject, trial, segment, side = side_key
        rows = row_cache[(subject, trial)][2]
        final_valid_frames = {
            int(float(row["frame"])) for row in rows
            if int(float(row["segment"])) == segment and int(float(row["valid"])) == 1
        }
        selected = sample["usable"]
        frames = sample["frames"]
        retained = selected & np.asarray([int(frame) in final_valid_frames for frame in frames])
        n_before, n_after = int(np.count_nonzero(selected)), int(np.count_nonzero(retained))
        percent = lambda flags, chosen, n: 100.0 * float(np.count_nonzero(flags & chosen)) / n if n else float("nan")
        side_after_rows.append({
            "subject": subject, "trial": trial, "segment": segment, "side": side,
            "usable_stance_frames": n_after,
            "old_criterion_before_mask_percent": percent(sample["old_correct"], selected, n_before),
            "new_criterion_before_mask_percent": percent(sample["new_correct"], selected, n_before),
            "old_criterion_after_mask_percent": percent(sample["old_correct"], retained, n_after),
            "new_criterion_after_mask_percent": percent(sample["new_correct"], retained, n_after),
            "correct_percent": percent(sample["new_correct"], retained, n_after),
            "opposite_near_ground_frames_after_mask": int(np.count_nonzero(sample["near"] & retained)),
        })
    _write_rows(logs / "side_criterion_before_after.csv", ["subject", "trial", "segment", "side", "usable_stance_frames", "old_criterion_before_mask_percent", "new_criterion_before_mask_percent", "old_criterion_after_mask_percent", "new_criterion_after_mask_percent", "correct_percent", "opposite_near_ground_frames_after_mask"], side_after_rows)

    # Rebuild manifests after valid/side_invalid changed the trial CSV and metadata hashes.
    for subject in SUBJECTS:
        manifest = output_root / f"execution_manifest_{subject}.csv"
        fields, records = _read_rows(manifest)
        for record in records:
            path = Path(record["output_csv"])
            meta_path = Path(record["metadata_json"])
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            record["output_sha256"] = _sha256(path)
            record["metadata_sha256"] = _sha256(meta_path)
            record["source_hashes_json"] = json.dumps(metadata.get("source_hashes", {}), sort_keys=True, separators=(",", ":"))
            record["source_details_json"] = json.dumps(metadata.get("source_details", {}), sort_keys=True, separators=(",", ":"))
        _write_rows(manifest, fields, records)
    aggregate(output_root, "nomuscles_copfix_tcode")

    # Criterion 2 per trial.
    offset_rows: list[dict[str, Any]] = []
    distance_pass_trials = 0
    for subject, trial in trial_keys:
        excluded_frames = mask_by_trial.get((subject, trial), set())
        values = np.asarray([
            value for frame, value in distance_after_by_frame[(subject, trial)]
            if frame not in excluded_frames
        ], dtype=float)
        median = float(np.median(values)) if values.size else float("nan")
        p90 = float(np.percentile(values, 90)) if values.size else float("nan")
        passed = bool(values.size and median < 5.0 and p90 < 10.0)
        distance_pass_trials += int(passed)
        offset_rows.append({"subject": subject, "trial": trial, "stance_samples": len(values), "median_cop_own_foot_cm": median, "p90_cop_own_foot_cm": p90, "pass": int(passed)})
    before_rows = []
    for subject, trial in trial_keys:
        values = np.asarray(distance_before[(subject, trial)], dtype=float)
        before_rows.append({
            "subject": subject, "trial": trial, "stance_samples": len(values),
            "median_cop_own_foot_cm": float(np.median(values)) if values.size else float("nan"),
            "p90_cop_own_foot_cm": float(np.percentile(values, 90)) if values.size else float("nan"),
        })
    _write_rows(logs / "cop_offset_before.csv", ["subject", "trial", "stance_samples", "median_cop_own_foot_cm", "p90_cop_own_foot_cm"], before_rows)
    _write_rows(logs / "cop_offset_after.csv", ["subject", "trial", "stance_samples", "median_cop_own_foot_cm", "p90_cop_own_foot_cm", "pass"], offset_rows)

    # Swing unchanged, hat integrity, and valid-frame integrity.
    swing_max = max(swing_deltas, default=float("inf"))
    total_output_valid = 0
    integrity_failures: list[str] = []
    for key, (fields, source_rows, output_rows) in row_cache.items():
        if len(source_rows) != len(output_rows):
            integrity_failures.append(f"{key[0]}/{key[1]}:frame_count")
        for old, new in zip(source_rows, output_rows):
            expected = int(float(old["valid"])) * (1 - int(float(new.get("side_invalid", "0"))))
            if int(float(new["valid"])) != expected:
                integrity_failures.append(f"{key[0]}/{key[1]}:valid_mask")
                break
            total_output_valid += int(float(new["valid"]))
            if int(float(new["valid"])):
                for coord in COORDS:
                    for suffix in ("target", "hat"):
                        if not math.isfinite(float(new[f"{coord}_tau_{suffix}"])):
                            integrity_failures.append(f"{key[0]}/{key[1]}:nan_{coord}_{suffix}")
                            break
    if total_frames != EXPECTED_FRAMES:
        integrity_failures.append(f"total_frames:{total_frames}!={EXPECTED_FRAMES}")

    side_blockers = [row for row in side_after_rows if int(row["usable_stance_frames"]) >= 50 and not float(row["correct_percent"]) > 95.0]
    distance_failures = [row for row in offset_rows if not row["pass"]]

    # Recompute magnitudes after the accepted mask has updated valid.
    magnitude_values.clear()
    trial_magnitude.clear()
    soccer_rows = []
    for key, arrays in subject_trial_arrays.items():
        subject, trial = key
        rows = arrays["output_rows"]
        valid_rows = [row for row in rows if int(float(row["valid"])) == 1]
        for coord in COORDS:
            values = np.asarray([float(row[f"{coord}_tau_target"]) / arrays["mass"] for row in valid_rows], dtype=float)
            magnitude_values[(subject, coord)].extend(np.abs(values).tolist())
            p99_trial = float(np.percentile(np.abs(values), 99)) if values.size else float("nan")
            trial_magnitude[key].append(p99_trial)
            if (subject, trial, coord) in (("s001", "soccerkick_1", "ankle_angle_l"), ("s006", "soccerkick_3", "ankle_angle_r")):
                source_rows = row_cache[key][1]
                before_values = np.asarray([float(row[f"{coord}_tau_target"]) / arrays["mass"] for row in source_rows], dtype=float)
                rounded = np.round(values, 2)
                target_plateau = -7.7 if subject == "s001" else -6.0
                plateau_frames = int(np.count_nonzero(rounded == target_plateau))
                soccer_rows.append({"subject": subject, "trial": trial, "coord": coord, "before_min_nmkg": float(np.min(before_values)), "after_min_nmkg": float(np.min(values)) if values.size else float("nan"), "after_p99_abs_nmkg": p99_trial, "frames_at_reported_plateau_rounded": plateau_frames})

    # Per-subject p99 moments and trial outliers.
    magnitude_summary: list[dict[str, Any]] = []
    for (subject, coord), values in magnitude_values.items():
        p99 = float(np.percentile(values, 99)) if values else float("nan")
        magnitude_summary.append({"subject": subject, "coord": coord, "p99_abs_tau_nmkg": p99, "pass_lt_5": int(p99 < 5.0)})
    _write_rows(logs / "moment_magnitude.csv", ["subject", "coord", "p99_abs_tau_nmkg", "pass_lt_5"], magnitude_summary)

    force_delta_max = max((float(row["pelvis_force_max_abs_delta_nm"]) for row in pelvis_rows), default=float("inf"))
    moment_decreased_trials = sum(float(row["pelvis_moment_rms_after_nm"]) < float(row["pelvis_moment_rms_before_nm"]) for row in pelvis_rows if math.isfinite(float(row["pelvis_moment_rms_before_nm"])) and math.isfinite(float(row["pelvis_moment_rms_after_nm"])))
    _write_rows(logs / "pelvis_residuals.csv", ["subject", "trial", "segment", "pelvis_force_max_abs_delta_nm", "pelvis_moment_rms_before_nm", "pelvis_moment_rms_after_nm"], pelvis_rows)

    activity_rows: list[dict[str, Any]] = []
    for (activity, side), vectors in offset_activity.items():
        values = np.asarray(vectors, dtype=float)
        if values.size:
            activity_rows.append({"activity": activity, "side": side, "median_offset_x_cm": float(np.median(values[:, 0]) * 100.0), "median_offset_y_cm": float(np.median(values[:, 1]) * 100.0), "n": int(len(values))})
    _write_rows(logs / "cop_residual_by_activity.csv", ["activity", "side", "median_offset_x_cm", "median_offset_y_cm", "n"], activity_rows)

    checks = {
        "1_tcode_and_token_integrity": {"status": "pass" if tcode_max < 1e-6 and mot_segments_verified > 0 else "fail", "blocking": True, "trials": len(audit_rows), "max_abs_delta_m": tcode_max, "measured_mot_segments_verified": mot_segments_verified, "valid_cop_points_shifted": valid_mot_points},
        "2_cop_to_own_foot_distance": {"status": "pass" if distance_pass_trials >= 200 else "fail", "blocking": True, "trials_passed": distance_pass_trials, "trials_total": 206, "thresholds": {"median_cm_lt": 5.0, "p90_cm_lt": 10.0}, "failures": distance_failures},
        "3_side_by_segment_side": {"status": "pass" if not side_blockers else "fail", "blocking": True, "segment_side_groups_scored": len([row for row in side_after_rows if int(row["usable_stance_frames"]) >= 50]), "threshold_percent_gt": 95.0, "failures": side_blockers, "proposed_valid_frames_masked": proposed_valid, "mask_fraction_of_frames": mask_fraction},
        "4_swing_target_unchanged": {"status": "pass" if swing_max < 1e-6 else "fail", "blocking": True, "max_abs_delta_nm": swing_max, "samples": len(swing_deltas)},
        "5_moment_magnitude": {"status": "reported", "blocking": False, "per_subject_coordinate": magnitude_summary, "trials_with_p99_over_4_nmkg": [f"{s}/{t}" for (s, t), values in trial_magnitude.items() if any(value > 4.0 for value in values)]},
        "6_pelvis_residuals": {"status": "reported", "blocking": False, "max_force_delta_nm": force_delta_max, "force_unchanged": force_delta_max < 1e-6, "segments": len(pelvis_rows), "trial_segments_moment_rms_decreased": moment_decreased_trials},
        "7_integrity": {"status": "pass" if not integrity_failures else "fail", "blocking": True, "trials": len(row_cache), "frames": total_frames, "expected_frames": EXPECTED_FRAMES, "valid_frames_before": total_source_valid, "valid_frames_after": total_output_valid, "failures": integrity_failures},
        "8_tau_hat_copied": {"status": "pass" if not hat_hash_failures else "fail", "blocking": True, "column_failures": hat_hash_failures},
        "9_soccerkick_ankle_plateau": {"status": "reported", "blocking": False, "trials": soccer_rows},
    }
    blocking_failures = [name for name, result in checks.items() if result["blocking"] and result["status"] != "pass"]
    report = {
        "protocol": "nomuscles_copfix_tcode",
        "source_root": str(source_root), "output_root": str(output_root),
        "side_invalid_valid_frames": proposed_valid, "side_invalid_fraction": mask_fraction,
        "checks": checks, "blocking_failures": blocking_failures,
        "cohort_status": "accepted" if not blocking_failures else "blocked",
    }
    logs.mkdir(parents=True, exist_ok=True)
    temp = logs / "acceptance_checks.json.partial"
    temp.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(logs / "acceptance_checks.json")

    # Tau-hat validation is written from the stored aligned target table and force signals.
    # Lag is estimated per subject (and again with soccerkick excluded) over -50..50 frames.
    tau_rows: list[dict[str, Any]] = []
    tau_samples: dict[tuple[str, str, str, int, int], dict[str, list[np.ndarray]]] = defaultdict(lambda: {"target": [], "prediction": []})
    tau_denominators: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    lag_rows: list[dict[str, Any]] = []
    tau_subject_records: dict[tuple[str, int], list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]] = defaultdict(list)
    for (subject, trial), arrays in subject_trial_arrays.items():
        for exclude_soccer in (0, 1):
            if exclude_soccer and trial.startswith("soccerkick"):
                continue
            tau_subject_records[(subject, exclude_soccer)].append((trial, subject, arrays["measured_total"], arrays["estimated_total"], arrays["usable"], arrays["times"], arrays["segments"], np.asarray([int(float(r["side_invalid"])) == 0 for r in arrays["output_rows"]]), arrays["mass"]))
    lag_lookup: dict[tuple[str, int], int] = {}
    for group_key, records in tau_subject_records.items():
        lag_scores: list[tuple[float, int, int]] = []
        for lag in range(-50, 51):
            correlations, weights = [], []
            for _trial, _subject, measured, estimated, valid, _times, segments, unmasked, _mass in records:
                keep = valid & unmasked
                for segment in np.unique(segments):
                    ids = np.flatnonzero(segments == segment)
                    if ids.size < 5:
                        continue
                    local_measured, local_estimated = measured[ids].copy(), estimated[ids].copy()
                    local_keep = keep[ids]
                    local_measured[~local_keep] = np.nan
                    local_estimated[~local_keep] = np.nan
                    corr, n = _lag_correlation(local_measured, local_estimated, lag)
                    if math.isfinite(corr):
                        correlations.append(corr * n)
                        weights.append(n)
            score = sum(correlations) / sum(weights) if weights else float("nan")
            lag_scores.append((score, lag, sum(weights)))
        finite = [row for row in lag_scores if math.isfinite(row[0])]
        chosen = max(finite, key=lambda row: row[0])[1] if finite else 0
        lag_lookup[group_key] = chosen
    for (subject, trial), arrays in subject_trial_arrays.items():
        for exclude_soccer in (0, 1):
            if exclude_soccer and trial.startswith("soccerkick"):
                continue
            subject_lag = lag_lookup[(subject, exclude_soccer)]
            force_keep = arrays["usable"] & np.asarray([int(float(row["side_invalid"])) == 0 for row in arrays["output_rows"]])
            per_trial_scores: list[tuple[float, int]] = []
            for candidate_lag in range(-50, 51):
                correlations, weights = [], []
                for segment in np.unique(arrays["segments"]):
                    ids = np.flatnonzero(arrays["segments"] == segment)
                    if ids.size < 5:
                        continue
                    measured = arrays["measured_total"][ids].copy()
                    estimated = arrays["estimated_total"][ids].copy()
                    keep = force_keep[ids]
                    measured[~keep] = np.nan
                    estimated[~keep] = np.nan
                    corr, n = _lag_correlation(measured, estimated, candidate_lag)
                    if math.isfinite(corr):
                        correlations.append(corr * n)
                        weights.append(n)
                per_trial_scores.append((sum(correlations) / sum(weights) if weights else float("nan"), candidate_lag))
            finite_trial_scores = [value for value in per_trial_scores if math.isfinite(value[0])]
            trial_lag = max(finite_trial_scores, default=(float("nan"), subject_lag), key=lambda value: value[0])[1]
            lag_rows.append({"subject": subject, "trial": trial, "con_soccerkick": 1 - exclude_soccer, "lag_frames_trial": trial_lag, "lag_frames_subject": subject_lag})
            lag = subject_lag
            rows = arrays["output_rows"]
            valid = arrays["usable"] & np.asarray([int(float(row["side_invalid"])) == 0 for row in rows])
            for side in SIDES:
                fz = arrays["fz"][side]
                stance = valid & (fz > FZ_VALIDATION_N)
                phases = {"stance": stance, "swing": valid & ~stance}
                for coord in COORDS:
                    if not coord.endswith(f"_{side}"):
                        continue
                    target = np.asarray([float(row[f"{coord}_tau_target"]) / arrays["mass"] for row in rows])
                    hat = np.asarray([float(row[f"{coord}_tau_hat"]) / arrays["mass"] for row in rows])
                    valid_target = target[valid]
                    include_soccer = 1 - exclude_soccer
                    if valid_target.size:
                        tau_denominators[(subject, coord, include_soccer)].append(valid_target)
                    for lag_corrected in (0, 1):
                        indices = np.flatnonzero(valid)
                        pred_idx = indices + lag if lag_corrected else indices
                        keep = (pred_idx >= 0) & (pred_idx < len(hat))
                        if lag_corrected:
                            keep &= arrays["segments"][np.clip(pred_idx, 0, len(hat) - 1)] == arrays["segments"][indices]
                        indices, pred_idx = indices[keep], pred_idx[keep]
                        for phase, phase_mask in phases.items():
                            selected = phase_mask[indices]
                            if np.any(selected):
                                sample_key = (subject, coord, phase, lag_corrected, include_soccer)
                                tau_samples[sample_key]["target"].append(target[indices[selected]])
                                tau_samples[sample_key]["prediction"].append(hat[pred_idx[selected]])
    for (subject, coord, phase, lag_corrected, include_soccer), samples in tau_samples.items():
        target_values = np.concatenate(samples["target"])
        prediction_values = np.concatenate(samples["prediction"])
        denominator_values = np.concatenate(tau_denominators[(subject, coord, include_soccer)])
        denominator = float(np.std(denominator_values, ddof=1)) if denominator_values.size > 1 else 0.0
        rmse, corr, nrmse, n = _finite_metrics(target_values, prediction_values, denominator)
        lag = lag_lookup[(subject, 1 - include_soccer)] if lag_corrected else 0
        tau_rows.append({"subject": subject, "coord": coord, "phase": phase, "lag_frames": lag, "lag_corrected": lag_corrected, "rmse_nmkg": rmse, "pearson_r": corr, "nrmse": nrmse, "n_frames": n, "con_soccerkick": include_soccer})
    _write_rows(logs / "tau_hat_lags_by_trial.csv", ["subject", "trial", "con_soccerkick", "lag_frames_trial", "lag_frames_subject"], lag_rows)
    grouped_tau: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in tau_rows:
        grouped_tau[(row["coord"], row["phase"], row["lag_corrected"], row["con_soccerkick"])].append(row)
    for (coord, phase, lag_corrected, include_soccer), records in grouped_tau.items():
        if len(records) != 6:
            continue
        summary: dict[str, Any] = {
            "subject": "mean_6_subjects", "coord": coord, "phase": phase,
            "lag_frames": int(round(float(np.mean([row["lag_frames"] for row in records])))),
            "lag_corrected": lag_corrected, "n_frames": sum(int(row["n_frames"]) for row in records),
            "con_soccerkick": include_soccer,
        }
        for metric in ("rmse_nmkg", "pearson_r", "nrmse"):
            values = np.asarray([float(row[metric]) for row in records], dtype=float)
            values = values[np.isfinite(values)]
            mean = float(np.mean(values)) if values.size else float("nan")
            half_width = 2.570581835 * float(np.std(values, ddof=1)) / math.sqrt(values.size) if values.size > 1 else float("nan")
            summary[metric] = mean
            summary[f"{metric}_ci95_low"] = mean - half_width if math.isfinite(half_width) else float("nan")
            summary[f"{metric}_ci95_high"] = mean + half_width if math.isfinite(half_width) else float("nan")
        tau_rows.append(summary)
    tau_fields = ["subject", "coord", "phase", "lag_frames", "lag_corrected", "rmse_nmkg", "rmse_nmkg_ci95_low", "rmse_nmkg_ci95_high", "pearson_r", "pearson_r_ci95_low", "pearson_r_ci95_high", "nrmse", "nrmse_ci95_low", "nrmse_ci95_high", "n_frames", "con_soccerkick"]
    _write_rows(logs / "tau_hat_validation.csv", tau_fields, tau_rows)

    def fmt(value: Any, digits: int = 3) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"

    criteria_rows = [
        ("1 · T_code e token MOT", f"max Δ={tcode_max:.3g} m; {mot_segments_verified} segmenti MOT verificati", checks["1_tcode_and_token_integrity"]["status"]),
        ("2 · CoP–piede proprio", f"{distance_pass_trials}/206 trial; median<5 cm, p90<10 cm", checks["2_cop_to_own_foot_distance"]["status"]),
        ("3 · Lato segmento×lato", f"{len(side_after_rows)} gruppi; {len(side_blockers)} fallimenti dopo maschera", checks["3_side_by_segment_side"]["status"]),
        ("4 · Swing invariato", f"max Δτ={fmt(swing_max, 6)} N·m", checks["4_swing_target_unchanged"]["status"]),
        ("5 · Magnitudine τ", f"max p99 soggetto/coord={fmt(max((r['p99_abs_tau_nmkg'] for r in magnitude_summary), default=float('nan')))} N·m/kg", "riportato"),
        ("6 · Residui bacino", f"max Δforza={fmt(force_delta_max, 6)} N; RMS momento diminuito in {moment_decreased_trials}/{len(pelvis_rows)} segmento", "riportato"),
        ("7 · Integrità", f"{len(row_cache)}/206 trial; {total_frames}/{EXPECTED_FRAMES} frame", checks["7_integrity"]["status"]),
        ("8 · τ̂ copiato", f"{len(hat_hash_failures)} mismatch hash/colonna", checks["8_tau_hat_copied"]["status"]),
        ("9 · Plateau soccerkick", "; ".join(f"{r['subject']}: {r['frames_at_reported_plateau_rounded']} frame al plateau" for r in soccer_rows), "riportato"),
    ]
    criteria_md = "| Criterio | Misura | Esito |\n|---|---|---|\n" + "\n".join(
        f"| {name} | {value} | {status} |" for name, value, status in criteria_rows
    )
    fail_trials = ", ".join(f"{row['subject']}/{row['trial']}" for row in distance_failures) or "nessuno"
    old_failures = sum(int(row["usable_stance_frames"]) >= 50 and float(row["old_criterion_before_mask_percent"]) <= 95.0 for row in side_after_rows)
    new_pre_mask_failures = sum(int(row["usable_stance_frames"]) >= 50 and float(row["new_criterion_before_mask_percent"]) <= 95.0 for row in side_after_rows)
    new_post_mask_failures = len(side_blockers)
    tau_summary = [row for row in tau_rows if row.get("subject") == "mean_6_subjects"]
    main_tau = [row for row in tau_summary if row["phase"] == "stance" and row["lag_corrected"] == 1 and row["con_soccerkick"] == 1]
    main_rmse = float(np.mean([float(row["rmse_nmkg"]) for row in main_tau])) if main_tau else float("nan")
    main_r = float(np.mean([float(row["pearson_r"]) for row in main_tau])) if main_tau else float("nan")
    main_nrmse = float(np.mean([float(row["nrmse"]) for row in main_tau])) if main_tau else float("nan")
    tau_md = "| Coord. | Fase | Lag corretto | Soccerkick incluso | RMSE, media [IC95%] | r, media [IC95%] | nRMSE, media [IC95%] |\n|---|---|---:|---:|---:|---:|---:|"
    for row in tau_summary:
        tau_md += "\n| {coord} | {phase} | {lag} | {soccer} | {rmse} [{rlo}, {rhi}] | {corr} [{clo}, {chi}] | {nrmse} [{nlo}, {nhi}] |".format(
            coord=row["coord"], phase=row["phase"], lag=row["lag_corrected"], soccer=row["con_soccerkick"],
            rmse=fmt(row["rmse_nmkg"]), rlo=fmt(row.get("rmse_nmkg_ci95_low")), rhi=fmt(row.get("rmse_nmkg_ci95_high")),
            corr=fmt(row["pearson_r"]), clo=fmt(row.get("pearson_r_ci95_low")), chi=fmt(row.get("pearson_r_ci95_high")),
            nrmse=fmt(row["nrmse"]), nlo=fmt(row.get("nrmse_ci95_low")), nhi=fmt(row.get("nrmse_ci95_high")),
        )
    _write_rows(logs / "tcode_recomputed_by_trial.csv", list(audit_rows[0]), audit_rows)
    (output_root / "README.md").write_text(
        "# GroundLink measured-CoP targets, T_code correction\n\n"
        "This tree was generated from the read-only `torque_targets_12hz_nomuscles` input. The measured CoP receives the per-trial translation `T_code = [0, 0.01 - min(marker_Y), 0]`, reconstructed from the saved MoSh++ source and SMPL-X marker map. Only valid-contact `px`/`py` tokens in copied measured-force MOT files change. Measured target ID was rerun with the saved no-muscle setups; `estimated_id/` and all `*_tau_hat` columns are copied unchanged.\n\n"
        "The output still depends on `../pipeline/` for the scaled model and IK coordinate files referenced by each setup. Do not edit those source artifacts. The older `_copfix` B2 candidate aligns CoP to the foot by construction and must not be used.\n",
        encoding="utf-8",
    )
    report_path = output_root / "report.md"
    report_path.write_text(
        "# GroundLink CoP Phase C: T_code\n\n"
        f"{checks['1_tcode_and_token_integrity']['status'].upper()} criteri bloccanti: {len(blocking_failures)} fallimenti. Coorte: {len(row_cache)}/206 trial, {total_frames}/{EXPECTED_FRAMES} frame. Maschera: {proposed_valid}/{total_source_valid} frame validi ({mask_fraction:.3%}).\n"
        f"Distanza CoP-piede: {distance_pass_trials}/206 trial entro soglia. τ̂ vs τ in stance, lag corretto, con soccerkick: RMSE {fmt(main_rmse)} N·m/kg, r {fmt(main_r)}, nRMSE {fmt(main_nrmse)}.\n"
        f"Scarto max T_code: {tcode_max:.3g} m; `τ̂` invariato: {len(hat_hash_failures)} differenze.\n\n"
        "## Verifica degli assi\n\n"
        "La coordinata verticale dei marker salvati è Z: il minimo trial-per-trial ha mediana 2,7375 cm e range 0,43–4,895 cm; tutti i 206 valori sono in `logs/axis_min_vertical_by_trial.csv`. Y è orizzontale e il minimo Y dei TRC è esattamente 1 cm in tutti i trial. Il codice usa `[0, 0, -g]` come gravità (`src/opensim_batch_dynamics/inverse_dynamics_no_grf.py:404`) e calcola l'altezza dei contatti lungo l'asse verticale (`:570–581`), quindi τ̂ non usa un asse orizzontale come verticale. L'offset Y nasce da `scripts/run_amass_to_bsm_csv.py:316` e `src/opensim_batch_dynamics/trc_export.py:103–104`; `_run_single_trial_pipeline` è condiviso con AMASS e applica lo stesso offset anche lì. Il CSV B1-bis `code_transform_by_trial.csv` è riprodotto in `logs/tcode_recomputed_by_trial.csv`.\n\n"
        "## Criteri di accettazione\n\n"
        + criteria_md + "\n\n"
        + f"Trial fuori soglia per distanza CoP-piede: {fail_trials}. Valori completi e motivazioni in `logs/acceptance_checks.json` e `logs/cop_offset_after.csv`.\n\n"
        + "## Maschera di lato\n\n"
        + f"Il criterio precedente falliva in {old_failures} gruppi segmento×lato; quello rivisto prima della maschera in {new_pre_mask_failures}, dopo in {new_post_mask_failures}. I casi risolti dipendono dal piede opposto sollevato: è concorrente solo quando min(Z calcaneus, Z toes) < 5 cm. Sono state applicate {proposed_valid} righe H1 valide; intervalli e motivi sono in `logs/side_invalid_intervals.csv`.\n\n"
        + "## Residui del bacino e del CoP per attività\n\n"
        + f"Le forze generalizzate del bacino hanno Δ massimo {fmt(force_delta_max, 6)} N; l'RMS dei momenti è diminuito in {moment_decreased_trials}/{len(pelvis_rows)} segmento. Dettagli in `logs/pelvis_residuals.csv`. Residui mediani CoP−midfoot per attività/lato: `logs/cop_residual_by_activity.csv`.\n\n"
        + "## Validazione τ̂ vs τ\n\n"
        + f"Riferimento pre-correzione (non valido): RMSE 0,687 N·m/kg, r 0,389, nRMSE 1,03. La tabella riporta media e IC95% sui sei soggetti per coordinata, stance/swing, con e senza lag e con/senza soccerkick. Lag per soggetto e trial: `logs/tau_hat_lags_by_trial.csv`; dati completi: `logs/tau_hat_validation.csv`.\n\n"
        + tau_md + "\n\n"
        + "## Note per gli autori della pipeline\n\n"
        + "Il difetto è l'uso di `vertical_offset` sulla coordinata Y orizzontale. La radice IK di `s001/lambadadance_3/segment_3` è azzerata a monte; è l'unico segmento fra i 541 controllati, non risultano altri segmenti GroundLink con pelvis_tx/ty/tz tutti a zero. Quel segmento è mascherato su entrambi i lati. L'offset Y è generato dallo stesso runner anche per AMASS; correggere la pipeline è necessario per future estrazioni, ma questa consegna non modifica `pipeline/`.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--tcode-audit", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.source_dir, args.output_dir, args.index, args.tcode_audit)
    print(json.dumps({"cohort_status": report["cohort_status"], "blocking_failures": report["blocking_failures"], "side_invalid_valid_frames": report["side_invalid_valid_frames"]}, indent=2))
    return 0 if report["cohort_status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""E4: compare estimated-GRF ID against ID from measured GroundLink GRF."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from opensim_batch_dynamics.inverse_dynamics_no_grf import (  # noqa: E402
    export_torque_csv_from_id_sto, parse_mot, run_inverse_dynamics_with_measured_grf,
)
from opensim_batch_dynamics.segments import (  # noqa: E402
    discover_segment_mot_paths,
    merge_csv_segments,
)
from batch_common import is_quality_pass  # noqa: E402
from validate_groundlink import discover_pairs, load_groundlink_force  # noqa: E402

JOINTS = {"hip": "hip_", "knee": "knee_angle", "ankle": "ankle_angle"}


def _csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def _residual_metrics(
    data: dict[str, np.ndarray],
    mass: float,
    height: float,
    valid: np.ndarray,
) -> dict[str, float]:
    out: dict[str, float] = {}
    force = np.column_stack([data[f"pelvis_t{axis}_tau"] for axis in "xyz"])[valid]
    moment = np.column_stack(
        [data[f"pelvis_{axis}_tau"] for axis in ("tilt", "list", "rotation")]
    )[valid]
    if force.size == 0 or moment.size == 0:
        for name in (
            "residual_force_pct_bw",
            "residual_force_median_pct_bw",
            "residual_force_p95_pct_bw",
            "residual_force_max_pct_bw",
            "residual_force_x_rms_pct_bw",
            "residual_force_y_rms_pct_bw",
            "residual_force_z_rms_pct_bw",
            "residual_moment_pct_bwh",
            "residual_moment_p95_pct_bwh",
            "residual_moment_max_pct_bwh",
        ):
            out[name] = math.nan
        return out
    bw = mass * 9.81
    bwh = bw * height
    force_norm = np.linalg.norm(force, axis=1)
    moment_norm = np.linalg.norm(moment, axis=1)
    out["residual_force_pct_bw"] = float(np.sqrt(np.mean(force_norm ** 2)) / bw * 100)
    out["residual_force_median_pct_bw"] = float(np.median(force_norm) / bw * 100)
    out["residual_force_p95_pct_bw"] = float(np.percentile(force_norm, 95) / bw * 100)
    out["residual_force_max_pct_bw"] = float(np.max(force_norm) / bw * 100)
    for axis_index, axis in enumerate("xyz"):
        out[f"residual_force_{axis}_rms_pct_bw"] = float(
            np.sqrt(np.mean(force[:, axis_index] ** 2)) / bw * 100
        )
    out["residual_moment_pct_bwh"] = float(
        np.sqrt(np.mean(moment_norm ** 2)) / bwh * 100
    )
    out["residual_moment_p95_pct_bwh"] = float(np.percentile(moment_norm, 95) / bwh * 100)
    out["residual_moment_max_pct_bwh"] = float(np.max(moment_norm) / bwh * 100)
    return out


def _trial_metrics(
    estimated: Path,
    reference: Path,
    mass: float,
    height: float,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    est, ref = _csv(estimated), _csv(reference)
    out: dict[str, float] = {}
    scale = mass * 9.81 * height
    est_time = est["time"]
    ref_time = ref["time"]
    ref_dt = np.diff(ref_time)
    ref_dt = ref_dt[np.isfinite(ref_dt) & (ref_dt > 0.0)]
    time_tolerance = 1.5 * float(np.median(ref_dt)) if ref_dt.size else 1e-8
    insertion = np.searchsorted(ref_time, est_time)
    left = np.clip(insertion - 1, 0, len(ref_time) - 1)
    right = np.clip(insertion, 0, len(ref_time) - 1)
    nearest_distance = np.minimum(
        np.abs(est_time - ref_time[left]),
        np.abs(est_time - ref_time[right]),
    )
    overlap = nearest_distance <= time_tolerance
    if valid_mask is None:
        valid = overlap
    else:
        if valid_mask.shape != est_time.shape:
            raise ValueError("valid_mask must match the estimated torque time vector.")
        valid = overlap & valid_mask
    for joint, prefix in JOINTS.items():
        columns = [name for name in est if name.startswith(prefix) and name in ref]
        errors = [
            (est[name][valid] - np.interp(est_time[valid], ref_time, ref[name])) ** 2
            for name in columns
        ]
        out[f"{joint}_rmse_mgh"] = (
            float(np.sqrt(np.mean(np.concatenate(errors))) / scale)
            if errors and np.any(valid)
            else math.nan
        )
    out.update(_residual_metrics(est, mass, height, valid))
    return out


def _reference_valid_mask(
    times: np.ndarray,
    measured_grf: np.ndarray,
    force_target_csv: Path,
    mass: float,
    threshold_bw: float,
) -> np.ndarray:
    """Select frames where measured plates capture the required external force."""
    if not force_target_csv.exists():
        raise FileNotFoundError(
            f"Reference validity requires the COM force target: {force_target_csv}"
        )
    target = _csv(force_target_csv)
    target_force = np.column_stack(
        [np.interp(times, target["time"], target[f"target_force_{axis}"]) for axis in "xyz"]
    )
    measured_total = np.sum(measured_grf, axis=1)
    mismatch = np.linalg.norm(target_force - measured_total, axis=1)
    valid = mismatch <= threshold_bw * mass * 9.81
    # The estimated-GRF model is foot-contact-only.  A frame in which the
    # contact allocator reports no active foot is therefore not a valid
    # reference comparison, even when both measured and target forces happen
    # to be close to zero (the common airborne-flight false positive).
    if "contact_activation_sum" in target:
        activation = np.interp(times, target["time"], target["contact_activation_sum"])
        valid &= activation > 0.5
    return valid


def _mass_height(final_csv: Path) -> tuple[float, float]:
    with final_csv.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    return float(row["subject_mass_kg"]), float(row["subject_height_m"])


def _run_trial(
    pair,
    pipeline_dir: Path,
    output_dir: Path,
    reference_balance_threshold_bw: float = 0.2,
) -> list[dict[str, str | float]]:
    # The validator marks trials whose external-contact assumptions are not
    # satisfied (for example balletsmalljump_1, which contains unsupported
    # flight phases).  Keep them out of the E4 comparison even if an old or
    # newly regenerated CSV happens to pass the generic pipeline quality
    # gate: a foot-only GRF cannot explain an airborne non-ballistic root.
    if getattr(pair, "exclusion", ""):
        return [{
            "subject": pair.subject,
            "trial": pair.trial,
            "method": "eligibility",
            "metric": f"excluded_{pair.exclusion}",
            "value": 1.0,
        }]
    final_csv = pipeline_dir / f"{pair.motion_path.stem}.csv"
    artifact_candidates = list(pipeline_dir.glob(f"{pair.motion_path.stem}_*/results"))
    # A retry leaves the previous AddBiomechanics run beside the new one.
    # Select the newest artifact rather than relying on filesystem iteration
    # order, otherwise E4 can compare against a stale ID result.
    artifact = max(artifact_candidates, key=lambda path: path.stat().st_mtime, default=None)
    if artifact is None or not final_csv.exists() or pair.force_path is None:
        return []
    if not is_quality_pass(final_csv):
        return [{
            "subject": pair.subject,
            "trial": pair.trial,
            "method": "eligibility",
            "metric": "excluded_pipeline_quality_gate",
            "value": 1.0,
        }]
    if pair.trial.lower().startswith("chair"):
        return [{
            "subject": pair.subject,
            "trial": pair.trial,
            "method": "eligibility",
            "metric": "excluded_unmodelled_external_contact",
            "value": 1.0,
        }]
    trial = artifact.parent.name
    model = artifact / "Models/match_markers_but_ignore_physics.osim"
    ik = artifact / "IK" / f"{trial}_ik.mot"
    estimated_sto = artifact / "ID_estimatedGRF" / f"{trial}_id_estimatedGRF.sto"
    if not all(path.exists() for path in (model, ik, estimated_sto)):
        return []
    labels, rows = parse_mot(ik); time_idx = labels.index("time")
    full_times = np.asarray([row[time_idx] for row in rows], dtype=float)
    cop_source, grf_source = load_groundlink_force(pair.force_path)
    source_t = np.arange(grf_source.shape[0]) / 250.0
    out = output_dir / pair.subject / pair.motion_path.stem
    out.mkdir(parents=True, exist_ok=True)
    segment_paths = discover_segment_mot_paths(ik)
    if len(segment_paths) == 1:
        grf = np.stack([
            np.interp(full_times, source_t, grf_source[:, foot, axis])
            for foot in range(2) for axis in range(3)
        ], axis=1).reshape(len(full_times), 2, 3)
        cop = np.stack([
            np.interp(full_times, source_t, cop_source[:, foot, axis])
            for foot in range(2) for axis in range(3)
        ], axis=1).reshape(len(full_times), 2, 3)
        _, reference_sto, *_ = run_inverse_dynamics_with_measured_grf(
            model, ik, out, trial, full_times, grf, cop
        )
        reference_csv, *_ = export_torque_csv_from_id_sto(
            reference_sto, model, out / "reference_torque.csv", full_times
        )
    else:
        reference_parts: list[Path] = []
        for segment_index, segment_path in enumerate(segment_paths):
            segment_labels, segment_rows = parse_mot(segment_path)
            segment_time_idx = segment_labels.index("time")
            segment_times = np.asarray(
                [row[segment_time_idx] for row in segment_rows], dtype=float
            )
            segment_grf = np.stack([
                np.interp(segment_times, source_t, grf_source[:, foot, axis])
                for foot in range(2) for axis in range(3)
            ], axis=1).reshape(len(segment_times), 2, 3)
            segment_cop = np.stack([
                np.interp(segment_times, source_t, cop_source[:, foot, axis])
                for foot in range(2) for axis in range(3)
            ], axis=1).reshape(len(segment_times), 2, 3)
            segment_out = out / "segments" / f"segment_{segment_index}"
            segment_trial = f"{trial}_segment_{segment_index}"
            _, segment_sto, *_ = run_inverse_dynamics_with_measured_grf(
                model,
                segment_path,
                segment_out,
                segment_trial,
                segment_times,
                segment_grf,
                segment_cop,
            )
            segment_csv, *_ = export_torque_csv_from_id_sto(
                segment_sto,
                model,
                segment_out / "reference_torque.csv",
                segment_times,
            )
            reference_parts.append(segment_csv)
        reference_csv, _ = merge_csv_segments(
            reference_parts, out / "reference_torque.csv"
        )
    estimated_csv, *_ = export_torque_csv_from_id_sto(
        estimated_sto, model, out / "estimated_torque.csv", None
    )
    mass, height = _mass_height(final_csv)
    estimated = _csv(estimated_csv)
    metric_times = estimated["time"]
    measured_grf = np.stack([
        np.interp(metric_times, source_t, grf_source[:, foot, axis])
        for foot in range(2) for axis in range(3)
    ], axis=1).reshape(len(metric_times), 2, 3)
    force_target_csv = artifact / "ID_estimatedGRF" / f"{trial}_contact_wrenches_estimated.csv"
    valid_mask = _reference_valid_mask(
        metric_times,
        measured_grf,
        force_target_csv,
        mass,
        reference_balance_threshold_bw,
    )
    metrics = _trial_metrics(estimated_csv, reference_csv, mass, height, valid_mask)
    rows_out = [
        {
            "subject": pair.subject,
            "trial": pair.trial,
            "method": "pipeline",
            "metric": name,
            "value": value,
        }
        for name, value in metrics.items()
        if name.endswith("mgh")
    ]
    rows_out += [
        {
            "subject": pair.subject,
            "trial": pair.trial,
            "method": "pipeline",
            "metric": name,
            "value": value,
        }
        for name, value in metrics.items()
        if name.startswith("residual")
    ]
    reference_times = _csv(reference_csv)["time"]
    reference_valid_mask = (
        np.interp(reference_times, metric_times, valid_mask.astype(float)) > 0.5
    )
    reference_valid_mask &= (
        (reference_times >= metric_times[0]) & (reference_times <= metric_times[-1])
    )
    reference_metrics = _trial_metrics(
        reference_csv, reference_csv, mass, height, reference_valid_mask
    )
    rows_out += [
        {
            "subject": pair.subject,
            "trial": pair.trial,
            "method": "reference_ID",
            "metric": name,
            "value": value,
        }
        for name, value in reference_metrics.items()
        if name.startswith("residual")
    ]
    rows_out.append({
        "subject": pair.subject,
        "trial": pair.trial,
        "method": "reference_ID",
        "metric": "valid_frame_fraction",
        "value": float(np.mean(valid_mask)),
    })
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument(
        "--reference-force-balance-threshold-bw",
        type=float,
        default=0.2,
        help="Maximum COM-force mismatch for a measured-GRF frame to be eligible.",
    )
    args = parser.parse_args()
    motion_root = args.dataset_root / "moshpp" / args.subject
    force_root = args.dataset_root / "force" / args.subject
    records = []
    for pair in discover_pairs(motion_root, force_root, args.subject):
        # Run the eligibility path for excluded trials as well, so the
        # report records why they were not compared instead of silently
        # dropping them (e.g. balletsmalljump_1 flight phases).
        if pair.exclusion or pair.force_path:
            records.extend(_run_trial(
                pair,
                args.pipeline_root / args.subject,
                args.output_dir,
                args.reference_force_balance_threshold_bw,
            ))
    report = args.output_dir / args.subject / "metrics_per_trial.csv"
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "trial", "method", "metric", "value"]); writer.writeheader(); writer.writerows(records)
    print(f"{report}: {len(records)} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

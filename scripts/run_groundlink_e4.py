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
from validate_groundlink import discover_pairs, load_groundlink_force  # noqa: E402

JOINTS = {"hip": "hip_", "knee": "knee_angle", "ankle": "ankle_angle"}


def _csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def _trial_metrics(estimated: Path, reference: Path, mass: float, height: float) -> dict[str, float]:
    est, ref = _csv(estimated), _csv(reference)
    out: dict[str, float] = {}
    scale = mass * 9.81 * height
    for joint, prefix in JOINTS.items():
        columns = [name for name in est if name.startswith(prefix) and name in ref]
        out[f"{joint}_rmse_mgh"] = float(np.sqrt(np.mean(np.concatenate([(est[name] - ref[name]) ** 2 for name in columns]))) / scale)
    out["residual_force_pct_bw"] = float(np.sqrt(np.mean(np.concatenate([est[f"pelvis_t{axis}_tau"] ** 2 for axis in "xyz"]))) / (mass * 9.81) * 100)
    out["residual_moment_pct_bwh"] = float(np.sqrt(np.mean(np.concatenate([est[f"pelvis_{axis}_tau"] ** 2 for axis in ("tilt", "list", "rotation")]))) / scale * 100)
    return out


def _mass_height(final_csv: Path) -> tuple[float, float]:
    with final_csv.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    return float(row["subject_mass_kg"]), float(row["subject_height_m"])


def _run_trial(pair, pipeline_dir: Path, output_dir: Path) -> list[dict[str, str | float]]:
    final_csv = pipeline_dir / f"{pair.motion_path.stem}.csv"
    artifact = next(iter(pipeline_dir.glob(f"{pair.motion_path.stem}_*/results")), None)
    if artifact is None or not final_csv.exists() or pair.force_path is None:
        return []
    trial = artifact.parent.name
    model = artifact / "Models/match_markers_but_ignore_physics.osim"
    ik = artifact / "IK" / f"{trial}_ik.mot"
    estimated_sto = artifact / "ID_estimatedGRF" / f"{trial}_id_estimatedGRF.sto"
    if not all(path.exists() for path in (model, ik, estimated_sto)):
        return []
    labels, rows = parse_mot(ik); time_idx = labels.index("time")
    times = np.asarray([row[time_idx] for row in rows], dtype=float)
    cop, grf = load_groundlink_force(pair.force_path)
    source_t = np.arange(grf.shape[0]) / 250.0
    grf = np.stack([np.interp(times, source_t, grf[:, foot, axis]) for foot in range(2) for axis in range(3)], axis=1).reshape(len(times), 2, 3)
    cop = np.stack([np.interp(times, source_t, cop[:, foot, axis]) for foot in range(2) for axis in range(3)], axis=1).reshape(len(times), 2, 3)
    out = output_dir / pair.subject / pair.motion_path.stem
    out.mkdir(parents=True, exist_ok=True)
    _, reference_sto, *_ = run_inverse_dynamics_with_measured_grf(model, ik, out, trial, times, grf, cop)
    estimated_csv, *_ = export_torque_csv_from_id_sto(estimated_sto, model, out / "estimated_torque.csv", times)
    reference_csv, *_ = export_torque_csv_from_id_sto(reference_sto, model, out / "reference_torque.csv", times)
    mass, height = _mass_height(final_csv)
    metrics = _trial_metrics(estimated_csv, reference_csv, mass, height)
    rows_out = [{"subject": pair.subject, "trial": pair.trial, "method": "pipeline", "metric": name, "value": value} for name, value in metrics.items() if name.endswith("mgh")]
    rows_out += [{"subject": pair.subject, "trial": pair.trial, "method": "pipeline", "metric": name, "value": value} for name, value in metrics.items() if name.startswith("residual")]
    reference_metrics = _trial_metrics(reference_csv, reference_csv, mass, height)
    rows_out += [{"subject": pair.subject, "trial": pair.trial, "method": "reference_ID", "metric": name, "value": value} for name, value in reference_metrics.items() if name.startswith("residual")]
    return rows_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", required=True)
    args = parser.parse_args()
    motion_root = args.dataset_root / "moshpp" / args.subject
    force_root = args.dataset_root / "force" / args.subject
    records = []
    for pair in discover_pairs(motion_root, force_root, args.subject):
        if pair.force_path and not pair.exclusion:
            records.extend(_run_trial(pair, args.pipeline_root / args.subject, args.output_dir))
    report = args.output_dir / args.subject / "metrics_per_trial.csv"; report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "trial", "method", "metric", "value"]); writer.writeheader(); writer.writerows(records)
    print(f"{report}: {len(records)} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

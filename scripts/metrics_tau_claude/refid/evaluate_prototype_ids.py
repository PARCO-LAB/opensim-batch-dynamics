#!/usr/bin/env python3
"""Score selected Phase D prototype IDs against measured ID targets."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).parents[1]))
from diagnose_modes import read_contact_wrenches
from evaluate_hybrid_ids import (
    COORDINATES, PHASES, TIME_TOLERANCE_S, TRANSITION_WINDOW_S, interpolate_columns, mapped_source_path, read_storage,
)
from groundlink import (
    DEFAULT_SOURCE_FORCE_ROOT, EvaluationConfig as GroundLinkEvaluationConfig,
    _force_path, _load_force_arrays, _lowpass_force,
)


def align_indices(source_times: np.ndarray, target_times: np.ndarray, path: Path) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(source_times[left] - target_times) < np.abs(source_times[right] - target_times)
    indices = np.where(use_left, left, right)
    error = np.abs(source_times[indices] - target_times)
    if np.any(error > TIME_TOLERANCE_S):
        raise ValueError(f"Cannot align {path}; max time difference={error.max():.8g}s")
    return indices


def read_model_mass(path: Path) -> float:
    root = ET.parse(path).getroot()
    masses = [float(node.text) for node in root.findall(".//Body/mass") if node.text and node.text.strip()]
    mass = sum(masses)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError(f"Invalid model mass in {path}")
    return mass


def pearson(target: np.ndarray, estimate: np.ndarray) -> float:
    if len(target) < 3 or np.std(target) < 1e-12 or np.std(estimate) < 1e-12:
        return math.nan
    return float(np.corrcoef(target, estimate)[0, 1])


def phase_labels_from_forces(
    times: np.ndarray, measured_force: np.ndarray, measured_raw: np.ndarray, bw_n: float
) -> np.ndarray:
    contact = measured_force[:, :, 2] > 0.05 * bw_n
    labels = np.full(len(times), "other", dtype=object)
    count = contact.sum(axis=1)
    labels[count == 1] = "single_stance"
    labels[count == 2] = "double_stance"
    labels[np.all(measured_raw[:, :, 2] <= 1.0, axis=1)] = "flight"
    transition = np.zeros(len(times), dtype=bool)
    for side in range(2):
        states = contact[:, side]
        edges = np.flatnonzero(states[1:] != states[:-1]) + 1
        if states.size and states[0]:
            edges = np.r_[0, edges]
        for edge in edges:
            transition |= np.abs(times - times[edge]) <= TRANSITION_WINDOW_S
    labels[transition] = "transition"
    return labels


def signflip_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    observed = abs(float(np.mean(values)))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        extreme += abs(float(np.mean(values * np.asarray(signs)))) >= observed - 1e-12
    return extreme / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--torque-root", type=Path, required=True)
    parser.add_argument("--source-torque-root", required=True)
    parser.add_argument("--force-root", type=Path, required=True)
    parser.add_argument("--source-force-root", default=DEFAULT_SOURCE_FORCE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    force_config = GroundLinkEvaluationConfig(
        torque_root=args.torque_root, force_root=args.force_root, output_dir=args.output.parent,
        source_torque_root=args.source_torque_root, source_force_root=args.source_force_root,
    )
    with args.manifest.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest:
        raise ValueError(f"Empty prototype ID manifest: {args.manifest}")

    samples: dict[tuple[str, str, str, str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    target_cache, metadata_cache, mass_cache, force_cache, wrench_cache, baseline_cache = (
        {}, {}, {}, {}, {}, {}
    )
    variants = sorted({row["variant"] for row in manifest})
    for row in manifest:
        variant, subject, trial, segment = row["variant"], row["subject"], row["trial"], int(row["segment"])
        target_path = Path(row["target_csv"])
        if target_path not in target_cache:
            target_cache[target_path] = pd.read_csv(target_path)
        target_csv = target_cache[target_path]
        target = target_csv[target_csv.segment.astype(int) == segment].reset_index(drop=True)
        times = target.time.to_numpy(dtype=float)
        valid = target.valid.to_numpy(dtype=int) == 1
        metadata_path = target_path.parent / "run_metadata.json"
        if metadata_path not in metadata_cache:
            metadata_cache[metadata_path] = json.loads(metadata_path.read_text())
        metadata = metadata_cache[metadata_path]
        model_path = mapped_source_path(metadata["source_paths"]["model"], args.source_torque_root, args.torque_root)
        pipeline_path = mapped_source_path(metadata["source_paths"]["pipeline_csv"], args.source_torque_root, args.torque_root)
        if pipeline_path not in mass_cache:
            mass_cache[pipeline_path] = float(pd.read_csv(pipeline_path, usecols=["subject_mass_kg"]).subject_mass_kg.iloc[0])
        pipeline_mass = mass_cache[pipeline_path]
        if model_path not in mass_cache:
            mass_cache[model_path] = read_model_mass(model_path)
        mass = mass_cache[model_path]
        bw_n = float(pipeline_mass) * 9.81
        force_path = _force_path(metadata["source_paths"]["measured_force"], force_config)
        if force_path not in force_cache:
            force_raw, _ = _load_force_arrays(force_path)
            force_cache[force_path] = (force_raw, _lowpass_force(force_raw, 250.0, 12.0))
        force_raw, measured_filtered = force_cache[force_path]
        raw_index = np.rint(times * 250.0).astype(int)
        in_range = (raw_index >= 0) & (raw_index < len(measured_filtered))
        safe_index = np.clip(raw_index, 0, len(measured_filtered) - 1)
        measured_force = measured_filtered[safe_index].copy()
        measured_raw = measured_raw[safe_index].copy()
        measured_force[~in_range] = np.nan
        measured_raw[~in_range] = np.nan
        phase = phase_labels_from_forces(times, measured_force, measured_raw, bw_n)
        measured_fz = measured_force[:, :, 2]
        true_contact = measured_fz > 0.05 * bw_n
        wrench_key = (model_path, pipeline_path, segment)
        if wrench_key not in wrench_cache:
            wrench_cache[wrench_key] = read_contact_wrenches(model_path, pipeline_path, segment, times)
        wrench = wrench_cache[wrench_key]
        net_force_z = wrench.target_force_z.to_numpy(dtype=float)
        force_valid = (
            in_range & np.isfinite(measured_fz).all(axis=1) & np.isfinite(net_force_z)
            & (np.abs(measured_fz.sum(axis=1) - net_force_z) <= 0.25 * bw_n)
        )
        valid &= force_valid
        target_moments = np.column_stack([
            target[f"{coordinate}_tau_target"].to_numpy(dtype=float) / mass
            for coordinate in COORDINATES
        ])
        baseline_path = Path(row["estimated_sto"])
        if baseline_path not in baseline_cache:
            baseline_cache[baseline_path] = read_storage(baseline_path)
        baseline = baseline_cache[baseline_path]
        baseline_moments = interpolate_columns(
            baseline, times, [f"{coordinate}_moment" for coordinate in COORDINATES], baseline_path
        ) / mass
        candidate_path = Path(row["output_dir"]) / row["output_name"]
        candidate = read_storage(candidate_path)
        candidate_moments = interpolate_columns(
            candidate, times, [f"{coordinate}_moment" for coordinate in COORDINATES], candidate_path
        ) / mass
        for coordinate_index, coordinate in enumerate(COORDINATES):
            for phase_name in PHASES:
                phase_mask = np.ones(len(target), dtype=bool) if phase_name == "total" else phase == phase_name
                side_index = 0 if coordinate.endswith("_l") else 1
                # Total reproduces Table I's ipsilateral measured-stance mask;
                # phase rows score valid, force-consistent target frames in that phase.
                mask = (valid & phase_mask & true_contact[:, side_index] if phase_name == "total"
                        else valid & phase_mask)
                if mask.any():
                    samples[(variant, subject, trial, coordinate, phase_name)].append((
                        target_moments[mask, coordinate_index],
                        baseline_moments[mask, coordinate_index],
                        candidate_moments[mask, coordinate_index],
                    ))

    rows: list[dict[str, object]] = []
    for variant in variants:
        for coordinate in COORDINATES:
            for phase_name in PHASES:
                subject_rows = []
                subjects = sorted({key[1] for key in samples
                                   if key[0] == variant and key[3] == coordinate and key[4] == phase_name})
                for subject in subjects:
                    trial_metrics = []
                    trials = sorted({key[2] for key in samples
                                     if key[0] == variant and key[1] == subject
                                     and key[3] == coordinate and key[4] == phase_name})
                    for trial in trials:
                        chunks = samples[(variant, subject, trial, coordinate, phase_name)]
                        target = np.concatenate([part[0] for part in chunks])
                        base = np.concatenate([part[1] for part in chunks])
                        candidate = np.concatenate([part[2] for part in chunks])
                        trial_metrics.append((
                            float(np.sqrt(np.mean((target - base) ** 2))),
                            float(np.sqrt(np.mean((target - candidate) ** 2))),
                            pearson(target, base), pearson(target, candidate), len(target),
                        ))
                    base_rmse = float(np.mean([item[0] for item in trial_metrics]))
                    candidate_rmse = float(np.mean([item[1] for item in trial_metrics]))
                    subject_rows.append((subject, base_rmse, candidate_rmse, candidate_rmse - base_rmse,
                                         float(np.nanmean([item[2] for item in trial_metrics])),
                                         float(np.nanmean([item[3] for item in trial_metrics])),
                                         int(sum(item[4] for item in trial_metrics))))
                for subject, base_rmse, candidate_rmse, delta, base_r, candidate_r, n in subject_rows:
                    rows.append({
                        "record_type": "subject", "variant": variant, "subject": subject,
                        "coordinate": coordinate, "phase": phase_name,
                        "baseline_rmse_nmkg": base_rmse, "variant_rmse_nmkg": candidate_rmse,
                        "delta_rmse_nmkg": delta, "baseline_r": base_r, "variant_r": candidate_r,
                        "n": n, "mean_delta_rmse_nmkg": math.nan,
                        "ci95_low_nmkg": math.nan, "ci95_high_nmkg": math.nan,
                        "signflip_p": math.nan,
                    })
                deltas = np.asarray([row[3] for row in subject_rows], dtype=float)
                if len(deltas):
                    mean_delta = float(np.mean(deltas))
                    half = float(t.ppf(0.975, len(deltas) - 1) * np.std(deltas, ddof=1) / math.sqrt(len(deltas))) if len(deltas) > 1 else math.nan
                    rows.append({
                        "record_type": "paired_summary", "variant": variant, "subject": "ALL",
                        "coordinate": coordinate, "phase": phase_name,
                        "baseline_rmse_nmkg": float(np.mean([row[1] for row in subject_rows])),
                        "variant_rmse_nmkg": float(np.mean([row[2] for row in subject_rows])),
                        "delta_rmse_nmkg": mean_delta,
                        "baseline_r": float(np.nanmean([row[4] for row in subject_rows])),
                        "variant_r": float(np.nanmean([row[5] for row in subject_rows])),
                        "n": int(sum(row[6] for row in subject_rows)),
                        "mean_delta_rmse_nmkg": mean_delta,
                        "ci95_low_nmkg": mean_delta - half if math.isfinite(half) else math.nan,
                        "ci95_high_nmkg": mean_delta + half if math.isfinite(half) else math.nan,
                        "signflip_p": signflip_p(deltas),
                    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"variants={len(variants)} subjects=6 rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

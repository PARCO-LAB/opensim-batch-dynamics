#!/usr/bin/env python3
"""Score selected Phase D prototype IDs against measured ID targets."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from evaluate_hybrid_ids import (
    COORDINATES,
    PHASES,
    TIME_TOLERANCE_S,
    interpolate_columns,
    mapped_source_path,
    metric,
    phase_labels,
    read_storage,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest:
        raise ValueError(f"Empty prototype ID manifest: {args.manifest}")

    samples: dict[tuple[str, str, str, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    variants = sorted({row["variant"] for row in manifest})
    for row in manifest:
        variant, subject, trial, segment = row["variant"], row["subject"], row["trial"], int(row["segment"])
        target_csv = pd.read_csv(row["target_csv"])
        target = target_csv[target_csv.segment.astype(int) == segment].reset_index(drop=True)
        times = target.time.to_numpy(dtype=float)
        valid = target.valid.to_numpy(dtype=int) == 1
        measured_path = Path(row["measured_mot"])
        measured_all = read_storage(measured_path)
        measured_idx = align_indices(measured_all.time.to_numpy(dtype=float), times, measured_path)
        measured = measured_all.iloc[measured_idx].reset_index(drop=True)
        metadata_path = Path(row["target_csv"]).parent / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        model_path = mapped_source_path(metadata["source_paths"]["model"], args.source_torque_root, args.torque_root)
        mass = read_model_mass(model_path)
        phase = phase_labels(times, measured, mass)
        target_moments = np.column_stack([
            target[f"{coordinate}_tau_target"].to_numpy(dtype=float) / mass
            for coordinate in COORDINATES
        ])
        baseline_path = Path(row["estimated_sto"])
        baseline = read_storage(baseline_path)
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
                mask = valid & phase_mask
                if mask.any():
                    samples[(variant, subject, coordinate, phase_name)].append((
                        target_moments[mask, coordinate_index],
                        baseline_moments[mask, coordinate_index],
                        candidate_moments[mask, coordinate_index],
                    ))

    rows: list[dict[str, object]] = []
    for variant in variants:
        for coordinate in COORDINATES:
            for phase_name in PHASES:
                subject_rows = []
                for subject in sorted({key[1] for key in samples if key[0] == variant and key[2] == coordinate and key[3] == phase_name}):
                    chunks = samples[(variant, subject, coordinate, phase_name)]
                    target = np.concatenate([part[0] for part in chunks])
                    base = np.concatenate([part[1] for part in chunks])
                    candidate = np.concatenate([part[2] for part in chunks])
                    base_rmse = float(np.sqrt(np.mean((target - base) ** 2)))
                    candidate_rmse = float(np.sqrt(np.mean((target - candidate) ** 2)))
                    delta = candidate_rmse - base_rmse
                    subject_rows.append((subject, base_rmse, candidate_rmse, delta,
                                         pearson(target, base), pearson(target, candidate), len(target)))
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
                        "delta_rmse_nmkg": mean_delta, "baseline_r": float(np.nanmean([row[4] for row in subject_rows])),
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

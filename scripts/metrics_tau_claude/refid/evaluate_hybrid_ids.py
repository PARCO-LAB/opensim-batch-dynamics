#!/usr/bin/env python3
"""Score hybrid-load ID runs by coordinate and measured contact phase."""

from __future__ import annotations

import argparse
import csv
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COORDINATES = (
    "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
)
VARIANTS = ("C", "V", "H", "P", "CV", "CVH", "CVHP", "PH", "PHV", "PHVC", "ALL")
PHASES = ("total", "single_stance", "double_stance", "flight", "transition", "other")
TIME_TOLERANCE_S = 2e-4
TRANSITION_WINDOW_S = 0.050


def read_storage(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    try:
        endheader = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as exc:
        raise ValueError(f"Missing endheader in {path}") from exc
    return pd.read_csv(path, sep=r"\s+", skiprows=endheader + 1, engine="python")


def read_model_mass(path: Path) -> float:
    root = ET.parse(path).getroot()
    masses = [float(node.text) for node in root.findall(".//Body/mass") if node.text and node.text.strip()]
    mass = sum(masses)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError(f"Invalid model mass in {path}")
    return mass


def mapped_source_path(path: str, source_root: str, torque_root: Path) -> Path:
    source, root = os.path.normpath(path), os.path.normpath(source_root)
    if source == root:
        return torque_root
    prefix = root + os.sep
    if not source.startswith(prefix):
        raise ValueError(f"Metadata path is outside source root: {source}")
    return torque_root / os.path.relpath(source, root)


def read_mot(path: Path) -> pd.DataFrame:
    return read_storage(path)


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray, path: Path) -> np.ndarray:
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(source_times[left] - target_times) < np.abs(source_times[right] - target_times)
    idx = np.where(use_left, left, right)
    error = np.abs(source_times[idx] - target_times)
    if np.any(error > TIME_TOLERANCE_S):
        raise ValueError(f"Cannot align {path}; max time difference={error.max():.8g}s")
    return idx


def interpolate_columns(frame: pd.DataFrame, target_times: np.ndarray, columns: Iterable[str], path: Path) -> np.ndarray:
    source_times = frame.time.to_numpy(dtype=float)
    if source_times.size < 2 or np.any(np.diff(source_times) <= 0):
        raise ValueError(f"Storage times must be strictly increasing in {path}")
    return np.column_stack([
        np.interp(target_times, source_times, frame[column].to_numpy(dtype=float))
        for column in columns
    ])


def phase_labels(times: np.ndarray, measured: pd.DataFrame, mass_kg: float) -> np.ndarray:
    bw = mass_kg * 9.81
    contact = np.column_stack([
        measured[f"ground_force_calcn_{side}_vz"].to_numpy(dtype=float) > 0.05 * bw
        for side in "lr"
    ])
    measured_fz = np.column_stack([
        measured[f"ground_force_calcn_{side}_vz"].to_numpy(dtype=float)
        for side in "lr"
    ])
    flight = np.all(measured_fz <= 1.0, axis=1)
    labels = np.full(times.shape, "other", dtype=object)
    labels[contact.sum(axis=1) == 1] = "single_stance"
    labels[contact.sum(axis=1) == 2] = "double_stance"
    labels[flight] = "flight"

    transition = np.zeros(times.shape, dtype=bool)
    for side_idx in range(2):
        states = contact[:, side_idx]
        edges = np.flatnonzero(states[1:] != states[:-1]) + 1
        if states.size and states[0]:
            edges = np.concatenate(([0], edges))
        for edge in edges:
            transition |= np.abs(times - times[edge]) <= TRANSITION_WINDOW_S
    labels[transition] = "transition"
    return labels


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def metric(target: np.ndarray, estimate: np.ndarray) -> tuple[float, float, int, dict[str, float]]:
    finite = np.isfinite(target) & np.isfinite(estimate)
    target, estimate = target[finite], estimate[finite]
    if not len(target):
        return math.nan, math.nan, 0, {name: 0.0 for name in ("sum_target", "sum_estimate", "sum_target_sq", "sum_estimate_sq", "sum_cross")}
    sums = {
        "sum_target": float(np.sum(target)),
        "sum_estimate": float(np.sum(estimate)),
        "sum_target_sq": float(np.dot(target, target)),
        "sum_estimate_sq": float(np.dot(estimate, estimate)),
        "sum_cross": float(np.dot(target, estimate)),
    }
    return float(np.sqrt(np.mean((target - estimate) ** 2))), pearson(target, estimate), int(len(target)), sums


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def score_segment(rows: list[dict[str, str]], source_torque_root: str, torque_root: Path):
    row0 = rows[0]
    target_csv = pd.read_csv(row0["target_csv"])
    segment = int(row0["segment"])
    target = target_csv[target_csv.segment.astype(int) == segment].copy()
    target_times = target.time.to_numpy(dtype=float)
    valid = target.valid.to_numpy(dtype=int) == 1
    measured_path = Path(row0["measured_mot"])
    measured_all = read_mot(measured_path)
    measured_idx = nearest_indices(measured_all.time.to_numpy(dtype=float), target_times, measured_path)
    measured = measured_all.iloc[measured_idx].reset_index(drop=True)
    metadata_path = Path(row0["target_csv"]).parent / "run_metadata.json"
    metadata = __import__("json").loads(metadata_path.read_text())
    pipeline_csv = mapped_source_path(metadata["source_paths"]["pipeline_csv"], source_torque_root, torque_root)
    pipeline = pd.read_csv(pipeline_csv, usecols=["subject_mass_kg"])
    phase = phase_labels(target_times, measured, float(pipeline.subject_mass_kg.iloc[0]))
    model_path = Path(row0["target_sto"]).parent / f"{row0['subject']}_{row0['trial']}_segment_{segment}_id_model_nomarkers.osim"
    # The model filename has some trials whose segment suffix is different from trial; resolve from setup XML instead.
    if not model_path.exists():
        model_candidates = list(Path(row0["target_csv"]).parent.glob(f"segments/segment_{segment}/*_id_model_nomarkers.osim"))
        if len(model_candidates) != 1:
            raise FileNotFoundError(f"Cannot find the segment model for {row0['subject']}/{row0['trial']}/{segment}")
        model_path = model_candidates[0]
    mass = read_model_mass(model_path)
    tau_target = np.column_stack([
        target[f"{coordinate}_tau_target"].to_numpy(dtype=float) / mass
        for coordinate in COORDINATES
    ])
    estimates: dict[str, np.ndarray] = {}
    variant_rows = {row["variant"]: row for row in rows}
    for variant, group in variant_rows.items():
        output_path = Path(group["output_dir"]) / group["output_name"]
        if not output_path.is_file():
            raise FileNotFoundError(f"Missing ID output: {output_path}")
        sto = read_storage(output_path)
        estimates[variant] = interpolate_columns(
            sto, target_times, [f"{coordinate}_moment" for coordinate in COORDINATES], output_path
        ) / mass

    baseline_sto = read_storage(Path(row0["estimated_sto"]))
    estimates["BASELINE"] = interpolate_columns(
        baseline_sto, target_times, [f"{coordinate}_moment" for coordinate in COORDINATES], Path(row0["estimated_sto"])
    ) / mass
    variants = ["BASELINE", *VARIANTS]
    metrics: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    decomposition_rows: list[dict[str, object]] = []
    for variant in variants:
        estimate = estimates[variant]
        for coordinate_index, coordinate in enumerate(COORDINATES):
            for phase_name in PHASES:
                phase_mask = np.ones(len(target), dtype=bool) if phase_name == "total" else phase == phase_name
                mask = valid & phase_mask
                rmse, corr, n, stats = metric(tau_target[mask, coordinate_index], estimate[mask, coordinate_index])
                metrics[(variant, coordinate, phase_name)] = (rmse, corr, n)
                base_rmse, base_corr, _ = metrics.get(("BASELINE", coordinate, phase_name), (math.nan, math.nan, 0))
                decomposition_rows.append({
                    "variant": variant,
                    "subject": row0["subject"],
                    "trial": row0["trial"],
                    "segment": segment,
                    "coordinate": coordinate,
                    "phase": phase_name,
                    "rmse_nmkg": rmse,
                    "r": corr,
                    "n": n,
                    "rmse_reduction_vs_baseline_nmkg": base_rmse - rmse if math.isfinite(base_rmse) else math.nan,
                    "r_change_vs_baseline": corr - base_corr if math.isfinite(corr) and math.isfinite(base_corr) else math.nan,
                    **stats,
                })

    # Verify the measured-load ID reproduces both the saved ID target and the target CSV.
    all_sto_path = next(Path(r["output_dir"]) / r["output_name"] for r in rows if r["variant"] == "ALL")
    all_sto = read_storage(all_sto_path)
    stored_target = read_storage(Path(row0["target_sto"]))
    stored_times = stored_target.time.to_numpy(dtype=float)
    all_to_stored = interpolate_columns(all_sto, stored_times, [f"{coordinate}_moment" for coordinate in COORDINATES], all_sto_path)
    stored_raw = stored_target[[f"{coordinate}_moment" for coordinate in COORDINATES]].to_numpy(dtype=float)
    all_to_csv = interpolate_columns(all_sto, target_times, [f"{coordinate}_moment" for coordinate in COORDINATES], all_sto_path)
    csv_raw = np.column_stack([target[f"{coordinate}_tau_target"].to_numpy(dtype=float) for coordinate in COORDINATES])
    check = {
        "subject": row0["subject"], "trial": row0["trial"], "segment": segment,
        "n_frames": int(valid.sum()),
        "max_abs_diff_saved_id_nm": float(np.max(np.abs(all_to_stored - stored_raw))) if len(stored_raw) else math.nan,
        "max_abs_diff_tau_target_nm": float(np.max(np.abs(all_to_csv[valid] - csv_raw[valid]))) if valid.any() else math.nan,
    }
    check["check_status"] = "checked" if valid.any() else "not_applicable_no_valid_frames"
    check["pass_1e6"] = check["max_abs_diff_saved_id_nm"] <= 1e-6 and (
        not valid.any() or check["max_abs_diff_tau_target_nm"] <= 1e-6
    )
    return decomposition_rows, check, metrics


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--torque-root", type=Path, required=True)
    parser.add_argument("--source-torque-root", default="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in manifest:
        grouped.setdefault((row["subject"], row["trial"], int(row["segment"])), []).append(row)
    decomposition: list[dict[str, object]] = []
    torque_check: list[dict[str, object]] = []
    for index, (key, rows) in enumerate(sorted(grouped.items())):
        part_rows, check, _ = score_segment(rows, args.source_torque_root, args.torque_root)
        decomposition.extend(part_rows)
        torque_check.append(check)
        if (index + 1) % 25 == 0:
            print(f"evaluated_segments={index + 1}/{len(grouped)}", flush=True)
    write_rows(args.output_dir / "decomposition_by_trial.csv", decomposition)
    write_rows(args.output_dir / "torque_check.csv", torque_check)

    # Pool frames within subject, then report a subject-level score for each variant.
    pooled: list[dict[str, object]] = []
    detail = pd.DataFrame(decomposition)
    for (variant, subject, coordinate, phase), part in detail.groupby(["variant", "subject", "coordinate", "phase"], sort=True):
        finite = part[part.n > 0]
        n = int(finite.n.sum())
        rmse = math.sqrt(float(np.sum(np.square(finite.rmse_nmkg) * finite.n) / n)) if n else math.nan
        if n:
            sum_x, sum_y = float(finite.sum_target.sum()), float(finite.sum_estimate.sum())
            sum_x2, sum_y2 = float(finite.sum_target_sq.sum()), float(finite.sum_estimate_sq.sum())
            sum_xy = float(finite.sum_cross.sum())
            cov = sum_xy - sum_x * sum_y / n
            var_x = max(0.0, sum_x2 - sum_x * sum_x / n)
            var_y = max(0.0, sum_y2 - sum_y * sum_y / n)
            corr = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else math.nan
        else:
            corr = math.nan
        base = detail[(detail.variant == "BASELINE") & (detail.subject == subject) & (detail.coordinate == coordinate) & (detail.phase == phase)]
        base_n = int(base.n.sum())
        base_rmse = math.sqrt(float(np.sum(np.square(base.rmse_nmkg) * base.n) / base_n)) if base_n else math.nan
        pooled.append({
            "variant": variant, "subject": subject, "coordinate": coordinate, "phase": phase,
            "rmse_nmkg": rmse, "r": corr, "n": n,
            "rmse_reduction_vs_baseline_nmkg": base_rmse - rmse if math.isfinite(base_rmse) else math.nan,
        })
    write_rows(args.output_dir / "decomposition.csv", pooled)
    write_rows(
        args.output_dir / "ablation.csv",
        [row.copy() for row in pooled if row["variant"] != "BASELINE"],
    )
    pooled_by_key = {
        (row["variant"], row["subject"], row["coordinate"], row["phase"]): float(row["rmse_nmkg"])
        for row in pooled
    }
    contributions: list[dict[str, object]] = []
    for subject in sorted({row["subject"] for row in pooled}):
        for coordinate in COORDINATES:
            for phase in PHASES:
                def rmse(variant: str) -> float:
                    return pooled_by_key.get((variant, subject, coordinate, phase), math.nan)
                order_a = {
                    "C": rmse("BASELINE") - rmse("C"),
                    "V": rmse("C") - rmse("CV"),
                    "H": rmse("CV") - rmse("CVH"),
                    "P": rmse("CVH") - rmse("CVHP"),
                }
                order_b = {
                    "P": rmse("BASELINE") - rmse("P"),
                    "H": rmse("P") - rmse("PH"),
                    "V": rmse("PH") - rmse("PHV"),
                    "C": rmse("PHV") - rmse("PHVC"),
                }
                for component in ("C", "V", "H", "P"):
                    a, b = order_a[component], order_b[component]
                    contributions.append({
                        "subject": subject,
                        "coordinate": coordinate,
                        "phase": phase,
                        "component": component,
                        "order_cvhp_rmse_reduction_nmkg": a,
                        "order_phvc_rmse_reduction_nmkg": b,
                        "two_order_mean_rmse_reduction_nmkg": (a + b) / 2.0 if math.isfinite(a) and math.isfinite(b) else math.nan,
                    })
    write_rows(args.output_dir / "decomposition_contributions.csv", contributions)
    passed = len(torque_check) == len(grouped) and all(bool(row["pass_1e6"]) for row in torque_check)
    print(f"segments={len(grouped)} decomposition_rows={len(pooled)} torque_check_pass={passed}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

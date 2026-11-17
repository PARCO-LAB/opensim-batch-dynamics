"""Evaluate GroundLink GRF and torque targets using the Table I protocol.

The estimator is evaluated from kinematics-only pipeline outputs. Measured
forces are read here only to score those outputs and to build reference
methods. A force-residual mask (default 0.25 BW) removes frames where the
force-plate total is inconsistent with the pipeline's net-force target.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import t as student_t


GRAVITY_MPS2 = 9.81
FORCE_FS_HZ = 250.0
FORCE_CUTOFF_HZ = 12.0
CONTACT_THRESHOLD_BW = 0.05
FALSE_CONTACT_THRESHOLD_N = 20.0
TARGET_COORDINATES = (
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
)

DEFAULT_SOURCE_TORQUE_ROOT = "/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque"
DEFAULT_SOURCE_FORCE_ROOT = "/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink/force"


@dataclass(frozen=True)
class EvaluationConfig:
    torque_root: Path
    force_root: Path
    output_dir: Path
    source_torque_root: str = DEFAULT_SOURCE_TORQUE_ROOT
    source_force_root: str = DEFAULT_SOURCE_FORCE_ROOT
    subjects: tuple[str, ...] = ("s001", "s002", "s003", "s004", "s005", "s006")
    valid_tau_bw: float = 0.25
    force_fs_hz: float = FORCE_FS_HZ
    force_cutoff_hz: float = FORCE_CUTOFF_HZ
    contact_threshold_bw: float = CONTACT_THRESHOLD_BW
    false_contact_threshold_n: float = FALSE_CONTACT_THRESHOLD_N


def _local_path(source_path: str, source_root: str, local_root: Path) -> Path:
    """Map an HPC absolute path to the corresponding local/NAS tree."""
    source = os.path.normpath(source_path)
    root = os.path.normpath(source_root)
    if source == root:
        return local_root
    prefix = root + os.sep
    if source.startswith(prefix):
        return local_root / os.path.relpath(source, root)
    return Path(source)


def _force_path(source_path: str, config: EvaluationConfig) -> Path:
    source = os.path.normpath(source_path)
    root = os.path.normpath(config.source_force_root)
    if source == root:
        return config.force_root
    prefix = root + os.sep
    if source.startswith(prefix):
        return config.force_root / os.path.relpath(source, root)
    return Path(source)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def _load_force_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True).item()
    grf = _as_numpy(payload["GRF"])
    cop = _as_numpy(payload["CoP"])
    if grf.ndim != 3 or grf.shape[1:] != (2, 3):
        raise ValueError(f"Unexpected GRF shape in {path}: {grf.shape}")
    if cop.shape != grf.shape:
        raise ValueError(f"GRF/CoP shape mismatch in {path}: {grf.shape} vs {cop.shape}")
    if not np.isfinite(grf).all():
        grf = np.nan_to_num(grf, copy=False)
    return grf * 1000.0, cop


def _lowpass_force(force_n: np.ndarray, fs_hz: float, cutoff_hz: float) -> np.ndarray:
    if force_n.shape[0] < 16:
        raise ValueError(f"Need at least 16 force samples for filtfilt, got {force_n.shape[0]}")
    b, a = butter(4, cutoff_hz / (fs_hz / 2.0))
    return filtfilt(b, a, force_n, axis=0)


def _read_model_mass(model_path: Path) -> float:
    tree = ET.parse(model_path)
    masses = [
        float(node.text)
        for node in tree.findall(".//Body/mass")
        if node.text and node.text.strip()
    ]
    if not masses or not math.isfinite(sum(masses)) or sum(masses) <= 0:
        raise ValueError(f"Could not derive model mass from {model_path}")
    return float(sum(masses))


def _wrench_files(metadata: dict[str, Any], pipeline_csv: Path, model_path: Path) -> tuple[list[Path], bool]:
    """Find target-force logs, preferring the run named by target metadata.

    Two sidestretch_1 recovery runs contain only setup XMLs. In those cases a
    completed sibling run supplies the net-force series; the fallback is
    returned in the evaluation log and is never used for the torque targets.
    """
    run_root = model_path.parents[2]
    exact_root = run_root / "results" / "ID_estimatedGRF"

    def collect(root: Path) -> list[Path]:
        if not root.is_dir():
            return []
        return sorted(Path(p) for p in glob.glob(str(root / "**" / "*_contact_wrenches_estimated.csv"), recursive=True))

    exact = collect(exact_root)
    if exact:
        return exact, False

    subject = str(metadata["subject"])
    stem = pipeline_csv.stem
    candidates: list[Path] = []
    for sibling in sorted(pipeline_csv.parent.glob(stem + "_*")):
        if not sibling.is_dir() or sibling == run_root:
            continue
        candidates.extend(collect(sibling / "results" / "ID_estimatedGRF"))
    return sorted(candidates), bool(candidates)


def _load_net_force_target(paths: list[Path], times: np.ndarray) -> np.ndarray:
    if not paths:
        raise FileNotFoundError("No contact_wrenches_estimated.csv found for trial")
    parts = [
        pd.read_csv(path, usecols=["time", "target_force_x", "target_force_y", "target_force_z"])
        for path in paths
    ]
    joined = pd.concat(parts, ignore_index=True).sort_values("time")
    joined = joined.drop_duplicates("time", keep="first")
    source_t = joined["time"].to_numpy(dtype=float)
    target = joined[["target_force_x", "target_force_y", "target_force_z"]].to_numpy(dtype=float)
    if source_t.size < 2:
        raise ValueError(f"Insufficient contact-wrench samples across {paths}")
    aligned = np.column_stack([np.interp(times, source_t, target[:, axis]) for axis in range(3)])
    outside = (times < source_t[0] - 1e-7) | (times > source_t[-1] + 1e-7)
    aligned[outside] = np.nan
    return aligned


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    if not a.size:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(a - b))))


def _f1(pred: np.ndarray, truth: np.ndarray) -> float:
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else float("nan")


def _weighted_cop(df: pd.DataFrame, side: str) -> np.ndarray:
    calcn = df[[f"calcn_{side}_cop_x", f"calcn_{side}_cop_y"]].to_numpy(dtype=float)
    toes = df[[f"toes_{side}_cop_x", f"toes_{side}_cop_y"]].to_numpy(dtype=float)
    wc = np.clip(df[f"calcn_{side}_grf_z"].to_numpy(dtype=float), 0.0, None)
    wt = np.clip(df[f"toes_{side}_grf_z"].to_numpy(dtype=float), 0.0, None)
    total = wc + wt
    midpoint = 0.5 * (calcn + toes)
    weighted = np.zeros_like(midpoint)
    nonzero = total > 1e-6
    weighted[nonzero] = (calcn[nonzero] * wc[nonzero, None] + toes[nonzero] * wt[nonzero, None]) / total[nonzero, None]
    weighted[~nonzero] = midpoint[~nonzero]
    return weighted


def _lag_frames(reference: np.ndarray, estimate: np.ndarray, valid: np.ndarray, max_lag: int = 30) -> int:
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            left = slice(-lag, None)
            right = slice(None, lag)
        elif lag > 0:
            left = slice(None, -lag)
            right = slice(lag, None)
        else:
            left = right = slice(None)
        mask = valid[left] & valid[right]
        corr = _pearson(reference[left][mask], estimate[right][mask])
        if math.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_lag


def _metric_row(
    *,
    subject: str,
    trial: str,
    method: str,
    metric: str,
    value: float,
    n: int,
    coordinate: str = "",
    level: str = "trial",
    ci_low: float = float("nan"),
    ci_high: float = float("nan"),
) -> dict[str, Any]:
    return {
        "level": level,
        "subject": subject,
        "trial": trial,
        "method": method,
        "coordinate": coordinate,
        "metric": metric,
        "value": value,
        "n_frames": int(n),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _trial_rows(
    metadata_path: Path,
    config: EvaluationConfig,
    mass_cache: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    metadata = json.loads(metadata_path.read_text())
    subject, trial = str(metadata["subject"]), str(metadata["trial"])
    paths = metadata["source_paths"]
    pipeline_csv = _local_path(paths["pipeline_csv"], config.source_torque_root, config.torque_root)
    model_path = _local_path(paths["model"], config.source_torque_root, config.torque_root)
    target_csv = metadata_path.parent / "torque_targets.csv"
    force_path = _force_path(paths["measured_force"], config)

    force_raw, cop_raw = _load_force_arrays(force_path)
    force_filtered = _lowpass_force(force_raw, config.force_fs_hz, config.force_cutoff_hz)

    fields = ["time", "subject_mass_kg"]
    for side in "lr":
        for body in (f"calcn_{side}", f"toes_{side}"):
            fields += [
                f"{body}_grf_{axis}" for axis in "xyz"
            ] + [f"{body}_cop_{axis}" for axis in "xy"] + [f"{body}_contact"]
    pipeline = pd.read_csv(pipeline_csv, usecols=fields)
    torque_fields = ["time", "valid", "cop_invalid", "side_invalid"]
    for coordinate in TARGET_COORDINATES:
        torque_fields.extend([f"{coordinate}_tau_target", f"{coordinate}_tau_hat"])
    torque = pd.read_csv(target_csv, usecols=torque_fields)
    times = pipeline["time"].to_numpy(dtype=float)
    if len(torque) != len(pipeline) or not np.allclose(torque["time"].to_numpy(dtype=float), times, atol=2e-5, rtol=0.0):
        raise ValueError(f"Target/pipeline time mismatch for {subject}/{trial}")

    mass_grf = float(pipeline["subject_mass_kg"].iloc[0])
    if not math.isfinite(mass_grf) or mass_grf <= 0:
        raise ValueError(f"Invalid pipeline subject_mass_kg for {subject}/{trial}: {mass_grf}")
    if subject not in mass_cache:
        mass_cache[subject] = _read_model_mass(model_path)
    mass_model = mass_cache[subject]
    bw = mass_grf * GRAVITY_MPS2

    raw_index = np.rint(times * config.force_fs_hz).astype(int)
    force_in_range = (raw_index >= 0) & (raw_index < len(force_filtered))
    safe_index = np.clip(raw_index, 0, len(force_filtered) - 1)
    measured_force = force_filtered[safe_index]
    measured_force[~force_in_range] = np.nan
    measured_cop = cop_raw[safe_index, :, :2].copy()
    measured_cop[~force_in_range] = np.nan
    shift = np.asarray(metadata.get("cop_shift_m", [0.0, 0.0]), dtype=float).reshape(-1)
    if shift.size >= 2:
        # Phase C's T_code is added directly to the copied measured-force MOT
        # points before ID. Older B2 shift metadata instead describes the
        # measured-minus-reference offset and is subtracted by the extractor.
        if metadata.get("cop_shift_source") == "tcode":
            measured_cop += shift[:2][None, None, :]
        else:
            measured_cop -= shift[:2][None, None, :]

    estimated_force: dict[str, np.ndarray] = {}
    estimated_contact: dict[str, np.ndarray] = {}
    estimated_cop: dict[str, np.ndarray] = {}
    true_contact: dict[str, np.ndarray] = {}
    measured_by_foot: dict[str, np.ndarray] = {}
    for side_idx, side in enumerate("lr"):
        calcn = pipeline[[f"calcn_{side}_grf_{a}" for a in "xyz"]].to_numpy(dtype=float)
        toes = pipeline[[f"toes_{side}_grf_{a}" for a in "xyz"]].to_numpy(dtype=float)
        estimated_force[side] = calcn + toes
        estimated_contact[side] = (
            (pipeline[f"calcn_{side}_contact"].to_numpy(dtype=float) > 0.5)
            | (pipeline[f"toes_{side}_contact"].to_numpy(dtype=float) > 0.5)
        )
        estimated_cop[side] = _weighted_cop(pipeline, side)
        measured_by_foot[side] = measured_force[:, side_idx, :]
        true_contact[side] = measured_by_foot[side][:, 2] > config.contact_threshold_bw * bw

    cw_paths, cw_fallback = _wrench_files(metadata, pipeline_csv, model_path)
    net_target = _load_net_force_target(cw_paths, times)
    measured_total_z = measured_force[:, :, 2].sum(axis=1)
    force_valid = (
        force_in_range
        & np.isfinite(measured_total_z)
        & np.isfinite(net_target[:, 2])
        & (np.abs(measured_total_z - net_target[:, 2]) <= config.valid_tau_bw * bw)
    )
    torque_valid = force_valid & torque["valid"].to_numpy(dtype=bool)
    cop_valid = force_valid & ~torque["cop_invalid"].to_numpy(dtype=bool)

    methods: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {
        "pipeline": (estimated_force, estimated_contact),
    }
    bw_split = {side: np.tile([0.0, 0.0, 0.5 * bw], (len(times), 1)) for side in "lr"}
    methods["BW_equal_split"] = (bw_split, {side: np.ones(len(times), dtype=bool) for side in "lr"})
    contact_count = true_contact["l"].astype(int) + true_contact["r"].astype(int)
    com_split: dict[str, np.ndarray] = {}
    for side in "lr":
        share = np.where(contact_count > 0, true_contact[side] / np.maximum(contact_count, 1), 0.0)
        com_split[side] = net_target * share[:, None]
    methods["CoM_equal_split"] = (com_split, {side: true_contact[side].copy() for side in "lr"})

    rows: list[dict[str, Any]] = []
    common_n = int(force_valid.sum())
    for method, (forces, contacts) in methods.items():
        fz_rmse = []
        fz_corr = []
        f1_scores = []
        horizontal_rmse = []
        false_contacts_flight = []
        false_contacts_all_valid = []
        cop_errors = []
        side_metrics: dict[str, dict[str, tuple[float, int]]] = {}
        for side in "lr":
            measured = measured_by_foot[side]
            truth = true_contact[side]
            selected_contact = force_valid & truth
            side_metrics[side] = {}
            if selected_contact.any():
                side_fz_rmse = _rmse(forces[side][selected_contact, 2], measured[selected_contact, 2]) / bw
                side_fz_corr = _pearson(forces[side][selected_contact, 2], measured[selected_contact, 2])
                side_horizontal_rmse = float(
                    np.sqrt(np.mean(np.sum(np.square(forces[side][selected_contact, :2] - measured[selected_contact, :2]), axis=1))) / bw
                )
                fz_rmse.append(side_fz_rmse)
                fz_corr.append(side_fz_corr)
                horizontal_rmse.append(side_horizontal_rmse)
                side_metrics[side]["fz_rmse_contact_bw"] = (side_fz_rmse, int(selected_contact.sum()))
                side_metrics[side]["fz_corr_contact"] = (side_fz_corr, int(selected_contact.sum()))
                side_metrics[side]["horizontal_rmse_bw"] = (side_horizontal_rmse, int(selected_contact.sum()))
            side_f1 = _f1(contacts[side][force_valid], truth[force_valid])
            f1_scores.append(side_f1)
            side_metrics[side]["contact_f1"] = (side_f1, common_n)
            flight = force_valid & ~truth
            false_positive_frames = int(np.sum(forces[side][flight, 2] > config.false_contact_threshold_n))
            if flight.any():
                side_flight_rate = false_positive_frames / int(flight.sum())
                false_contacts_flight.append(side_flight_rate)
                side_metrics[side]["false_contact_fraction"] = (side_flight_rate, int(flight.sum()))
            if common_n:
                side_all_valid_rate = false_positive_frames / common_n
                false_contacts_all_valid.append(side_all_valid_rate)
                side_metrics[side]["false_contact_fraction_all_valid"] = (side_all_valid_rate, common_n)
            if method == "pipeline":
                cop_mask = cop_valid & truth
                if cop_mask.any():
                    side_cop_error = float(np.mean(np.linalg.norm(estimated_cop[side][cop_mask] - measured_cop[cop_mask, 0 if side == "l" else 1, :], axis=1)))
                    cop_errors.append(side_cop_error)
                    side_metrics[side]["cop_error_cm"] = (100.0 * side_cop_error, int(cop_mask.sum()))
        total_est = forces["l"] + forces["r"]
        rho = float(np.mean(np.linalg.norm(net_target[force_valid] - total_est[force_valid], axis=1) / bw)) if force_valid.any() else float("nan")
        body_flight = force_valid & ~true_contact["l"] & ~true_contact["r"]
        body_false_contact = float("nan")
        if body_flight.any():
            estimated_any_contact = (
                (forces["l"][:, 2] > config.false_contact_threshold_n)
                | (forces["r"][:, 2] > config.false_contact_threshold_n)
            )
            body_false_contact = float(np.mean(estimated_any_contact[body_flight]))
        values = {
            "fz_rmse_contact_bw": float(np.nanmean(fz_rmse)) if fz_rmse else float("nan"),
            "fz_corr_contact": float(np.nanmean(fz_corr)) if fz_corr else float("nan"),
            "contact_f1": float(np.nanmean(f1_scores)) if f1_scores else float("nan"),
            "rho_bw": rho,
            "horizontal_rmse_bw": float(np.nanmean(horizontal_rmse)) if horizontal_rmse else float("nan"),
            "false_contact_fraction": float(np.nanmean(false_contacts_flight)) if false_contacts_flight else float("nan"),
            "false_contact_fraction_all_valid": float(np.nanmean(false_contacts_all_valid)) if false_contacts_all_valid else float("nan"),
            "false_contact_fraction_body_flight": body_false_contact,
            "cop_error_cm": 100.0 * float(np.nanmean(cop_errors)) if cop_errors else float("nan"),
        }
        for metric, value in values.items():
            if metric.startswith("cop_") or metric.startswith("false_"):
                if method != "pipeline":
                    continue
            rows.append(_metric_row(subject=subject, trial=trial, method=method, metric=metric, value=value, n=common_n))
        for side, side_values in side_metrics.items():
            for metric, (value, n) in side_values.items():
                if metric.startswith(("cop_", "false_")) and method != "pipeline":
                    continue
                rows.append(_metric_row(
                    subject=subject, trial=trial, method=method, metric=metric,
                    coordinate=f"foot_{side}", value=value, n=n,
                ))

    for side in "lr":
        lag = _lag_frames(measured_by_foot[side][:, 2], estimated_force[side][:, 2], force_valid)
        rows.append(_metric_row(subject=subject, trial=trial, method="pipeline", metric=f"lag_frames_{side}", value=float(lag), n=common_n))

    torque_samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for coordinate in TARGET_COORDINATES:
        # Lower-limb moments are scored only during measured stance on the
        # corresponding side, using the same >0.05 BW contact threshold as the
        # force metrics. Including swing frames changes the published Table I
        # torque baseline substantially.
        side = "l" if coordinate.endswith("_l") else "r"
        selected = torque_valid & true_contact[side]
        y = torque.loc[selected, f"{coordinate}_tau_target"].to_numpy(dtype=float) / mass_model
        yhat = torque.loc[selected, f"{coordinate}_tau_hat"].to_numpy(dtype=float) / mass_model
        torque_samples[coordinate] = (y, yhat)
        rows.append(_metric_row(subject=subject, trial=trial, method="pipeline", metric="tau_rmse_nmkg", coordinate=coordinate, value=_rmse(y, yhat), n=len(y)))
        rows.append(_metric_row(subject=subject, trial=trial, method="pipeline", metric="tau_pearson_r", coordinate=coordinate, value=_pearson(y, yhat), n=len(y)))

    log = {
        "subject": subject,
        "trial": trial,
        "status": "ok",
        "frames": len(times),
        "force_valid_frames": common_n,
        "torque_valid_frames": int(torque_valid.sum()),
        "torque_stance_frames_l": int((torque_valid & true_contact["l"]).sum()),
        "torque_stance_frames_r": int((torque_valid & true_contact["r"]).sum()),
        "mass_grf_kg": mass_grf,
        "mass_model_kg": mass_model,
        "model_path": str(model_path),
        "pipeline_csv": str(pipeline_csv),
        "force_path": str(force_path),
        "contact_wrench_paths": [str(p) for p in cw_paths],
        "contact_wrench_fallback": bool(cw_fallback),
        "cop_shift_m": shift.tolist(),
    }
    return rows, log, torque_samples


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    trial_rows = frame[frame["level"] == "trial"]
    summaries: list[dict[str, Any]] = []
    keys = ["method", "coordinate", "metric"]
    for key, group in trial_rows.groupby(keys, dropna=False):
        method, coordinate, metric = key
        per_subject = group.groupby("subject")["value"].mean().dropna()
        for subject, value in per_subject.items():
            summaries.append(_metric_row(subject=str(subject), trial="", method=str(method), metric=str(metric), coordinate=str(coordinate), value=float(value), n=int(group.loc[group.subject == subject, "n_frames"].sum()), level="subject"))
        values = per_subject.to_numpy(dtype=float)
        if values.size:
            mean = float(np.mean(values))
            if values.size > 1:
                half = float(student_t.ppf(0.975, values.size - 1) * np.std(values, ddof=1) / math.sqrt(values.size))
            else:
                half = float("nan")
            summaries.append(_metric_row(subject="mean_6_subjects", trial="", method=str(method), metric=str(metric), coordinate=str(coordinate), value=mean, n=int(values.size), level="overall", ci_low=mean - half, ci_high=mean + half))
    return summaries


def evaluate_dataset(config: EvaluationConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_root = config.torque_root / "torque_targets_12hz_nomuscles_copfix_tcode"
    metadata_files = sorted(target_root.glob("s[0-9][0-9][0-9]/*/run_metadata.json"))
    metadata_files = [
        path for path in metadata_files
        if not config.subjects or path.parents[1].name in config.subjects
    ]
    if not metadata_files:
        raise FileNotFoundError(f"No target run_metadata.json files under {target_root}")

    trial_rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    mass_cache: dict[str, float] = {}
    torque_samples_by_subject: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for metadata_path in metadata_files:
        try:
            rows, log, torque_samples = _trial_rows(metadata_path, config, mass_cache)
            trial_rows.extend(rows)
            logs.append(log)
            for coordinate, samples in torque_samples.items():
                torque_samples_by_subject[log["subject"]][coordinate].append(samples)
            print(f"{log['subject']} {log['trial']}: ok ({log['force_valid_frames']}/{log['frames']} force-valid)", flush=True)
        except Exception as exc:
            metadata = json.loads(metadata_path.read_text())
            logs.append({"subject": metadata.get("subject", ""), "trial": metadata.get("trial", ""), "status": f"error: {exc}"})
            print(f"{metadata.get('subject')} {metadata.get('trial')}: ERROR {exc}", flush=True)

    summary_rows = _summarize(trial_rows)
    pooled_torque_rows: list[dict[str, Any]] = []
    for coordinate in TARGET_COORDINATES:
        subject_metrics: dict[str, dict[str, float]] = {}
        for subject, coordinate_data in torque_samples_by_subject.items():
            samples = coordinate_data.get(coordinate, [])
            if not samples:
                continue
            target = np.concatenate([sample[0] for sample in samples])
            estimate = np.concatenate([sample[1] for sample in samples])
            rmse = _rmse(target, estimate)
            corr = _pearson(target, estimate)
            subject_metrics[subject] = {"tau_rmse_nmkg": rmse, "tau_pearson_r": corr}
            pooled_torque_rows.append(_metric_row(
                subject=subject, trial="", method="pipeline_subject_pooled",
                metric="tau_rmse_nmkg", coordinate=coordinate, value=rmse,
                n=len(target), level="subject",
            ))
            pooled_torque_rows.append(_metric_row(
                subject=subject, trial="", method="pipeline_subject_pooled",
                metric="tau_pearson_r", coordinate=coordinate, value=corr,
                n=len(target), level="subject",
            ))
        for metric in ("tau_rmse_nmkg", "tau_pearson_r"):
            values = np.asarray([
                record[metric] for record in subject_metrics.values()
                if math.isfinite(record[metric])
            ], dtype=float)
            if not values.size:
                continue
            mean = float(np.mean(values))
            half = (
                float(student_t.ppf(0.975, values.size - 1) * np.std(values, ddof=1) / math.sqrt(values.size))
                if values.size > 1 else float("nan")
            )
            pooled_torque_rows.append(_metric_row(
                subject="mean_6_subjects", trial="", method="pipeline_subject_pooled",
                metric=metric, coordinate=coordinate, value=mean, n=int(values.size),
                level="overall", ci_low=mean - half, ci_high=mean + half,
            ))
    all_rows = trial_rows + summary_rows + pooled_torque_rows
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(config.output_dir / "baseline_metrics.csv", index=False)
    pd.DataFrame(logs).to_csv(config.output_dir / "eval_log.csv", index=False)
    run_meta = {
        "subjects": list(config.subjects),
        "trials_requested": len(metadata_files),
        "trials_ok": sum(log.get("status") == "ok" for log in logs),
        "valid_tau_bw": config.valid_tau_bw,
        "force_fs_hz": config.force_fs_hz,
        "force_cutoff_hz": config.force_cutoff_hz,
        "contact_threshold_bw": config.contact_threshold_bw,
        "false_contact_threshold_n": config.false_contact_threshold_n,
        "false_contact_denominator": "force_consistent_measured_flight_frames_per_foot; all-valid and bilateral-flight rates are also reported",
        "torque_frame_mask": "force-consistency mask AND torque_targets.valid AND ipsilateral measured contact > 0.05 BW",
        "torque_aggregation": "trial mean per subject plus pooled frames per subject, both reported",
        "torque_root": str(config.torque_root),
        "force_root": str(config.force_root),
    }
    (config.output_dir / "evaluation_metadata.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    return pd.DataFrame(all_rows), pd.DataFrame(logs)


def _parse_args(argv: list[str] | None = None) -> EvaluationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-root", type=Path, required=True, help="GroundLink_torque root containing pipeline/ and torque target folders")
    parser.add_argument("--force-root", type=Path, required=True, help="Root containing s001/*.npy measured-force files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-torque-root", default=DEFAULT_SOURCE_TORQUE_ROOT)
    parser.add_argument("--source-force-root", default=DEFAULT_SOURCE_FORCE_ROOT)
    parser.add_argument("--subjects", nargs="*", default=["s001", "s002", "s003", "s004", "s005", "s006"])
    parser.add_argument("--valid-tau-bw", type=float, default=0.25)
    args = parser.parse_args(argv)
    return EvaluationConfig(
        torque_root=args.torque_root,
        force_root=args.force_root,
        output_dir=args.output_dir,
        source_torque_root=args.source_torque_root,
        source_force_root=args.source_force_root,
        subjects=tuple(args.subjects),
        valid_tau_bw=args.valid_tau_bw,
    )


def main(argv: list[str] | None = None) -> int:
    config = _parse_args(argv)
    metrics, logs = evaluate_dataset(config)
    overall = metrics[metrics.level == "overall"]
    for _, row in overall.iterrows():
        if row.metric in {"fz_rmse_contact_bw", "fz_corr_contact", "contact_f1", "rho_bw", "cop_error_cm", "horizontal_rmse_bw", "false_contact_fraction", "false_contact_fraction_all_valid", "false_contact_fraction_body_flight", "tau_rmse_nmkg", "tau_pearson_r"}:
            print(f"{row.method} {row.metric} {row.coordinate}: {row.value:.6f} [{row.ci95_low:.6f}, {row.ci95_high:.6f}]")
    print(f"trials_ok={sum(logs.status == 'ok')} trials_total={len(logs)} output={config.output_dir}")
    return 0 if bool((logs.status == "ok").all()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

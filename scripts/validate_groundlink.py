#!/usr/bin/env python3
"""Audit and evaluate GroundLink force/motion pairs without touching AMASS batch input."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WALKING = {"walk_1": 1078, "walk_2": 967, "walk_3": 1028}
EXCLUDED_MOTIONS = {"dog", "ballethighleg", "idling"}


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _npz_array_shape(path: Path, name: str) -> tuple[int, ...] | None:
    """Read one NPZ member header without inflating its array."""
    try:
        with zipfile.ZipFile(path) as archive, archive.open(f"{name}.npy") as stream:
            version = stream.read(8)
            if version[:6] != b"\x93NUMPY":
                return None
            major, minor = version[6], version[7]
            import numpy.lib.format as fmt
            shape, _, _ = (fmt.read_array_header_1_0 if (major, minor) == (1, 0) else fmt.read_array_header_2_0)(stream)
            return tuple(shape)
    except (KeyError, OSError, ValueError):
        return None


def load_groundlink_force(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CoP in metres and GRF in N from a GroundLink force file."""
    path = Path(path)
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        data: Any = {key: loaded[key] for key in loaded.files}
        loaded.close()
    elif isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object:
        data = loaded.item()
    else:
        raise ValueError(f"Expected object-array force file: {path}")
    def get(*names: str) -> Any:
        for name in names:
            if name in data:
                return data[name]
        raise KeyError(f"Missing any of {names} in {path}")
    cop = _numpy(get("CoP", "cop")).astype(float)
    grf = _numpy(get("GRF", "grf", "force")).astype(float)
    if cop.shape != grf.shape or cop.ndim != 3 or cop.shape[1:] != (2, 3):
        raise ValueError(f"Expected CoP/GRF shape (T,2,3), got {cop.shape}/{grf.shape}")
    return cop, grf * 1000.0


def _subject_trial(path: Path, strip_full: bool = False) -> tuple[str, str]:
    parts = path.stem.replace("_stageii", "").split("_")
    subject = parts[0] if parts and parts[0].startswith("s") else "unknown"
    trial = "_".join(parts[2:]) if len(parts) > 2 else path.stem
    if strip_full and trial.endswith("_full"):
        trial = trial[:-5]
    return subject, trial


@dataclass(frozen=True)
class Pair:
    subject: str
    trial: str
    motion_path: Path
    force_path: Path | None
    frames: int | None
    exclusion: str = ""


def discover_pairs(
    root: str | Path,
    force_root: str | Path | None = None,
    subject: str | None = None,
) -> list[Pair]:
    root = Path(root)
    force_root = Path(force_root) if force_root is not None else root
    force_files = {_subject_trial(p)[1]: p for p in force_root.glob("*.npy")}
    motion_files = {
        _subject_trial(p)[1]: p
        for p in root.glob("*_stageii.npz")
        if not _subject_trial(p)[1].startswith("walk_")
    }
    for p in (root / "walk_with_start_end").glob("*_full_stageii.npz"):
        motion_files[_subject_trial(p, strip_full=True)[1]] = p
    pairs: list[Pair] = []
    for trial, motion in sorted(motion_files.items()):
        detected_subject = _subject_trial(motion)[0]
        motion_key = trial
        excluded = next((x for x in EXCLUDED_MOTIONS if motion_key.startswith(x)), "")
        force = force_files.get(trial)
        frames = None
        shape = _npz_array_shape(motion, "trans")
        if shape:
            frames = int(shape[0])
        pairs.append(Pair(subject or detected_subject, motion_key, motion, force, frames, excluded))
    return pairs


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def write_manifest(pairs: list[Pair], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["subject", "trial", "motion_path", "force_path", "frames", "force_frames", "status", "exclusion"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in pairs:
            force_frames = ""
            status = "excluded" if pair.exclusion else "missing_force" if pair.force_path is None else "ok"
            if pair.force_path is not None:
                try:
                    force_frames = str(load_groundlink_force(pair.force_path)[0].shape[0])
                    if pair.frames is not None and int(force_frames) != pair.frames:
                        status = "frame_mismatch"
                except Exception:
                    status = "invalid_force"
            writer.writerow({"subject": pair.subject, "trial": pair.trial, "motion_path": str(pair.motion_path),
                             "force_path": str(pair.force_path or ""), "frames": pair.frames or "",
                             "force_frames": force_frames, "status": status, "exclusion": pair.exclusion})
    return output


def _read_csv(path: Path, columns: set[str] | None = None) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            raise ValueError(f"CSV has no header: {path}")
        selected = [(index, name) for index, name in enumerate(header) if columns is None or name in columns]
        names = [name for _, name in selected]
        values = {name: [] for name in names}
        for row in reader:
            for index, name in selected:
                values[name].append(float(row[index] or "nan"))
    return names, {name: np.asarray(value, dtype=float) for name, value in values.items()}


def _interp(source: np.ndarray, target_n: int, target_t: np.ndarray | None = None) -> np.ndarray:
    if target_t is None:
        return source.copy()
    source_t = np.arange(source.shape[0], dtype=float) / 250.0
    if target_t.min(initial=0.0) < source_t[0] or target_t.max(initial=0.0) > source_t[-1] * 1.01:
        raise ValueError("Prediction time coverage is below the required 99%.")
    return np.interp(target_t, source_t, source)


def _metric_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    valid = np.isfinite(pred) & np.isfinite(ref)
    return float(np.sqrt(np.mean((pred[valid] - ref[valid]) ** 2))) if valid.any() else math.nan


def _metric_corr(pred: np.ndarray, ref: np.ndarray) -> float:
    valid = np.isfinite(pred) & np.isfinite(ref)
    if valid.sum() < 2 or np.std(pred[valid]) == 0 or np.std(ref[valid]) == 0:
        return math.nan
    return float(np.corrcoef(pred[valid], ref[valid])[0, 1])


def _contact_stances(contact: np.ndarray) -> list[slice]:
    """Return contiguous reference-contact intervals."""
    edges = np.diff(np.r_[False, contact, False].astype(np.int8))
    return [slice(start, end) for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _physics_residual(pred_total: np.ndarray, target_force: np.ndarray, mass_kg: float) -> float:
    """Mean net-force mismatch against the common COM-derived target."""
    bw = mass_kg * 9.81
    valid = np.isfinite(pred_total).all(axis=1) & np.isfinite(target_force).all(axis=1)
    return float(np.mean(np.linalg.norm(target_force[valid] - pred_total[valid], axis=1)) / bw) if valid.any() else math.nan


def _com_acceleration(data: dict[str, np.ndarray], target_t: np.ndarray) -> np.ndarray | None:
    keys = [f"com_acc_{axis}" for axis in "xyz"]
    if not all(key in data for key in keys):
        return None
    source_t = data.get("time", np.arange(len(next(iter(data.values()))), dtype=float) / 250.0)
    return np.column_stack([np.interp(target_t, source_t, data[key]) for key in keys])


def _force_target(data: dict[str, np.ndarray], target_t: np.ndarray) -> np.ndarray | None:
    keys = [f"target_force_{axis}" for axis in "xyz"]
    if not all(key in data for key in keys):
        return None
    source_t = data.get("time", np.arange(len(next(iter(data.values()))), dtype=float) / 250.0)
    return np.column_stack([np.interp(target_t, source_t, data[key]) for key in keys])


def evaluate_signals(
    pred_grf: np.ndarray,
    ref_grf: np.ndarray,
    pred_cop: np.ndarray,
    ref_cop: np.ndarray,
    mass_kg: float,
    dt: float = 1.0 / 250.0,
) -> list[dict[str, float | str]]:
    """Return the requested force/contact/CoP metrics for one side."""
    if pred_grf.shape != ref_grf.shape or pred_grf.ndim != 2 or pred_grf.shape[1] != 3:
        raise ValueError("GRF arrays must both have shape (T, 3).")
    bw = mass_kg * 9.81
    ref_contact = ref_grf[:, 2] > 0.05 * bw
    pred_contact = np.nan_to_num(pred_grf[:, 2], nan=0.0) > 0.05 * bw
    tp = float(np.sum(pred_contact & ref_contact))
    fp = float(np.sum(pred_contact & ~ref_contact))
    fn = float(np.sum(~pred_contact & ref_contact))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    contact_pred = pred_grf[ref_contact, 2]
    contact_ref = ref_grf[ref_contact, 2]
    peak_errors = np.asarray([
        (np.nanmax(pred_grf[stance, 2]) - np.nanmax(ref_grf[stance, 2])) / bw
        for stance in _contact_stances(ref_contact)
    ], dtype=float)
    rows: list[dict[str, float | str]] = []
    for axis, index in zip(("x", "y", "z"), range(3)):
        rows.extend((
            {"metric": f"grf_rmse_{axis}", "value": _metric_rmse(pred_grf[:, index], ref_grf[:, index]) / bw, "unit": "BW"},
            {"metric": f"grf_corr_{axis}", "value": _metric_corr(pred_grf[:, index], ref_grf[:, index]), "unit": "1"},
        ))
    rows.extend((
        {"metric": "grf_rmse_vector", "value": _metric_rmse(np.linalg.norm(pred_grf, axis=1), np.linalg.norm(ref_grf, axis=1)) / bw, "unit": "BW"},
        {"metric": "fz_peak_error", "value": (float(np.nanmax(pred_grf[:, 2])) - float(np.nanmax(ref_grf[:, 2]))) / bw, "unit": "BW"},
        {"metric": "fz_peak_error_abs_median", "value": float(np.median(np.abs(peak_errors))) if peak_errors.size else math.nan, "unit": "BW"},
        {"metric": "fz_peak_error_iqr", "value": float(np.subtract(*np.percentile(peak_errors, [75, 25]))) if peak_errors.size else math.nan, "unit": "BW"},
        {"metric": "fz_impulse_error", "value": float((np.nansum(pred_grf[:, 2]) - np.nansum(ref_grf[:, 2])) * dt / bw), "unit": "s"},
        {"metric": "fz_impulse_error_abs", "value": abs(float((np.nansum(pred_grf[:, 2]) - np.nansum(ref_grf[:, 2])) * dt / bw)), "unit": "s"},
        {"metric": "fz_rmse_contact", "value": _metric_rmse(contact_pred, contact_ref) / bw, "unit": "BW"},
        {"metric": "fz_corr_contact", "value": _metric_corr(contact_pred, contact_ref), "unit": "1"},
        {"metric": "contact_precision", "value": precision, "unit": "1"},
        {"metric": "contact_recall", "value": recall, "unit": "1"},
        {"metric": "contact_f1", "value": f1, "unit": "1"},
    ))
    cop_valid = ref_contact & np.isfinite(pred_cop).all(axis=1) & np.isfinite(ref_cop).all(axis=1)
    cop_error = np.linalg.norm(pred_cop[cop_valid, :2] - ref_cop[cop_valid, :2], axis=1)
    rows.append({"metric": "pfa_planar_error", "value": float(np.sqrt(np.mean(cop_error ** 2))) if cop_error.size else math.nan, "unit": "m"})
    return rows


def _columns_for_side(data: dict[str, np.ndarray], side: str, suffix: str) -> np.ndarray:
    values = []
    for body in (f"calcn_{side}", f"toes_{side}"):
        key = f"{body}_{suffix}"
        if key in data:
            values.append(data[key])
    if not values:
        raise KeyError(f"Missing predicted {suffix} columns for {side} foot.")
    return np.sum(values, axis=0)


def _aggregate_side_cop(data: dict[str, np.ndarray], side: str) -> np.ndarray:
    """Combine calcn/toes CoPs using each body's vertical GRF as weight."""
    weighted: list[tuple[np.ndarray, np.ndarray]] = []
    fallback: np.ndarray | None = None
    for body in (f"calcn_{side}", f"toes_{side}"):
        cop_keys = [f"{body}_cop_{axis}" for axis in "xyz"]
        force_key = f"{body}_grf_z"
        if not all(key in data for key in (*cop_keys, force_key)):
            continue
        cop = np.column_stack([data[key] for key in cop_keys])
        force_z = np.asarray(data[force_key], dtype=float)
        valid = np.isfinite(cop).all(axis=1) & np.isfinite(force_z)
        weights = np.where(valid, np.maximum(force_z, 0.0), 0.0)
        weighted.append((np.nan_to_num(cop, nan=0.0), weights))
        if body == f"calcn_{side}":
            fallback = cop
    if not weighted:
        return np.full((len(next(iter(data.values()))), 3), np.nan)
    numerator = sum(cop * weights[:, None] for cop, weights in weighted)
    denominator = sum(weights for _, weights in weighted)
    result = np.full_like(numerator, np.nan, dtype=float)
    valid = denominator > 1e-9
    result[valid] = numerator[valid] / denominator[valid, None]
    if fallback is not None:
        result[~valid] = fallback[~valid]
    return result


def evaluate_trial(
    pair: Pair,
    pipeline_csv: str | Path,
    mass_kg: float = 69.86,
    method_name: str = "pipeline",
    com_reference_csv: str | Path | None = None,
    force_target_csv: str | Path | None = None,
) -> list[dict[str, str | float]]:
    """Evaluate one final pipeline CSV against its GroundLink force pair."""
    if pair.force_path is None or pair.exclusion:
        return []
    # ponytail: final CSVs have hundreds of unused joint columns; load only force/PFA fields.
    prediction_columns = {"time"} | {
        f"{body}_{kind}_{axis}"
        for body in ("calcn_l", "toes_l", "calcn_r", "toes_r")
        for kind in ("grf", "cop") for axis in "xyz"
    }
    _, data = _read_csv(Path(pipeline_csv), prediction_columns)
    gt_cop, gt_grf = load_groundlink_force(pair.force_path)
    pred_t = data.get("time", np.arange(len(next(iter(data.values()))), dtype=float) / 250.0)
    ref_t = np.arange(gt_grf.shape[0], dtype=float) / 250.0
    if pred_t.min(initial=0.0) < ref_t[0] or pred_t.max(initial=0.0) > ref_t[-1] * 1.01 or pred_t[-1] < ref_t[-1] * 0.99:
        raise ValueError(f"Prediction time coverage is below 99% for {pair.trial}.")
    com_data = data if com_reference_csv is None else _read_csv(
        Path(com_reference_csv), {"time", "com_acc_x", "com_acc_y", "com_acc_z"}
    )[1]
    com_acc = _com_acceleration(com_data, pred_t)
    target_data = _read_csv(
        Path(force_target_csv), {"time", "target_force_x", "target_force_y", "target_force_z"}
    )[1] if force_target_csv is not None else None
    target_force = _force_target(target_data, pred_t) if target_data is not None else None
    if target_force is None and com_acc is not None:
        target_force = mass_kg * (com_acc - np.array([0.0, 0.0, 9.81]))
    records: list[dict[str, str | float]] = []
    total_forces: dict[str, np.ndarray] = {}
    bw_baseline = np.zeros((pred_t.shape[0], 3), dtype=float)
    bw_baseline[:, 2] = mass_kg * 9.81 / 2.0
    for side, index in (("l", 0), ("r", 1)):
        pred_grf = np.column_stack([_columns_for_side(data, side, f"grf_{axis}") for axis in "xyz"])
        ref_grf = np.column_stack([np.interp(pred_t, ref_t, gt_grf[:, index, axis]) for axis in range(3)])
        pred_cop = _aggregate_side_cop(data, side)
        ref_cop = np.column_stack([np.interp(pred_t, ref_t, gt_cop[:, index, axis]) for axis in range(3)])
        methods = [(method_name, pred_grf)]
        methods.append(("BW_equal_split", bw_baseline))
        com_total = target_force
        if com_total is None and com_acc is not None:
            com_total = mass_kg * (com_acc - np.array([0.0, 0.0, 9.81]))
        if com_total is not None:
            active = np.column_stack([
                np.interp(pred_t, ref_t, gt_grf[:, foot, 2]) > 0.05 * mass_kg * 9.81
                for foot in range(2)
            ])
            active_count = np.maximum(active.sum(axis=1), 1)
            com_baseline = com_total / active_count[:, None]
            com_baseline[~active[:, index], :] = 0.0
            methods.append(("COM_equal_split", com_baseline))
        for method, method_grf in methods:
            total_forces[method] = total_forces.get(method, np.zeros_like(method_grf)) + method_grf
            method_cop = pred_cop if method == method_name else np.full_like(pred_cop, np.nan)
            for row in evaluate_signals(method_grf, ref_grf, method_cop, ref_cop, mass_kg):
                records.append({"subject": pair.subject, "trial": pair.trial, "motion": pair.trial,
                                "side": side, "method": method, **row})
    if target_force is not None:
        for method, total_force in total_forces.items():
            records.append({"subject": pair.subject, "trial": pair.trial, "motion": pair.trial,
                            "side": "total", "method": method, "metric": "physics_residual",
                            "value": _physics_residual(total_force, target_force, mass_kg), "unit": "BW"})
    return records


def _prediction_csv(root: Path, pair: Pair) -> Path:
    candidates = [
        root / pair.subject / f"{pair.motion_path.stem}.csv",
        root / f"{pair.motion_path.stem}.csv",
        root / f"{pair.trial}.csv",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def _force_target_csv(root: Path, pair: Pair) -> Path | None:
    matches = sorted(root.glob(f"{pair.motion_path.stem}_*/results/ID_estimatedGRF/*_contact_wrenches_estimated.csv"))
    return matches[0] if matches else None


def write_metrics(records: list[dict[str, str | float]], output_dir: Path) -> tuple[Path, Path]:
    fields = ["subject", "trial", "motion", "side", "method", "metric", "value", "unit"]
    per_trial = output_dir / "metrics_per_trial.csv"
    with per_trial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in records:
        value = float(row["value"])
        if math.isfinite(value):
            grouped.setdefault((str(row["method"]), str(row["metric"])), []).append(value)
    summary = output_dir / "metrics_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "metric", "value", "unit"])
        writer.writeheader()
        for (method, metric), values in sorted(grouped.items()):
            unit = next(str(row["unit"]) for row in records if row["method"] == method and row["metric"] == metric)
            writer.writerow({"method": method, "metric": metric, "value": float(np.mean(values)), "unit": unit})
    return per_trial, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--force-root", type=Path, default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--mass-kg", type=float, default=69.86)
    parser.add_argument("--height-m", type=float, default=1.68)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/groundlink"))
    parser.add_argument("--mode", choices=("audit", "evaluate"), default="audit")
    parser.add_argument("--pipeline-dir", type=Path, default=None, help="Directory containing <trial>.csv files for evaluate mode.")
    parser.add_argument("--method-name", default="pipeline", help="Label for predictions in --pipeline-dir.")
    parser.add_argument("--com-reference-dir", type=Path, default=None, help="Optional CSV directory supplying common com_acc_* for physics_residual.")
    parser.add_argument("--force-target-dir", type=Path, default=None, help="Optional pipeline directory containing COM-derived contact-wrench targets.")
    args = parser.parse_args()
    pairs = discover_pairs(args.input_root, args.force_root, args.subject)
    manifest = write_manifest(pairs, args.output_dir / "manifest.csv")
    excluded = sum(bool(p.exclusion) for p in pairs)
    missing = sum(p.force_path is None and not p.exclusion for p in pairs)
    force_pairs = sum(p.force_path is not None for p in pairs)
    present = force_pairs - excluded
    invalid = sum(
        row["status"] in {"invalid_force", "frame_mismatch"}
        for row in csv.DictReader(manifest.open(encoding="utf-8"))
    )
    payload = {"commit": _git_commit(), "mode": args.mode, "pairs": len(pairs),
               "force_pairs": force_pairs, "core_force_present": present,
               "missing_force": missing, "excluded": excluded, "invalid": invalid,
               "manifest": str(manifest), "mass_kg": args.mass_kg, "height_m": args.height_m,
               "cutoff_hz": 12.0, "contact_threshold_bw": 0.05,
               "excluded_trials": [p.trial for p in pairs if p.exclusion]}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.mode == "evaluate":
        if args.pipeline_dir is None:
            parser.error("--pipeline-dir is required in evaluate mode")
        records: list[dict[str, str | float]] = []
        for pair in pairs:
            csv_path = _prediction_csv(args.pipeline_dir, pair)
            if csv_path.exists():
                com_reference = _prediction_csv(args.com_reference_dir, pair) if args.com_reference_dir else None
                force_target = _force_target_csv(args.force_target_dir, pair) if args.force_target_dir else None
                records.extend(evaluate_trial(pair, csv_path, args.mass_kg, args.method_name,
                                              com_reference if com_reference and com_reference.exists() else None,
                                              force_target))
        write_metrics(records, args.output_dir)
    print(json.dumps(payload, indent=2))
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())

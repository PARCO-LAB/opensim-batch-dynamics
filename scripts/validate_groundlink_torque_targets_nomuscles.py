#!/usr/bin/env python3
"""Validate a GroundLink no-muscles inverse-dynamics rerun against its source."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from opensim_batch_dynamics.inverse_dynamics_no_grf import _run_opensim_tool, parse_mot
from rerun_groundlink_id_nomuscles import (
    ANKLE_DRIFT_LIMIT_NM,
    ANKLE_DRIFT_LIMIT_P99_PERCENT,
    ANKLE_DRIFT_REPORT_THRESHOLD_NM,
    EXPECTED_COUNTS,
    _ankle_drift_metrics,
    _nearest_index,
    _read_records,
    _read_frame_csv,
    _tool,
    _value_path,
)
from extract_groundlink_torque_targets import TARGET_COORDINATES

EXPECTED_FRAMES = 861_064
PELVIS_LIMIT_NM_OR_N = 1e-6
SWING_MEDIAN_LIMIT_NM = 1e-4
MOMENT_P99_LIMIT_NM_KG = 5.0
MOMENT_REPORT_LIMIT_NM_KG = 4.0
ANKLE_COORDINATES = {"l": "ankle_angle_l", "r": "ankle_angle_r"}
MOMENT_COORDINATES = ("hip_flexion_l", "knee_angle_l", "hip_flexion_r", "knee_angle_r")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _nearest_rank_p99(values: list[float]) -> float:
    ordered = sorted(abs(float(value)) for value in values)
    return ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)] if ordered else 0.0


def _setup_model_and_force_data(
    segment_dir: Path, setup_pattern: str = "Setup_ID_measuredGRF*.xml"
) -> tuple[Path, np.ndarray, dict[str, np.ndarray]]:
    setups = sorted(segment_dir.glob(setup_pattern))
    if len(setups) != 1:
        raise ValueError(f"Expected one measured setup under {segment_dir}")
    tool = _tool(setups[0])
    model = _value_path(tool.findtext("model_file") or "", segment_dir)
    loads_path = _value_path(tool.findtext("external_loads_file") or "", segment_dir)
    loads_root = ET.parse(loads_path).getroot()
    datafile = loads_root.find(".//datafile")
    if datafile is None or not datafile.text:
        raise ValueError(f"External loads have no datafile: {loads_path}")
    mot_path = _value_path(datafile.text.strip(), loads_path.parent)
    labels, rows = parse_mot(mot_path)
    label_index = {label.lower(): index for index, label in enumerate(labels)}
    if "time" not in label_index:
        raise ValueError(f"Measured force MOT has no time column: {mot_path}")
    times = np.asarray([row[label_index["time"]] for row in rows], dtype=float)
    vertical_forces = {}
    for side in ("l", "r"):
        label = f"ground_force_calcn_{side}_vz"
        if label not in label_index:
            raise ValueError(f"Measured force MOT lacks {label}: {mot_path}")
        vertical_forces[side] = np.asarray([row[label_index[label]] for row in rows], dtype=float)
    return model, times, vertical_forces


def _criterion1(output_root: Path) -> dict[str, Any]:
    segment = output_root / "s001/hopping_1/segments/segment_0/id_aligned_all_dofs.csv"
    _, rows = _read_frame_csv(segment)
    row = min(rows, key=lambda value: abs(float(value["time"])))
    expected = {
        "knee_angle_r_tau": 0.2,
        "knee_angle_l_tau": 1.3,
        "hip_adduction_r_tau": -6.8,
        "hip_flexion_r_tau": 0.1,
    }
    observed = {key: float(row[key]) for key in expected}
    errors = {key: abs(observed[key] - value) for key, value in expected.items()}
    tolerance = 1.0
    return {
        "status": "pass" if max(errors.values()) < tolerance else "fail",
        "trial": "s001/hopping_1/segment_0",
        "time_s": float(row["time"]),
        "expected_nm": expected,
        "observed_nm": observed,
        "absolute_error_nm": errors,
        "approximation_tolerance_nm": tolerance,
    }


def _criterion2(source_root: Path, output_root: Path, trial_keys: list[tuple[str, str]]) -> dict[str, Any]:
    entries = []
    violations = []
    for subject, trial in trial_keys:
        source_rows = _rows(source_root / subject / trial / "torque_targets.csv")
        output_rows = _rows(output_root / subject / trial / "torque_targets.csv")
        metrics = _ankle_drift_metrics(source_rows, output_rows)
        entry = {"subject": subject, "trial": trial, **metrics}
        entries.append(entry)
        for side, side_metrics in metrics["sides"].items():
            if not side_metrics["criteria_pass"]:
                violations.append({"subject": subject, "trial": trial, "side": side, **side_metrics})
    return {
        "status": "reported_violations" if violations else "pass",
        "blocking": False,
        "limits": {
            "max_abs_delta_nm_strictly_below": ANKLE_DRIFT_LIMIT_NM,
            "max_delta_over_baseline_abs_tau_p99_percent_strictly_below": ANKLE_DRIFT_LIMIT_P99_PERCENT,
            "changed_frame_threshold_nm": ANKLE_DRIFT_REPORT_THRESHOLD_NM,
        },
        "trial_side_count": sum(2 for _ in entries),
        "violating_trial_sides": violations,
        "per_trial": entries,
    }


def _criterion3(source_root: Path, output_root: Path, trial_keys: list[tuple[str, str]]) -> dict[str, Any]:
    details = []
    global_max = 0.0
    column_names: set[str] = set()
    for subject, trial in trial_keys:
        source_trial = source_root / subject / trial
        output_trial = output_root / subject / trial
        source_segments = sorted((source_trial / "segments").glob("segment_*"), key=lambda p: int(p.name.split("_")[-1]))
        output_segments = sorted((output_trial / "segments").glob("segment_*"), key=lambda p: int(p.name.split("_")[-1]))
        if len(source_segments) != len(output_segments):
            raise ValueError(f"Measured segment count changed for {subject}/{trial}")
        for src, dst in zip(source_segments, output_segments):
            setup_source = next(iter(sorted(src.glob("Setup_ID_measuredGRF*.xml"))), None)
            setup_output = next(iter(sorted(dst.glob("Setup_ID_measuredGRF*.xml"))), None)
            if setup_source is None or setup_output is None:
                raise FileNotFoundError(f"Missing measured setup in {src} or {dst}")
            src_name = Path(_tool(setup_source).findtext("output_gen_force_file") or "").name
            dst_name = Path(_tool(setup_output).findtext("output_gen_force_file") or "").name
            old_path, new_path = src / src_name, dst / dst_name
            old_labels, old_rows = parse_mot(old_path)
            new_labels, new_rows = parse_mot(new_path)
            old_index = {label: i for i, label in enumerate(old_labels)}
            new_index = {label: i for i, label in enumerate(new_labels)}
            names = sorted(
                label for label in old_labels
                if label.lower().startswith("pelvis_") and label.lower().endswith(("_force", "_moment"))
            )
            if not names or not set(names).issubset(new_index) or len(old_rows) != len(new_rows):
                raise ValueError(f"Pelvis residual schema/frame mismatch in {subject}/{trial}/{src.name}")
            old_times = np.asarray([row[old_index["time"]] for row in old_rows])
            new_times = np.asarray([row[new_index["time"]] for row in new_rows])
            if not np.array_equal(old_times, new_times):
                raise ValueError(f"Pelvis residual timestamps changed in {subject}/{trial}/{src.name}")
            delta = max(
                abs(old_rows[i][old_index[name]] - new_rows[i][new_index[name]])
                for i in range(len(old_rows)) for name in names
            )
            global_max = max(global_max, delta)
            column_names.update(names)
            details.append({
                "subject": subject, "trial": trial, "segment": int(src.name.split("_")[-1]),
                "frames": len(old_rows), "pelvis_columns": names, "max_abs_delta": delta,
            })
    return {
        "status": "pass" if global_max < PELVIS_LIMIT_NM_OR_N else "fail",
        "blocking": True,
        "max_abs_delta": global_max,
        "strict_limit": PELVIS_LIMIT_NM_OR_N,
        "columns": sorted(column_names),
        "segments_compared": len(details),
        "per_segment": details,
    }


def _criterion4(output_root: Path, trial_keys: list[tuple[str, str]]) -> dict[str, Any]:
    samples: dict[str, list[float]] = {side: [] for side in ("l", "r")}
    measured_swing_samples: dict[str, list[float]] = {side: [] for side in ("l", "r")}
    qualifying_frames = {side: 0 for side in ("l", "r")}
    measured_swing_frames = {side: 0 for side in ("l", "r")}
    for subject, trial in trial_keys:
        trial_root = output_root / subject / trial
        rows = _rows(trial_root / "torque_targets.csv")
        segment_cache = {}
        for row in rows:
            segment = int(float(row["segment"]))
            if segment not in segment_cache:
                segment_path = trial_root / "segments" / f"segment_{segment}"
                _, measured_times, measured_forces = _setup_model_and_force_data(segment_path)
                _, estimated_times, estimated_forces = _setup_model_and_force_data(
                    segment_path / "estimated_id", "Setup_ID_estimatedGRF*.xml"
                )
                segment_cache[segment] = (measured_times, measured_forces, estimated_times, estimated_forces)
            measured_times, measured_forces, estimated_times, estimated_forces = segment_cache[segment]
            measured_idx = _nearest_index(measured_times, float(row["time"]))
            estimated_idx = _nearest_index(estimated_times, float(row["time"]))
            for side, coordinate in ANKLE_COORDINATES.items():
                if measured_forces[side][measured_idx] <= 20.0:
                    delta = abs(float(row[f"{coordinate}_tau_target"]) - float(row[f"{coordinate}_tau_hat"]))
                    measured_swing_samples[side].append(delta)
                    measured_swing_frames[side] += 1
                    if estimated_forces[side][estimated_idx] <= 20.0:
                        samples[side].append(delta)
                        qualifying_frames[side] += 1
    result = {}
    all_pass = True
    for side in ("l", "r"):
        median = float(np.median(samples[side])) if samples[side] else math.inf
        measured_only_median = (
            float(np.median(measured_swing_samples[side]))
            if measured_swing_samples[side] else math.inf
        )
        passed = bool(samples[side]) and median < SWING_MEDIAN_LIMIT_NM
        all_pass &= passed
        result[side] = {
            "frames_with_measured_and_estimated_vertical_force_le_20_n": qualifying_frames[side],
            "median_abs_ankle_target_hat_delta_nm_both_force_free": median,
            "frames_with_measured_vertical_force_le_20_n": measured_swing_frames[side],
            "median_abs_ankle_target_hat_delta_nm_measured_swing_only": measured_only_median,
            "strict_limit_nm": SWING_MEDIAN_LIMIT_NM,
            "status": "pass" if passed else "fail",
        }
    return {"status": "pass" if all_pass else "fail", "blocking": True, "per_side": result}


def _model_mass_kg(model_path: Path, cache: dict[str, float]) -> float:
    key = str(model_path.resolve())
    if key not in cache:
        model_root = ET.parse(model_path).getroot()
        body_masses = []
        for element in model_root.iter():
            if element.tag.rsplit("}", 1)[-1] != "Body":
                continue
            mass_element = next(
                (child for child in element if child.tag.rsplit("}", 1)[-1] == "mass"),
                None,
            )
            if mass_element is not None and mass_element.text:
                body_masses.append(float(mass_element.text))
        if not body_masses:
            raise ValueError(f"No Body.mass values found in model {model_path}")
        mass = math.fsum(body_masses)
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError(f"Invalid model mass {mass} from {model_path}")
        cache[key] = mass
    return cache[key]


def _criterion5(output_root: Path, trial_keys: list[tuple[str, str]], full_cohort: bool) -> dict[str, Any]:
    if not full_cohort:
        return {"status": "skipped_pilot", "blocking": True}
    by_subject: dict[str, dict[str, list[float]]] = {}
    trial_p99: list[dict[str, Any]] = []
    mass_cache: dict[str, float] = {}
    for subject, trial in trial_keys:
        trial_root = output_root / subject / trial
        rows = _rows(trial_root / "torque_targets.csv")
        model_path = next(iter(sorted((trial_root / "segments").glob("segment_*/model.osim"))), None)
        if model_path is None:
            # The measured setup is authoritative when the model has another filename.
            setup = next(iter(sorted((trial_root / "segments").glob("segment_*/Setup_ID_measuredGRF*.xml"))), None)
            if setup is None:
                raise FileNotFoundError(f"No measured setup for {subject}/{trial}")
            model_path = _value_path(_tool(setup).findtext("model_file") or "", setup.parent)
        mass = _model_mass_kg(model_path, mass_cache)
        trial_values: dict[str, list[float]] = {}
        subject_values = by_subject.setdefault(subject, {f"{coord}_{suffix}": [] for coord in MOMENT_COORDINATES for suffix in ("target", "hat")})
        for coordinate in MOMENT_COORDINATES:
            for suffix in ("target", "hat"):
                key = f"{coordinate}_{suffix}"
                column = f"{coordinate}_tau_{suffix}"
                values = [abs(float(row[column])) / mass for row in rows if int(float(row["valid"])) == 1]
                if not values:
                    raise ValueError(f"No valid frames for {subject}/{trial}/{column}")
                trial_values[key] = values
                subject_values[key].extend(values)
        trial_summary = {key: _nearest_rank_p99(value) for key, value in trial_values.items()}
        above4 = {key: value for key, value in trial_summary.items() if value > MOMENT_REPORT_LIMIT_NM_KG}
        if above4:
            trial_p99.append({"subject": subject, "trial": trial, "mass_kg": mass, "p99_nm_kg": above4})
    subject_results = {}
    all_pass = True
    for subject, signals in by_subject.items():
        p99 = {key: _nearest_rank_p99(value) for key, value in signals.items()}
        passed = all(value < MOMENT_P99_LIMIT_NM_KG for value in p99.values())
        all_pass &= passed
        subject_results[subject] = {
            "p99_abs_tau_nm_kg": p99,
            "maximum_p99_nm_kg": max(p99.values()),
            "status": "pass" if passed else "fail",
        }
    return {
        "status": "pass" if all_pass else "fail",
        "blocking": True,
        "model_mass_method": "sum of Body.mass elements in each saved OpenSim model",
        "unique_models_read": len(mass_cache),
        "p99_limit_strict_nm_kg": MOMENT_P99_LIMIT_NM_KG,
        "trial_report_threshold_nm_kg": MOMENT_REPORT_LIMIT_NM_KG,
        "per_subject": subject_results,
        "trials_above_4_nm_kg": trial_p99,
    }


def _criterion6(source_root: Path, output_root: Path, trial_keys: list[tuple[str, str]], full_cohort: bool) -> dict[str, Any]:
    if not full_cohort:
        return {"status": "skipped_pilot", "blocking": True}
    total_frames = 0
    valid_frames = 0
    cop_invalid_frames = 0
    failures = []
    for subject, trial in trial_keys:
        source_rows = _rows(source_root / subject / trial / "torque_targets.csv")
        output_rows = _rows(output_root / subject / trial / "torque_targets.csv")
        if len(source_rows) != len(output_rows):
            failures.append(f"{subject}/{trial}: frame_count")
            continue
        for old, new in zip(source_rows, output_rows):
            if any(old[key] != new[key] for key in ("frame", "time", "segment")):
                failures.append(f"{subject}/{trial}: frame_time_segment")
                break
            expected_valid = int(float(old["valid"])) and not int(float(new["cop_invalid"]))
            if int(float(new["valid"])) != int(expected_valid):
                failures.append(f"{subject}/{trial}: valid_mask")
                break
            if int(float(new["valid"])):
                for coordinate in TARGET_COORDINATES:
                    for suffix in ("target", "hat"):
                        value = float(new[f"{coordinate}_tau_{suffix}"])
                        if not math.isfinite(value):
                            failures.append(f"{subject}/{trial}: valid_nan_{coordinate}_{suffix}")
                            break
            valid_frames += int(float(new["valid"]))
            cop_invalid_frames += int(float(new["cop_invalid"]))
        total_frames += len(output_rows)
    if total_frames != EXPECTED_FRAMES:
        failures.append(f"total_frames:{total_frames}!={EXPECTED_FRAMES}")
    if len(trial_keys) != 206:
        failures.append(f"trial_count:{len(trial_keys)}!=206")
    return {
        "status": "pass" if not failures else "fail",
        "blocking": True,
        "trials": len(trial_keys),
        "frames": total_frames,
        "expected_frames": EXPECTED_FRAMES,
        "valid_frames": valid_frames,
        "cop_invalid_frames": cop_invalid_frames,
        "failures": failures,
    }


def _criterion7(output_root: Path, pilot_trial: tuple[str, str]) -> dict[str, Any]:
    subject, trial = pilot_trial
    segment_dir = output_root / subject / trial / "segments/segment_0"
    setup_path = next(iter(sorted(segment_dir.glob("Setup_ID_measuredGRF*.xml"))), None)
    if setup_path is None:
        raise FileNotFoundError(f"No measured setup for equivalence check in {segment_dir}")
    original_tool = _tool(setup_path)
    original_output = segment_dir / Path(original_tool.findtext("output_gen_force_file") or "").name
    copied_setup = segment_dir / "Setup_ID_equivalence_Muscles.xml"
    equivalent_output = segment_dir / "equivalence_muscles.sto"
    tree = ET.parse(setup_path)
    try:
        tool = tree.getroot().find("InverseDynamicsTool")
        if tool is None:
            raise ValueError(f"No InverseDynamicsTool in {copied_setup}")
        tool.find("forces_to_exclude").text = "Muscles"
        tool.find("output_gen_force_file").text = equivalent_output.name
        tree.write(copied_setup, encoding="utf-8", xml_declaration=True)
        _run_opensim_tool(copied_setup, working_directory=segment_dir)
        new_labels, new_rows = parse_mot(equivalent_output)
    finally:
        copied_setup.unlink(missing_ok=True)
        equivalent_output.unlink(missing_ok=True)
    old_labels, old_rows = parse_mot(original_output)
    if old_labels != new_labels or len(old_rows) != len(new_rows):
        raise ValueError("Actuators/Muscles equivalence output schemas or frame counts differ")
    old_array = np.asarray(old_rows, dtype=float)
    new_array = np.asarray(new_rows, dtype=float)
    max_delta = float(np.max(np.abs(old_array - new_array)))
    return {
        "status": "pass" if max_delta == 0.0 else "fail",
        "blocking": True,
        "trial": f"{subject}/{trial}/segment_0",
        "max_abs_delta": max_delta,
        "exactly_equal": bool(np.array_equal(old_array, new_array)),
        "forces_to_exclude_compared": ["Actuators", "Muscles"],
    }


def validate(source_root: Path, output_root: Path, index_path: Path, *, pilot: bool = False) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    index_rows = [row for row in _rows(index_path) if row.get("included") == "1"]
    full_cohort = not pilot
    if full_cohort:
        counts = {subject: sum(row["subject"] == subject for row in index_rows) for subject in EXPECTED_COUNTS}
        if len(index_rows) != 206 or counts != EXPECTED_COUNTS:
            raise ValueError(f"Invalid H1 index selection: {len(index_rows)} trials, {counts}")
        trial_keys = [(row["subject"], row["trial"]) for row in index_rows]
    else:
        trial_keys = [("s001", "hopping_1")]
        if not (output_root / "s001/hopping_1/torque_targets.csv").is_file():
            raise FileNotFoundError(f"Pilot trial missing under {output_root}")
    for subject, trial in trial_keys:
        if not (source_root / subject / trial / "torque_targets.csv").is_file():
            raise FileNotFoundError(f"Source trial missing: {subject}/{trial}")
        if not (output_root / subject / trial / "torque_targets.csv").is_file():
            raise FileNotFoundError(f"Output trial missing: {subject}/{trial}")

    checks = {
        "1_reproducibility": _criterion1(output_root),
        "2_ankle_drift": _criterion2(source_root, output_root, trial_keys),
        "3_pelvis_residuals": _criterion3(source_root, output_root, trial_keys),
        "4_swing_target_hat": _criterion4(output_root, trial_keys),
        "5_moment_magnitude": _criterion5(output_root, trial_keys, full_cohort),
        "6_integrity": _criterion6(source_root, output_root, trial_keys, full_cohort),
        "7_actuators_equals_muscles": _criterion7(output_root, ("s001", "hopping_1")),
    }
    blocking_failures = [
        name for name, check in checks.items()
        if check.get("blocking", True) and check.get("status") not in ("pass", "skipped_pilot")
    ]
    report = {
        "protocol": "nomuscles",
        "forces_to_exclude": "Actuators",
        "cohort": "pilot_s001_hopping_1" if pilot else "H1_206_trials",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "checks": checks,
        "blocking_failures": blocking_failures,
        "cohort_status": "blocked" if blocking_failures else "accepted",
    }
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    report_path = logs / "acceptance_checks.json"
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    if checks["2_ankle_drift"]["per_trial"]:
        drift_path = logs / "ankle_drift_per_trial.json"
        drift_path.write_text(
            json.dumps(checks["2_ankle_drift"]["per_trial"], indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    report = validate(args.source_dir, args.output_dir, args.index, pilot=args.pilot)
    print(json.dumps({"cohort_status": report["cohort_status"], "blocking_failures": report["blocking_failures"], "checks": {key: value["status"] for key, value in report["checks"].items()}}, indent=2))
    return 0 if report["cohort_status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())

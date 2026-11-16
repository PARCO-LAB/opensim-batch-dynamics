#!/usr/bin/env python3
"""Rerun saved GroundLink inverse-dynamics setups with muscle actuators excluded."""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from extract_groundlink_torque_targets import (  # noqa: E402
    FORCE_THRESHOLD_N,
    TARGET_COORDINATES,
    _clone_setup_with_forces_to_exclude,
    _read_frame_csv,
    _sha256,
    _shift_mot_add_tcode,
    _verify_mot_add_tcode,
)
from opensim_batch_dynamics.inverse_dynamics_no_grf import (  # noqa: E402
    _run_opensim_tool,
    export_torque_csv_from_id_sto,
    parse_mot,
)
from opensim_batch_dynamics.segments import merge_csv_segments  # noqa: E402

EXPECTED_COUNTS = {"s001": 33, "s002": 33, "s003": 36, "s004": 27, "s005": 35, "s006": 42}
MANIFEST_FIELDS = [
    "subject", "trial", "protocol", "status", "frames", "cutoff_hz",
    "moment_source", "output_csv", "metadata_json", "output_sha256",
    "metadata_sha256", "source_hashes_json", "source_details_json", "error",
]
ANKLE_DRIFT_LIMIT_NM = 2.0
ANKLE_DRIFT_LIMIT_P99_PERCENT = 2.0
ANKLE_DRIFT_REPORT_THRESHOLD_NM = 1e-6
SUBJECT_SEX = {
    "s001": "female", "s002": "male", "s003": "female",
    "s004": "male", "s005": "male", "s006": "female",
}


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    _write_csv(path, MANIFEST_FIELDS, records)


def _load_tcode_reference(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"subject", "trial", "source_frame", "T_x", "T_y", "T_z"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"T_code reference CSV lacks required columns {sorted(required)}: {path}")
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["subject"].strip(), row["trial"].strip())
        if not all(key) or key in result:
            raise ValueError(f"Duplicate or empty T_code reference key in {path}: {key}")
        result[key] = row
    return result


def _tcode_from_marker_min_y(
    marker_min_y: float, reference: dict[str, str], *, tolerance_m: float = 1e-6
) -> tuple[tuple[float, float, float], float]:
    if not math.isfinite(float(marker_min_y)):
        raise ValueError("Raw source marker minimum Y must be finite")
    transform = (0.0, 0.01 - float(marker_min_y), 0.0)
    reference_transform = tuple(float(reference[f"T_{axis}"]) for axis in "xyz")
    if not all(math.isfinite(value) for value in reference_transform):
        raise ValueError("Reference T_code values must be finite")
    max_delta = max(abs(a - b) for a, b in zip(transform, reference_transform))
    if max_delta >= tolerance_m:
        raise ValueError(f"Recomputed T_code differs from B1-bis by {max_delta:.9g} m")
    return transform, max_delta


@functools.lru_cache(maxsize=3)
def _smplx_cpu_model(model_file: str, gender: str, num_betas: int):
    try:
        import smplx
        import torch
    except ImportError as exc:
        raise RuntimeError("T_code recomputation requires torch and smplx on the execution node") from exc
    model = smplx.create(
        model_path=model_file,
        model_type="smplx",
        gender=gender,
        ext="npz",
        use_pca=False,
        flat_hand_mean=False,
        num_betas=num_betas,
        batch_size=1,
    ).to("cpu")
    model.eval()
    return torch, model


def _recompute_tcode_for_trial(
    *,
    subject: str,
    trial: str,
    source_trial: Path,
    dataset_root: Path,
    smplx_model_dir: Path,
    marker_yaml: Path,
    reference: dict[str, str],
) -> tuple[float, float, float, dict[str, Any]]:
    """Rebuild the pipeline's constant Y offset from the source SMPL-X frame."""
    from opensim_batch_dynamics.amass_loader import load_amass_npz
    from opensim_batch_dynamics.bsm_markers import load_bsm_marker_map

    metadata_path = source_trial / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    force_path = Path(metadata["source_paths"]["measured_force"])
    raw_motion = dataset_root / "moshpp" / subject / f"{force_path.stem}_stageii.npz"
    source_frame = int(reference["source_frame"])
    sequence = load_amass_npz(raw_motion)
    if source_frame < 0 or source_frame >= sequence.n_frames:
        raise ValueError(f"Invalid T_code source frame {source_frame} for {subject}/{trial}")
    gender = SUBJECT_SEX[subject]
    model_file = smplx_model_dir / f"SMPLX_{gender.upper()}.npz"
    if not model_file.is_file():
        raise FileNotFoundError(f"Required subject-specific SMPL-X model missing: {model_file}")
    if sequence.betas.size < 16:
        raise ValueError(f"Expected 16 SMPL-X betas in {raw_motion}, got {sequence.betas.size}")
    marker_map = load_bsm_marker_map(marker_yaml)
    marker_ids = np.asarray(list(marker_map.values()), dtype=int)
    torch, model = _smplx_cpu_model(str(model_file), gender, 16)
    def tensor(field: np.ndarray) -> Any:
        return torch.as_tensor(field[source_frame:source_frame + 1], dtype=torch.float32)
    betas = torch.as_tensor(sequence.betas[:16], dtype=torch.float32).reshape(1, 16)
    with torch.no_grad():
        output = model(
            global_orient=tensor(sequence.root_orient),
            body_pose=tensor(sequence.body_pose),
            left_hand_pose=tensor(sequence.left_hand_pose),
            right_hand_pose=tensor(sequence.right_hand_pose),
            jaw_pose=tensor(sequence.jaw_pose),
            leye_pose=tensor(sequence.leye_pose),
            reye_pose=tensor(sequence.reye_pose),
            transl=tensor(sequence.trans),
            betas=betas,
            return_verts=True,
        )
    vertices = output.vertices[0].detach().cpu().numpy()
    if marker_ids.size == 0 or marker_ids.max() >= vertices.shape[0]:
        raise ValueError(f"Invalid BSM marker vertex map for model {model_file}")
    marker_min_y = float(np.min(vertices[marker_ids, 1]))
    transform, max_delta = _tcode_from_marker_min_y(marker_min_y, reference)
    reference_transform = tuple(float(reference[f"T_{axis}"]) for axis in "xyz")
    return (*transform, {
        "subject": subject,
        "trial": trial,
        "source_frame": source_frame,
        "raw_marker_min_y_m": marker_min_y,
        "reference_T_x_m": reference_transform[0],
        "reference_T_y_m": reference_transform[1],
        "reference_T_z_m": reference_transform[2],
        "recomputed_T_x_m": transform[0],
        "recomputed_T_y_m": transform[1],
        "recomputed_T_z_m": transform[2],
        "max_abs_delta_m": max_delta,
        "raw_motion": str(raw_motion),
        "model_file": str(model_file),
        "marker_yaml": str(marker_yaml),
    })


def preflight_tcode_cohort(
    *,
    source_root: Path,
    dataset_root: Path,
    index_path: Path,
    tcode_reference_csv: Path,
    smplx_model_dir: Path,
    marker_yaml: Path,
    audit_csv: Path,
) -> dict[str, Any]:
    """Run the blocking T_code reference check read-only across all 206 trials."""
    with index_path.open(newline="", encoding="utf-8") as handle:
        included = [row for row in csv.DictReader(handle) if row.get("included") == "1"]
    if len(included) != 206:
        raise ValueError(f"H1 index must select exactly 206 trials; got {len(included)}")
    counts = {name: sum(row["subject"] == name for row in included) for name in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected H1 trial counts: {counts}")
    references = _load_tcode_reference(tcode_reference_csv)
    if len(references) != 206:
        raise ValueError(f"B1-bis T_code reference must contain 206 trials; got {len(references)}")
    audit_rows: list[dict[str, Any]] = []
    for row in included:
        subject, trial = row["subject"], row["trial"]
        key = (subject, trial)
        if key not in references:
            raise ValueError(f"T_code reference is missing {subject}/{trial}")
        _, _, _, details = _recompute_tcode_for_trial(
            subject=subject,
            trial=trial,
            source_trial=source_root / subject / trial,
            dataset_root=dataset_root,
            smplx_model_dir=smplx_model_dir,
            marker_yaml=marker_yaml,
            reference=references[key],
        )
        audit_rows.append(details)
        print(
            f"[{len(audit_rows)}/206] {subject}/{trial}: "
            f"max |ΔT|={details['max_abs_delta_m']:.3g} m",
            flush=True,
        )
    fields = list(audit_rows[0])
    _write_csv(audit_csv, fields, audit_rows)
    max_delta = max(float(row["max_abs_delta_m"]) for row in audit_rows)
    if max_delta >= 1e-6:
        raise ValueError(f"Cohort T_code gate failed: max |ΔT|={max_delta:.9g} m")
    return {
        "status": "pass",
        "trials": len(audit_rows),
        "max_abs_delta_m": max_delta,
        "audit_csv": str(audit_csv),
    }


def _tool(setup_path: Path) -> ET.Element:
    tool = ET.parse(setup_path).getroot().find("InverseDynamicsTool")
    if tool is None:
        raise ValueError(f"No InverseDynamicsTool in {setup_path}")
    return tool


def _value_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _segment_number(path: Path) -> int:
    try:
        return int(path.name.removeprefix("segment_"))
    except ValueError as exc:
        raise ValueError(f"Invalid segment directory: {path}") from exc


def _estimated_sources(setup_path: Path) -> tuple[Path, Path, Path, Path]:
    """Resolve persistent copies of files referenced by a saved pipeline setup."""
    tool = _tool(setup_path)
    model_text = tool.findtext("model_file") or ""
    coordinates_text = tool.findtext("coordinates_file") or ""
    external_text = tool.findtext("external_loads_file") or ""
    if not model_text or not coordinates_text or not external_text:
        raise ValueError(f"Incomplete estimated-ID setup: {setup_path}")

    # The setup XML was archived after its temporary engine directory disappeared.
    # The named files remain alongside the setup (model/load) or under results/IK.
    model = setup_path.parent / Path(model_text).name
    results_roots = [parent for parent in setup_path.parents if parent.name == "results"]
    if len(results_roots) != 1:
        raise ValueError(f"Could not identify results/ ancestor for {setup_path}")
    results_root = results_roots[0]
    coordinates = results_root / "IK" / Path(coordinates_text).name
    external_xml = setup_path.parent / Path(external_text).name
    if not model.is_file() or not coordinates.is_file() or not external_xml.is_file():
        raise FileNotFoundError(
            f"Persistent estimated-ID input missing for {setup_path}: "
            f"model={model.is_file()}, coordinates={coordinates.is_file()}, "
            f"external_loads={external_xml.is_file()}"
        )
    external_root = ET.parse(external_xml).getroot()
    datafile = external_root.find(".//datafile")
    if datafile is None or not datafile.text:
        raise ValueError(f"Estimated external-load XML has no datafile: {external_xml}")
    mot = external_xml.parent / Path(datafile.text.strip()).name
    if not mot.is_file():
        raise FileNotFoundError(f"Estimated force MOT missing: {mot}")
    return model.resolve(), coordinates.resolve(), external_xml.resolve(), mot.resolve()


def _cop_invalid_mask(fz: np.ndarray, cop_x: np.ndarray, cop_y: np.ndarray) -> np.ndarray:
    fz = np.asarray(fz, dtype=float)
    cop_x = np.asarray(cop_x, dtype=float)
    cop_y = np.asarray(cop_y, dtype=float)
    if fz.shape != cop_x.shape or fz.shape != cop_y.shape:
        raise ValueError("Fz and CoP components must have matching shapes")
    return (fz > FORCE_THRESHOLD_N) & (np.abs(cop_x) <= 1e-12) & (np.abs(cop_y) <= 1e-12)


def _nearest_rank_p99(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(abs(float(value)) for value in values)
    return ordered[max(0, math.ceil(0.99 * len(ordered)) - 1)]


def _ankle_drift_metrics(
    baseline_rows: list[dict[str, str]], output_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize measured changes in ankle moments against the source cohort."""
    if len(baseline_rows) != len(output_rows):
        raise ValueError("Cannot compare ankle drift with different frame counts")
    report: dict[str, Any] = {
        "limits": {
            "max_abs_delta_nm_strictly_below": ANKLE_DRIFT_LIMIT_NM,
            "max_delta_over_baseline_p99_percent_strictly_below": ANKLE_DRIFT_LIMIT_P99_PERCENT,
            "frame_delta_report_threshold_nm": ANKLE_DRIFT_REPORT_THRESHOLD_NM,
        },
        "sides": {},
    }
    for side in ("l", "r"):
        coordinate = f"ankle_angle_{side}"
        series: dict[str, Any] = {}
        changed_any = np.zeros(len(baseline_rows), dtype=bool)
        criteria_pass = True
        for suffix in ("target", "hat"):
            column = f"{coordinate}_tau_{suffix}"
            baseline = np.asarray([float(row[column]) for row in baseline_rows], dtype=float)
            output = np.asarray([float(row[column]) for row in output_rows], dtype=float)
            delta = np.abs(output - baseline)
            p99 = _nearest_rank_p99(baseline.tolist())
            max_delta = float(np.max(delta)) if len(delta) else 0.0
            percent = (max_delta / p99 * 100.0) if p99 > 0 else (0.0 if max_delta == 0 else math.inf)
            changed_any |= delta > ANKLE_DRIFT_REPORT_THRESHOLD_NM
            series[suffix] = {
                "max_abs_delta_nm": max_delta,
                "baseline_abs_tau_p99_nm": p99,
                "max_delta_over_baseline_p99_percent": percent,
                "frames_abs_delta_gt_1e-6": int(np.count_nonzero(delta > ANKLE_DRIFT_REPORT_THRESHOLD_NM)),
            }
            criteria_pass &= max_delta < ANKLE_DRIFT_LIMIT_NM and percent < ANKLE_DRIFT_LIMIT_P99_PERCENT
        report["sides"][side] = {
            "series": series,
            "max_abs_delta_nm": max(item["max_abs_delta_nm"] for item in series.values()),
            "max_delta_over_baseline_p99_percent": max(
                item["max_delta_over_baseline_p99_percent"] for item in series.values()
            ),
            "frames_with_any_abs_delta_gt_1e-6": int(np.count_nonzero(changed_any)),
            "criteria_pass": bool(criteria_pass),
        }
    return report


def _mot_max_abs_difference(left: Path, right: Path) -> float:
    left_labels, left_rows = parse_mot(left)
    right_labels, right_rows = parse_mot(right)
    if left_labels != right_labels or len(left_rows) != len(right_rows):
        raise ValueError(f"MOT schemas or frame counts differ: {left} / {right}")
    left_array = np.asarray(left_rows, dtype=float)
    right_array = np.asarray(right_rows, dtype=float)
    return float(np.max(np.abs(left_array - right_array))) if left_array.size else 0.0


def _copy_estimated_setup(
    source_setup: Path,
    destination_dir: Path,
    measured_setup: Path,
    forces_to_exclude: str,
) -> tuple[Path, Path, Path, dict[str, dict[str, str]]]:
    estimated_model, estimated_q, source_external_xml, source_mot = _estimated_sources(source_setup)
    measured_tool = _tool(measured_setup)
    measured_model_text = measured_tool.findtext("model_file") or ""
    measured_q_text = measured_tool.findtext("coordinates_file") or ""
    measured_model = _value_path(measured_model_text, measured_setup.parent)
    measured_q = _value_path(measured_q_text, measured_setup.parent)
    if _sha256(estimated_model) != _sha256(measured_model):
        raise ValueError(f"Estimated and measured ID models differ: {source_setup}")
    q_difference = _mot_max_abs_difference(estimated_q, measured_q)
    if q_difference > 1e-12:
        raise ValueError(
            f"Estimated and measured ID coordinates differ by {q_difference:.6g}: {source_setup}"
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    local_mot = destination_dir / source_mot.name
    local_external_xml = destination_dir / source_external_xml.name
    shutil.copy2(source_mot, local_mot)
    external_tree = ET.parse(source_external_xml)
    datafile = external_tree.getroot().find(".//datafile")
    if datafile is None:
        raise ValueError(f"Estimated external-load XML has no datafile: {source_external_xml}")
    datafile.text = local_mot.name
    temp_xml = local_external_xml.with_suffix(local_external_xml.suffix + ".partial")
    external_tree.write(temp_xml, encoding="utf-8", xml_declaration=True)
    temp_xml.replace(local_external_xml)

    source_tool = _tool(source_setup)
    setup_tree = ET.parse(source_setup)
    tool = setup_tree.getroot().find("InverseDynamicsTool")
    assert tool is not None
    path_values = {
        "model_file": os.path.relpath(measured_model, destination_dir),
        "coordinates_file": os.path.relpath(measured_q, destination_dir),
        "external_loads_file": local_external_xml.name,
        "results_directory": ".",
    }
    for tag, value in path_values.items():
        node = tool.find(tag)
        if node is None:
            raise ValueError(f"Missing {tag!r} in estimated setup {source_setup}")
        node.text = value
    output_node = tool.find("output_gen_force_file")
    if output_node is None or not output_node.text:
        raise ValueError(f"Missing output_gen_force_file in {source_setup}")
    output_node.text = Path(output_node.text.strip()).name
    excluded = tool.find("forces_to_exclude")
    if excluded is None:
        excluded = ET.SubElement(tool, "forces_to_exclude")
    excluded.text = forces_to_exclude

    destination_setup = destination_dir / source_setup.name
    temp_setup = destination_setup.with_suffix(destination_setup.suffix + ".partial")
    setup_tree.write(temp_setup, encoding="utf-8", xml_declaration=True)
    temp_setup.replace(destination_setup)
    output_sto = destination_dir / output_node.text
    inputs = {
        "estimated_setup": {"path": str(source_setup), "sha256": _sha256(source_setup)},
        "estimated_model": {"path": str(estimated_model), "sha256": _sha256(estimated_model)},
        "estimated_coordinates": {"path": str(estimated_q), "sha256": _sha256(estimated_q)},
        "coordinates_used_by_nomuscles_id": {
            "path": str(measured_q), "sha256": _sha256(measured_q),
            "max_abs_difference_from_estimated_setup_q": q_difference,
        },
        "estimated_external_loads": {
            "path": str(source_external_xml), "sha256": _sha256(source_external_xml),
        },
        "estimated_force_mot": {"path": str(source_mot), "sha256": _sha256(source_mot)},
    }
    if source_tool.findtext("lowpass_cutoff_frequency_for_coordinates") != tool.findtext(
        "lowpass_cutoff_frequency_for_coordinates"
    ):
        raise AssertionError("Estimated setup coordinate cutoff changed during cloning")
    if source_tool.findtext("time_range") != tool.findtext("time_range"):
        raise AssertionError("Estimated setup time_range changed during cloning")
    return destination_setup, output_sto, local_external_xml, inputs


def _nearest_index(times: np.ndarray, value: float, tolerance: float = 1.1e-5) -> int:
    insertion = int(np.searchsorted(times, value))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(times)]
    if not candidates:
        raise ValueError(f"No ID frame near {value}")
    index = min(candidates, key=lambda candidate: abs(float(times[candidate]) - value))
    if abs(float(times[index]) - value) > tolerance:
        raise ValueError(f"No ID frame within {tolerance}s of {value}")
    return index


def _measured_cop_invalid(segment_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    setup_path = next(iter(sorted(segment_dir.glob("Setup_ID_measuredGRF*.xml"))), None)
    if setup_path is None:
        raise FileNotFoundError(f"No measured setup in {segment_dir}")
    tool = _tool(setup_path)
    loads_text = tool.findtext("external_loads_file") or ""
    loads_path = _value_path(loads_text, setup_path.parent)
    loads_root = ET.parse(loads_path).getroot()
    datafile = loads_root.find(".//datafile")
    if datafile is None or not datafile.text:
        raise ValueError(f"Measured external-load XML has no datafile: {loads_path}")
    mot_path = _value_path(datafile.text.strip(), loads_path.parent)
    labels, rows = parse_mot(mot_path)
    label_index = {label.lower(): index for index, label in enumerate(labels)}
    time_idx = label_index.get("time")
    if time_idx is None:
        raise ValueError(f"Measured MOT has no time column: {mot_path}")
    times = np.asarray([float(row[time_idx]) for row in rows], dtype=float)
    by_side: list[np.ndarray] = []
    for side in ("l", "r"):
        prefix = f"ground_force_calcn_{side}_"
        needed = [prefix + "vz", prefix + "px", prefix + "py"]
        if any(label not in label_index for label in needed):
            raise ValueError(f"Measured MOT lacks CoP/force columns {needed}: {mot_path}")
        fz = np.asarray([float(row[label_index[needed[0]]]) for row in rows])
        px = np.asarray([float(row[label_index[needed[1]]]) for row in rows])
        py = np.asarray([float(row[label_index[needed[2]]]) for row in rows])
        by_side.append(_cop_invalid_mask(fz, px, py))
    foot_invalid = np.column_stack(by_side)
    return times, foot_invalid


def _rows_by_segment_time(
    paths: list[Path],
) -> dict[int, tuple[np.ndarray, list[dict[str, str]]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for path in paths:
        _, rows = _read_frame_csv(path)
        for row in rows:
            segment = int(float(row["segment"]))
            grouped.setdefault(segment, []).append(row)
    result = {}
    for segment, rows in grouped.items():
        rows.sort(key=lambda row: float(row["time"]))
        times = np.asarray([float(row["time"]) for row in rows], dtype=float)
        if np.any(np.diff(times) <= 0):
            raise ValueError(f"Duplicate or unordered times in merged segment {segment}")
        result[segment] = (times, rows)
    return result


def _match_segment_row(
    mapping: dict[int, tuple[np.ndarray, list[dict[str, str]]]], segment: int, time_s: float
) -> dict[str, str]:
    if segment not in mapping:
        raise ValueError(f"No torque rows for segment {segment}")
    times, rows = mapping[segment]
    index = _nearest_index(times, time_s)
    if abs(float(times[index]) - time_s) > 1.1e-5:
        raise ValueError(f"No segment {segment} torque row near {time_s:.9f}s")
    insertion = int(np.searchsorted(times, time_s))
    if 0 < insertion < len(times) and math.isclose(
        abs(float(times[insertion - 1]) - time_s),
        abs(float(times[insertion]) - time_s),
        abs_tol=1e-12,
    ):
        raise ValueError(f"Ambiguous segment {segment} torque row near {time_s:.9f}s")
    return rows[index]


def _write_segment_outputs(
    *,
    setup_path: Path,
    sto_path: Path,
    model_path: Path,
    segment_dir: Path,
    segment_index: int,
    reference_times: np.ndarray,
    suffix: str,
    output_csv: Path,
) -> tuple[Path, Path]:
    aligned_csv = segment_dir / "id_aligned_all_dofs.csv"
    _, _, _, missing = export_torque_csv_from_id_sto(sto_path, model_path, aligned_csv, reference_times)
    missing_targets = sorted(set(TARGET_COORDINATES).intersection(missing))
    if missing_targets:
        raise ValueError(f"ID result {sto_path} is missing target coordinates {missing_targets}")
    _, rows = _read_frame_csv(aligned_csv)
    fields = ["frame", "time", "segment", *[f"{name}_tau_{suffix}" for name in TARGET_COORDINATES]]
    output_rows = []
    for frame, row in enumerate(rows):
        output_rows.append({
            "frame": frame,
            "time": row["time"],
            "segment": segment_index,
            **{f"{name}_tau_{suffix}": row[f"{name}_tau"] for name in TARGET_COORDINATES},
        })
    _write_csv(output_csv, fields, output_rows)
    return aligned_csv, output_csv


def _process_trial(
    *,
    subject: str,
    trial: str,
    source_trial: Path,
    destination_trial: Path,
    pipeline_root: Path,
    forces_to_exclude: str,
) -> dict[str, Any]:
    source_csv = source_trial / "torque_targets.csv"
    source_metadata_path = source_trial / "run_metadata.json"
    if not source_csv.is_file() or not source_metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete source trial directory: {source_trial}")
    source_fields, baseline_rows = _read_frame_csv(source_csv)
    baseline_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if len(baseline_rows) != int(baseline_metadata["frames"]):
        raise ValueError(f"Source trial frame count disagrees with metadata: {source_trial}")
    source_setups = baseline_metadata.get("source_details", {}).get("original_id_setups", [])
    if not source_setups:
        raise ValueError(f"Source metadata has no estimated-ID setup list: {source_trial}")

    segment_dirs = sorted(
        (path for path in (destination_trial / "segments").glob("segment_*") if path.is_dir()),
        key=_segment_number,
    )
    if not segment_dirs:
        raise FileNotFoundError(f"No copied segment folders in {destination_trial}")
    estimated_setups = sorted(
        (Path(item["path"]) for item in source_setups), key=_segment_number_from_setup
    )
    if len(estimated_setups) != len(segment_dirs):
        raise ValueError(
            f"Measured/estimated segment counts differ for {subject}/{trial}: "
            f"{len(segment_dirs)} vs {len(estimated_setups)}"
        )

    target_csvs: list[Path] = []
    estimated_csvs: list[Path] = []
    segment_sources: list[dict[str, Any]] = []
    cop_invalid_by_segment: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    foot_frame_count = 0
    coordinate_cutoffs: list[float] = []

    for segment_dir, source_estimated_setup in zip(segment_dirs, estimated_setups):
        segment_index = _segment_number(segment_dir)
        if not source_estimated_setup.is_file() or not source_estimated_setup.resolve().is_relative_to(
            pipeline_root.resolve()
        ):
            raise FileNotFoundError(f"Estimated setup missing or outside pipeline root: {source_estimated_setup}")
        measured_setups = sorted(segment_dir.glob("Setup_ID_measuredGRF*.xml"))
        if len(measured_setups) != 1:
            raise ValueError(f"Expected one measured setup in {segment_dir}, got {len(measured_setups)}")
        measured_setup = measured_setups[0]
        measured_tool_before = _tool(measured_setup)
        cutoff_text = measured_tool_before.findtext("lowpass_cutoff_frequency_for_coordinates")
        if cutoff_text is None:
            raise ValueError(f"Measured setup lacks coordinate cutoff: {measured_setup}")
        coordinate_cutoffs.append(float(cutoff_text))

        # The source CSV is the exact encoder frame grid for this output segment.
        old_segment_csv = source_trial / "segments" / segment_dir.name / "target_segment.csv"
        _, reference_rows = _read_frame_csv(old_segment_csv)
        reference_times = np.asarray([float(row["time"]) for row in reference_rows], dtype=float)
        if not len(reference_times):
            raise ValueError(f"No reference timestamps in {old_segment_csv}")

        measured_model = _value_path(measured_tool_before.findtext("model_file") or "", segment_dir)
        measured_sto_name = Path(
            measured_tool_before.findtext("output_gen_force_file") or ""
        ).name
        if not measured_sto_name:
            raise ValueError(f"Measured setup lacks output_gen_force_file: {measured_setup}")
        measured_sto = segment_dir / measured_sto_name
        _clone_setup_with_forces_to_exclude(measured_setup, measured_setup, forces_to_exclude)
        _run_opensim_tool(measured_setup, working_directory=segment_dir)
        if not measured_sto.is_file():
            raise FileNotFoundError(f"Measured ID did not create {measured_sto}")
        measured_aligned, target_csv = _write_segment_outputs(
            setup_path=measured_setup,
            sto_path=measured_sto,
            model_path=measured_model,
            segment_dir=segment_dir,
            segment_index=segment_index,
            reference_times=reference_times,
            suffix="target",
            output_csv=segment_dir / "target_segment.csv",
        )
        target_csvs.append(target_csv)

        estimated_dir = segment_dir / "estimated_id"
        estimated_setup, estimated_sto, estimated_xml, estimated_inputs = _copy_estimated_setup(
            source_estimated_setup,
            estimated_dir,
            measured_setup,
            forces_to_exclude,
        )
        _run_opensim_tool(estimated_setup, working_directory=estimated_dir)
        if not estimated_sto.is_file():
            raise FileNotFoundError(f"Estimated ID did not create {estimated_sto}")
        estimated_model = _value_path(_tool(estimated_setup).findtext("model_file") or "", estimated_dir)
        estimated_aligned, estimated_csv = _write_segment_outputs(
            setup_path=estimated_setup,
            sto_path=estimated_sto,
            model_path=estimated_model,
            segment_dir=estimated_dir,
            segment_index=segment_index,
            reference_times=reference_times,
            suffix="hat",
            output_csv=estimated_dir / "estimated_segment.csv",
        )
        estimated_csvs.append(estimated_csv)

        measured_setup_source = source_trial / "segments" / segment_dir.name / measured_setup.name
        measured_external_path = _value_path(
            measured_tool_before.findtext("external_loads_file") or "", segment_dir
        )
        measured_load_node = ET.parse(measured_external_path).getroot().find(".//datafile")
        if measured_load_node is None or not measured_load_node.text:
            raise ValueError(f"Measured external-load XML has no datafile: {measured_external_path}")
        measured_mot = _value_path(measured_load_node.text.strip(), measured_external_path.parent)
        cop_times, cop_foot_invalid = _measured_cop_invalid(segment_dir)
        measured_labels, measured_rows = parse_mot(measured_mot)
        measured_time_index = measured_labels.index("time")
        measured_times = np.asarray(
            [float(row[measured_time_index]) for row in measured_rows], dtype=float
        )
        if not np.array_equal(cop_times, measured_times):
            raise AssertionError(f"Measured MOT timestamps changed while reading: {measured_mot}")
        cop_invalid_by_segment[segment_index] = (cop_times, cop_foot_invalid)
        segment_sources.append({
            "segment": segment_index,
            "measured_setup": {"path": str(measured_setup_source), "sha256": _sha256(measured_setup_source)},
            "measured_external_loads": {"path": str(measured_external_path), "sha256": _sha256(measured_external_path)},
            "measured_force_mot": {"path": str(measured_mot), "sha256": _sha256(measured_mot)},
            "measured_id_sto": {"path": str(measured_sto), "sha256": _sha256(measured_sto)},
            "measured_id_aligned": {"path": str(measured_aligned), "sha256": _sha256(measured_aligned)},
            "estimated_inputs": estimated_inputs,
            "estimated_setup_output": {"path": str(estimated_setup), "sha256": _sha256(estimated_setup)},
            "estimated_external_loads_output": {"path": str(estimated_xml), "sha256": _sha256(estimated_xml)},
            "estimated_force_mot_copy": {"path": str(estimated_dir / Path(estimated_inputs["estimated_force_mot"]["path"]).name), "sha256": _sha256(estimated_dir / Path(estimated_inputs["estimated_force_mot"]["path"]).name)},
            "estimated_id_sto": {"path": str(estimated_sto), "sha256": _sha256(estimated_sto)},
            "estimated_id_aligned": {"path": str(estimated_aligned), "sha256": _sha256(estimated_aligned)},
        })

    if any(cutoff != coordinate_cutoffs[0] for cutoff in coordinate_cutoffs[1:]):
        raise ValueError(f"Coordinate cutoffs differ across segments: {coordinate_cutoffs}")
    target_merged = destination_trial / "target_moments_by_segment.csv"
    estimated_merged = destination_trial / "estimated_moments_by_segment.csv"
    merge_csv_segments(target_csvs, target_merged)
    merge_csv_segments(estimated_csvs, estimated_merged)
    target_values = _rows_by_segment_time([target_merged])
    estimated_values = _rows_by_segment_time([estimated_merged])

    output_fields = ["frame", "time", "segment", "valid", "cop_invalid"]
    output_fields += [f"{name}_tau_target" for name in TARGET_COORDINATES]
    output_fields += [f"{name}_tau_hat" for name in TARGET_COORDINATES]
    output_rows: list[dict[str, Any]] = []
    distinct_invalid_frames = 0
    valid_frames = 0
    for baseline in baseline_rows:
        segment = int(float(baseline["segment"]))
        time_s = float(baseline["time"])
        if segment not in cop_invalid_by_segment:
            raise ValueError(f"Baseline references missing segment {segment}: {source_csv}")
        cop_times, cop_foot_mask = cop_invalid_by_segment[segment]
        foot_mask = cop_foot_mask[_nearest_index(cop_times, time_s)]
        foot_frame_count += int(foot_mask.sum())
        cop_flag = bool(foot_mask.any())
        target = _match_segment_row(target_values, segment, time_s)
        estimate = _match_segment_row(estimated_values, segment, time_s)
        h1_valid = int(float(baseline["valid"]))
        new_valid = int(bool(h1_valid) and not cop_flag)
        distinct_invalid_frames += int(cop_flag)
        valid_frames += new_valid
        output_row: dict[str, Any] = {
            "frame": baseline["frame"],
            "time": baseline["time"],
            "segment": segment,
            "valid": new_valid,
            "cop_invalid": int(cop_flag),
        }
        for name in TARGET_COORDINATES:
            target_column = f"{name}_tau_target"
            estimate_column = f"{name}_tau_hat"
            target_value = float(target[target_column])
            estimate_value = float(estimate[estimate_column])
            if not math.isfinite(target_value) or not math.isfinite(estimate_value):
                raise ValueError(f"Non-finite ID moment for {subject}/{trial}/{name} at {time_s}")
            output_row[target_column] = target_value
            output_row[estimate_column] = estimate_value
        output_rows.append(output_row)
    if len(output_rows) != len(baseline_rows):
        raise ValueError(f"Frame count changed for {subject}/{trial}")
    for old, new in zip(baseline_rows, output_rows):
        if old["frame"] != str(new["frame"]) and int(float(old["frame"])) != int(new["frame"]):
            raise ValueError(f"Frame index changed for {subject}/{trial}")
        if int(float(old["segment"])) != int(new["segment"]) or float(old["time"]) != float(new["time"]):
            raise ValueError(f"Time/segment changed for {subject}/{trial}")
    output_csv = destination_trial / "torque_targets.csv"
    _write_csv(output_csv, output_fields, output_rows)
    ankle_drift = _ankle_drift_metrics(baseline_rows, output_rows)

    metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "protocol": "nomuscles",
        "forces_to_exclude": forces_to_exclude,
        "frames": len(output_rows),
        "valid_frames": valid_frames,
        "cop_invalid_frames": distinct_invalid_frames,
        "cop_invalid_foot_frames": foot_frame_count,
        "cop_invalid_rule": "Fz > 20 N and CoP_x = CoP_y = 0 in measured force MOT; either foot invalid masks the frame",
        "ankle_drift_vs_source": ankle_drift,
        "estimated_id_source": "saved pipeline estimated-GRF setup and MOT; same BSM, IK coordinates, cutoff and time_range",
        "nomuscles_id_segments": segment_sources,
        "effective_coordinate_cutoff_hz": coordinate_cutoffs[0],
        "artifact_files": sorted(
            str(path.relative_to(destination_trial))
            for path in destination_trial.rglob("*")
            if path.is_file() and not path.is_symlink()
            and "Geometry" not in path.relative_to(destination_trial).parts[:-1]
        ),
    })
    metadata_path = destination_trial / "run_metadata.json"
    temp_metadata = metadata_path.with_suffix(".json.partial")
    temp_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temp_metadata.replace(metadata_path)
    source_details = dict(baseline_metadata.get("source_details", {}))
    source_details["nomuscles_id_segments"] = segment_sources
    return {
        "subject": subject,
        "trial": trial,
        "protocol": "nomuscles",
        "status": "ok",
        "frames": len(output_rows),
        "cutoff_hz": coordinate_cutoffs[0],
        "moment_source": "measured_and_saved_estimated_grf",
        "output_csv": str(output_csv),
        "metadata_json": str(metadata_path),
        "output_sha256": _sha256(output_csv),
        "metadata_sha256": _sha256(metadata_path),
        "source_hashes_json": json.dumps({
            "source_torque_targets_sha256": _sha256(source_csv),
            "source_run_metadata_sha256": _sha256(source_metadata_path),
            "measured_setup_sha256": [entry["measured_setup"]["sha256"] for entry in segment_sources],
            "estimated_setup_sha256": [entry["estimated_inputs"]["estimated_setup"]["sha256"] for entry in segment_sources],
            "measured_force_mot_sha256": [entry["measured_force_mot"]["sha256"] for entry in segment_sources],
            "estimated_force_mot_sha256": [entry["estimated_inputs"]["estimated_force_mot"]["sha256"] for entry in segment_sources],
        }, sort_keys=True, separators=(",", ":")),
        "source_details_json": json.dumps(source_details, sort_keys=True, separators=(",", ":")),
        "error": "",
    }


def _tree_file_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            result[str(path.relative_to(root))] = "symlink:" + os.readlink(path)
        elif path.is_file():
            result[str(path.relative_to(root))] = _sha256(path)
    return result


def _column_sha256(rows: list[dict[str, str]], column: str) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row[column].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _process_tcode_trial(
    *,
    subject: str,
    trial: str,
    source_trial: Path,
    destination_trial: Path,
    shift: tuple[float, float, float],
    reference_csv: Path,
    reference_details: dict[str, Any],
) -> dict[str, Any]:
    """Rerun only measured ID after applying T_code to a copied nomuscles trial."""
    source_csv = source_trial / "torque_targets.csv"
    source_metadata_path = source_trial / "run_metadata.json"
    output_csv = destination_trial / "torque_targets.csv"
    metadata_path = destination_trial / "run_metadata.json"
    source_fields, baseline_rows = _read_frame_csv(source_csv)
    if not metadata_path.is_file() or not source_metadata_path.is_file():
        raise FileNotFoundError(f"Missing trial metadata in {source_trial} or {destination_trial}")
    baseline_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(baseline_rows) != int(baseline_metadata["frames"]):
        raise ValueError(f"Source frame count disagrees with metadata: {source_trial}")
    estimated_source_hashes: dict[str, dict[str, str]] = {}
    estimated_output_hashes: dict[str, dict[str, str]] = {}
    target_csvs: list[Path] = []
    cutoff_values: list[float] = []
    for segment_dir in sorted(
        (path for path in (destination_trial / "segments").glob("segment_*") if path.is_dir()),
        key=_segment_number,
    ):
        segment_index = _segment_number(segment_dir)
        source_segment = source_trial / "segments" / segment_dir.name
        estimated_src = source_segment / "estimated_id"
        estimated_dst = segment_dir / "estimated_id"
        estimated_source_hashes[segment_dir.name] = _tree_file_hashes(estimated_src)
        if _tree_file_hashes(estimated_dst) != estimated_source_hashes[segment_dir.name]:
            raise ValueError(f"estimated_id changed while copying {subject}/{trial}/{segment_dir.name}")
        setups = sorted(segment_dir.glob("Setup_ID_measuredGRF*.xml"))
        source_setups = sorted(source_segment.glob("Setup_ID_measuredGRF*.xml"))
        if len(setups) != 1 or len(source_setups) != 1:
            raise ValueError(f"Expected one measured setup in {segment_dir}")
        setup_path, source_setup = setups[0], source_setups[0]
        tool = _tool(setup_path)
        if tool.findtext("forces_to_exclude", "").strip() != "Actuators":
            raise ValueError(f"Source measured setup must already exclude Actuators: {setup_path}")
        setup_sha_before = _sha256(setup_path)
        cutoff_text = tool.findtext("lowpass_cutoff_frequency_for_coordinates")
        if cutoff_text is None:
            raise ValueError(f"Measured setup lacks coordinate cutoff: {setup_path}")
        cutoff_values.append(float(cutoff_text))
        model_text = tool.findtext("model_file") or ""
        measured_model = _value_path(model_text, segment_dir)
        external_text = tool.findtext("external_loads_file") or ""
        external_xml = _value_path(external_text, segment_dir)
        external_root = ET.parse(external_xml).getroot()
        datafile = external_root.find(".//datafile")
        if datafile is None or not datafile.text:
            raise ValueError(f"Measured external-load XML has no datafile: {external_xml}")
        mot = _value_path(datafile.text.strip(), external_xml.parent)
        source_external = _value_path(
            _tool(source_setup).findtext("external_loads_file") or "", source_segment
        )
        source_loads = ET.parse(source_external).getroot().find(".//datafile")
        if source_loads is None or not source_loads.text:
            raise ValueError(f"Source measured external-load XML has no datafile: {source_external}")
        source_mot = _value_path(source_loads.text.strip(), source_external.parent)
        if not mot.resolve().is_relative_to(destination_trial.resolve()):
            raise ValueError(f"Refusing to patch measured MOT outside copied output trial: {mot}")
        if not source_mot.resolve().is_relative_to(source_trial.resolve()):
            raise ValueError(f"Source measured MOT is outside the read-only input trial: {source_mot}")
        if not source_mot.is_file() or not mot.is_file():
            raise FileNotFoundError(f"Measured force MOT missing for {subject}/{trial}/{segment_dir.name}")
        shifted_points = _shift_mot_add_tcode(mot, shift[0], shift[1])
        verified_points = _verify_mot_add_tcode(source_mot, mot, shift[0], shift[1])
        if shifted_points != verified_points:
            raise AssertionError(f"MOT token verification count differs: {mot}")

        source_target_segment = source_segment / "target_segment.csv"
        _, reference_rows = _read_frame_csv(source_target_segment)
        reference_times = np.asarray([float(row["time"]) for row in reference_rows], dtype=float)
        if not len(reference_times):
            raise ValueError(f"No source ID timestamps in {source_target_segment}")
        output_name = Path(tool.findtext("output_gen_force_file") or "").name
        if not output_name:
            raise ValueError(f"Measured setup lacks output_gen_force_file: {setup_path}")
        measured_sto = segment_dir / output_name
        _run_opensim_tool(setup_path, working_directory=segment_dir)
        if not measured_sto.is_file():
            raise FileNotFoundError(f"Measured ID did not create {measured_sto}")
        _, target_segment_csv = _write_segment_outputs(
            setup_path=setup_path,
            sto_path=measured_sto,
            model_path=measured_model,
            segment_dir=segment_dir,
            segment_index=segment_index,
            reference_times=reference_times,
            suffix="target",
            output_csv=segment_dir / "target_segment.csv",
        )
        target_csvs.append(target_segment_csv)
        if _sha256(setup_path) != setup_sha_before or _sha256(source_setup) != setup_sha_before:
            raise ValueError(f"Measured setup changed during T_code rerun: {setup_path}")
        estimated_output_hashes[segment_dir.name] = _tree_file_hashes(estimated_dst)
        if estimated_output_hashes[segment_dir.name] != estimated_source_hashes[segment_dir.name]:
            raise ValueError(f"estimated_id changed during measured-ID rerun: {estimated_dst}")

    if not cutoff_values or any(value != cutoff_values[0] for value in cutoff_values[1:]):
        raise ValueError(f"Measured ID cutoffs differ across segments: {cutoff_values}")
    target_merged = destination_trial / "target_moments_by_segment.csv"
    merge_csv_segments(target_csvs, target_merged)
    target_mapping = _rows_by_segment_time([target_merged])
    output_fields = ["frame", "time", "segment", "valid", "cop_invalid", "side_invalid"]
    output_fields += [f"{name}_tau_target" for name in TARGET_COORDINATES]
    output_fields += [f"{name}_tau_hat" for name in TARGET_COORDINATES]
    output_rows: list[dict[str, str | int]] = []
    mandatory_mask_proposal_rows = 0
    mandatory_mask_valid_proposal_rows = 0
    for baseline in baseline_rows:
        segment = int(float(baseline["segment"]))
        time_s = float(baseline["time"])
        target = _match_segment_row(target_mapping, segment, time_s)
        mandatory_mask_proposal = int(subject == "s001" and trial == "lambadadance_3" and segment == 3)
        mandatory_mask_proposal_rows += mandatory_mask_proposal
        mandatory_mask_valid_proposal_rows += mandatory_mask_proposal * int(float(baseline["valid"]) == 1.0)
        row: dict[str, str | int] = {
            "frame": baseline["frame"],
            "time": baseline["time"],
            "segment": baseline["segment"],
            "valid": baseline["valid"],
            "cop_invalid": baseline["cop_invalid"],
            "side_invalid": 0,
        }
        for name in TARGET_COORDINATES:
            target_column, hat_column = f"{name}_tau_target", f"{name}_tau_hat"
            target_value = target[target_column]
            if not math.isfinite(float(target_value)):
                raise ValueError(f"Non-finite rerun target for {subject}/{trial}/{name}/{time_s}")
            row[target_column] = target_value
            row[hat_column] = baseline[hat_column]
        output_rows.append(row)
    expected_mandatory_mask_rows = 17 if subject == "s001" and trial == "lambadadance_3" else 0
    if mandatory_mask_proposal_rows != expected_mandatory_mask_rows:
        raise ValueError(
            f"Expected {expected_mandatory_mask_rows} mandatory side-invalid frames in "
            f"{subject}/{trial}, found {mandatory_mask_proposal_rows}"
        )
    if len(output_rows) != len(baseline_rows):
        raise ValueError(f"Frame count changed for {subject}/{trial}")
    for column in (f"{name}_tau_hat" for name in TARGET_COORDINATES):
        if _column_sha256(baseline_rows, column) != _column_sha256(output_rows, column):
            raise ValueError(f"Tau-hat column changed during T_code correction: {subject}/{trial}/{column}")
    _write_csv(output_csv, output_fields, output_rows)
    metadata.update({
        "protocol": "nomuscles_copfix_tcode",
        "cop_shift_source": "tcode",
        "cop_shift_m": [float(shift[0]), float(shift[1]), float(shift[2])],
        "cop_shift_reference": str(reference_csv),
        "cop_shift_code_reference": {
            "repository": "opensim-batch-dynamics",
            "file": "scripts/run_amass_to_bsm_csv.py and src/opensim_batch_dynamics/trc_export.py",
            "lines": {
                "scripts/run_amass_to_bsm_csv.py": "316",
                "src/opensim_batch_dynamics/trc_export.py": "103-104",
            },
            "commit": "7107c1bee71d25dac97c6aa3129d243feddebc7b",
        },
        "side_invalid_frames": 0,
        "side_invalid_valid_frames": 0,
        "side_invalid_frames_proposed_mandatory": mandatory_mask_proposal_rows,
        "side_invalid_valid_frames_proposed_mandatory": mandatory_mask_valid_proposal_rows,
        "side_invalid_mask_applied": False,
        "estimated_force_filter_provenance": (
            "tau-hat copied unchanged from _nomuscles torque_targets.csv; originally recalculated "
            "with saved estimated-GRF ID setup and forces_to_exclude=Actuators"
        ),
        "tau_hat_source": "_nomuscles torque_targets.csv, copied unchanged",
        "target_id_regenerated": True,
        "estimated_id_tree_sha256_by_segment": estimated_output_hashes,
        "source_tau_hat_sha256_by_column": {
            f"{name}_tau_hat": _column_sha256(baseline_rows, f"{name}_tau_hat")
            for name in TARGET_COORDINATES
        },
        "reference_recomputed_tcode": reference_details,
        "effective_coordinate_cutoff_hz": cutoff_values[0],
        "output_csv": str(output_csv),
        "frames": len(output_rows),
        "artifact_files": sorted(
            str(path.relative_to(destination_trial))
            for path in destination_trial.rglob("*")
            if path.is_file() and not path.is_symlink()
            and "Geometry" not in path.relative_to(destination_trial).parts[:-1]
        ),
    })
    partial = metadata_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    partial.replace(metadata_path)
    return {
        "subject": subject,
        "trial": trial,
        "protocol": "nomuscles_copfix_tcode",
        "status": "ok",
        "frames": len(output_rows),
        "cutoff_hz": cutoff_values[0],
        "moment_source": "measured T_code; tau-hat copied unchanged",
        "output_csv": str(output_csv),
        "metadata_json": str(metadata_path),
        "output_sha256": _sha256(output_csv),
        "metadata_sha256": _sha256(metadata_path),
        "source_hashes_json": json.dumps({
            "source_torque_targets_sha256": _sha256(source_csv),
            "estimated_id_by_segment": estimated_source_hashes,
            "estimated_id_output_by_segment": estimated_output_hashes,
        }, sort_keys=True, separators=(",", ":")),
        "source_details_json": json.dumps(reference_details, sort_keys=True, separators=(",", ":")),
        "error": "",
    }


def _segment_number_from_setup(path: Path) -> int:
    for parent in path.parents:
        if parent.name.startswith("segment_"):
            return _segment_number(parent)
    import re

    match = re.search(r"_segment_(\d+)\.xml$", path.name)
    return int(match.group(1)) if match else 0


def _read_records(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _record_complete(record: dict[str, str], output_root: Path) -> bool:
    if record.get("status") != "ok" or record.get("protocol") != "nomuscles":
        return False
    csv_path = Path(record.get("output_csv", ""))
    metadata_path = Path(record.get("metadata_json", ""))
    if not csv_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fields, rows = _read_frame_csv(csv_path)
        return (
            metadata.get("forces_to_exclude") == "Actuators"
            and "cop_invalid" in fields
            and len(rows) == int(record.get("frames", 0))
            and _sha256(csv_path) == record.get("output_sha256")
            and _sha256(metadata_path) == record.get("metadata_sha256")
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_subject(
    *,
    source_root: Path,
    pipeline_root: Path,
    index_path: Path,
    output_root: Path,
    subject: str,
    trial_filter: str | None = None,
    forces_to_exclude: str = "Actuators",
    resume: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        all_rows = [row for row in csv.DictReader(handle) if row.get("included") == "1"]
    if len(all_rows) != 206:
        raise ValueError(f"H1 index must select 206 trials; got {len(all_rows)}")
    counts = {name: sum(row["subject"] == name for row in all_rows) for name in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected H1 trial counts: {counts}")
    included = [row for row in all_rows if row["subject"] == subject]
    if trial_filter:
        included = [row for row in included if row["trial"] == trial_filter]
        if len(included) != 1:
            raise ValueError(f"Expected one included trial {subject}/{trial_filter}, got {len(included)}")
    if not included:
        raise ValueError(f"No included trials for {subject}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"execution_manifest_{subject}.csv"
    previous = {
        (row.get("subject", ""), row.get("trial", "")): row
        for row in (_read_records(manifest_path) if resume else [])
        if _record_complete(row, output_root)
    }
    results: list[dict[str, Any]] = []
    failures = 0
    for index, row in enumerate(included, start=1):
        subject_name, trial_name = row["subject"], row["trial"]
        key = (subject_name, trial_name)
        prior = previous.get(key)
        if prior is not None:
            results.append(prior)
            _write_manifest(manifest_path, results)
            print(f"[{index}/{len(included)}] {subject_name}/{trial_name}: resumed", flush=True)
            continue
        source_trial = source_root / subject_name / trial_name
        destination_trial = output_root / subject_name / trial_name
        try:
            if destination_trial.exists():
                shutil.rmtree(destination_trial)
            shutil.copytree(source_trial, destination_trial, symlinks=True, copy_function=shutil.copy2)
            result = _process_trial(
                subject=subject_name,
                trial=trial_name,
                source_trial=source_trial,
                destination_trial=destination_trial,
                pipeline_root=pipeline_root,
                forces_to_exclude=forces_to_exclude,
            )
        except Exception as exc:
            failures += 1
            result = {
                "subject": subject_name, "trial": trial_name, "protocol": "nomuscles",
                "status": "failed", "frames": 0, "cutoff_hz": "",
                "moment_source": "", "output_csv": "", "metadata_json": "",
                "output_sha256": "", "metadata_sha256": "",
                "source_hashes_json": "", "source_details_json": "",
                "error": f"{type(exc).__name__}: {exc}"[:1500],
            }
        results.append(result)
        _write_manifest(manifest_path, results)
        print(
            f"[{index}/{len(included)}] {subject_name}/{trial_name}: "
            f"{result['status']} {result.get('error', '')}",
            flush=True,
        )
    return results, failures


def run_tcode_subject(
    *,
    source_root: Path,
    dataset_root: Path,
    index_path: Path,
    tcode_reference_csv: Path,
    smplx_model_dir: Path,
    marker_yaml: Path,
    output_root: Path,
    subject: str,
    trial_filter: str | None = None,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], int]:
    """Recompute T_code, stage a copied cohort, then rerun measured ID only."""
    with index_path.open(newline="", encoding="utf-8") as handle:
        all_rows = [row for row in csv.DictReader(handle) if row.get("included") == "1"]
    if len(all_rows) != 206:
        raise ValueError(f"H1 index must select 206 trials; got {len(all_rows)}")
    counts = {name: sum(row["subject"] == name for row in all_rows) for name in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected H1 trial counts: {counts}")
    included = [row for row in all_rows if row["subject"] == subject]
    if trial_filter:
        included = [row for row in included if row["trial"] == trial_filter]
        if len(included) != 1:
            raise ValueError(f"Expected one included trial {subject}/{trial_filter}, got {len(included)}")
    if not included:
        raise ValueError(f"No included trials for {subject}")
    references = _load_tcode_reference(tcode_reference_csv)
    missing = [
        f"{row['subject']}/{row['trial']}"
        for row in included
        if (row["subject"], row["trial"]) not in references
    ]
    if missing:
        raise ValueError("T_code reference is missing selected trials: " + ", ".join(missing))

    # Recompute every selected transform before creating or copying any outputs.
    transforms: dict[tuple[str, str], tuple[tuple[float, float, float], dict[str, Any]]] = {}
    audit_rows: list[dict[str, Any]] = []
    for row in included:
        key = (row["subject"], row["trial"])
        source_trial = source_root / key[0] / key[1]
        if not source_trial.is_dir():
            raise FileNotFoundError(f"Source _nomuscles trial missing: {source_trial}")
        dx, dy, dz, details = _recompute_tcode_for_trial(
            subject=key[0],
            trial=key[1],
            source_trial=source_trial,
            dataset_root=dataset_root,
            smplx_model_dir=smplx_model_dir,
            marker_yaml=marker_yaml,
            reference=references[key],
        )
        transforms[key] = ((dx, dy, dz), details)
        audit_rows.append(details)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_root = output_root / "logs"
    _write_csv(
        logs_root / f"tcode_recomputed_{subject}.csv",
        list(audit_rows[0]),
        audit_rows,
    )
    manifest_path = output_root / f"execution_manifest_{subject}.csv"
    prior_records = _read_records(manifest_path) if resume else []
    prior_ok = {
        (record.get("subject", ""), record.get("trial", "")): record
        for record in prior_records
        if record.get("status") == "ok"
    }
    results: list[dict[str, Any]] = []
    failures = 0
    for index, row in enumerate(included, start=1):
        key = (row["subject"], row["trial"])
        source_trial = source_root / key[0] / key[1]
        destination_trial = output_root / key[0] / key[1]
        shift, details = transforms[key]
        previous = prior_ok.get(key)
        if previous is not None and destination_trial.is_dir():
            try:
                metadata = json.loads((destination_trial / "run_metadata.json").read_text(encoding="utf-8"))
                output_path = destination_trial / "torque_targets.csv"
                if (
                    metadata.get("cop_shift_source") == "tcode"
                    and metadata.get("cop_shift_m") == list(shift)
                    and metadata.get("target_id_regenerated") is True
                    and output_path.is_file()
                    and _sha256(output_path) == previous.get("output_sha256")
                    and _sha256(destination_trial / "run_metadata.json") == previous.get("metadata_sha256")
                ):
                    results.append(previous)
                    _write_manifest(manifest_path, results)
                    print(f"[{index}/{len(included)}] {key[0]}/{key[1]}: resumed T_code", flush=True)
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        if destination_trial.exists():
            raise FileExistsError(
                f"Refusing to replace an incomplete or differently configured T_code trial: {destination_trial}"
            )
        try:
            shutil.copytree(source_trial, destination_trial, symlinks=True, copy_function=shutil.copy2)
            result = _process_tcode_trial(
                subject=key[0],
                trial=key[1],
                source_trial=source_trial,
                destination_trial=destination_trial,
                shift=shift,
                reference_csv=tcode_reference_csv,
                reference_details=details,
            )
        except Exception as exc:
            if destination_trial.exists():
                shutil.rmtree(destination_trial)
            failures += 1
            result = {
                "subject": key[0], "trial": key[1], "protocol": "nomuscles_copfix_tcode",
                "status": "failed", "frames": 0, "cutoff_hz": "",
                "moment_source": "", "output_csv": "", "metadata_json": "",
                "output_sha256": "", "metadata_sha256": "", "source_hashes_json": "",
                "source_details_json": "", "error": f"{type(exc).__name__}: {exc}"[:1500],
            }
        results.append(result)
        _write_manifest(manifest_path, results)
        print(f"[{index}/{len(included)}] {key[0]}/{key[1]}: {result['status']} {result.get('error', '')}", flush=True)
    return results, failures


def preflight_inputs(
    *, source_root: Path, pipeline_root: Path, index_path: Path, output_root: Path
) -> dict[str, Any]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        selected = [row for row in csv.DictReader(handle) if row.get("included") == "1"]
    if len(selected) != 206:
        raise ValueError(f"H1 index must select exactly 206 trials; got {len(selected)}")
    counts = {subject: sum(row["subject"] == subject for row in selected) for subject in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected H1 trial counts: {counts}")
    source_manifest = _read_records(source_root / "execution_manifest.csv")
    if len(source_manifest) != 206:
        raise ValueError(f"Source manifest must have 206 trials; got {len(source_manifest)}")
    source_records = {(row["subject"], row["trial"]): row for row in source_manifest}
    if len(source_records) != 206 or any(row.get("status") != "ok" for row in source_manifest):
        raise ValueError("Source manifest has duplicate or unsuccessful trials")
    total_frames = 0
    setup_count = 0
    source_files = 0
    missing_files: list[str] = []
    for selected_row in selected:
        subject, trial = selected_row["subject"], selected_row["trial"]
        manifest_row = source_records.get((subject, trial))
        if manifest_row is None:
            raise ValueError(f"Trial missing from source manifest: {subject}/{trial}")
        source_trial = source_root / subject / trial
        csv_path = source_trial / "torque_targets.csv"
        metadata_path = source_trial / "run_metadata.json"
        if not csv_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Source trial output missing: {source_trial}")
        if _sha256(csv_path) != manifest_row.get("output_sha256"):
            raise ValueError(f"Source torque target hash mismatch: {csv_path}")
        if _sha256(metadata_path) != manifest_row.get("metadata_sha256"):
            raise ValueError(f"Source metadata hash mismatch: {metadata_path}")
        fields, rows = _read_frame_csv(csv_path)
        if not {"frame", "time", "segment", "valid"}.issubset(fields):
            raise ValueError(f"Source torque target schema incomplete: {csv_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if len(rows) != int(metadata.get("frames", -1)):
            raise ValueError(f"Source frame count disagrees with metadata: {csv_path}")
        total_frames += len(rows)
        estimated_details = metadata.get("source_details", {}).get("original_id_setups", [])
        if not estimated_details:
            raise ValueError(f"No estimated ID setup metadata: {metadata_path}")
        segment_dirs = sorted(
            (path for path in (source_trial / "segments").glob("segment_*") if path.is_dir()),
            key=_segment_number,
        )
        if len(segment_dirs) != len(estimated_details):
            raise ValueError(f"Measured/estimated setup count mismatch: {source_trial}")
        estimated_paths = [Path(item["path"]) for item in estimated_details]
        estimated_paths.sort(key=_segment_number_from_setup)
        for segment_dir, estimated_setup in zip(segment_dirs, estimated_paths):
            setup_count += 1
            if not estimated_setup.is_file() or not estimated_setup.resolve().is_relative_to(pipeline_root.resolve()):
                raise FileNotFoundError(f"Estimated setup absent/outside pipeline: {estimated_setup}")
            details_for_segment = next(
                item for item in estimated_details
                if _segment_number_from_setup(Path(item["path"])) == _segment_number(segment_dir)
            )
            if _sha256(estimated_setup) != details_for_segment.get("sha256"):
                raise ValueError(f"Estimated setup hash mismatch: {estimated_setup}")
            measured_setups = sorted(segment_dir.glob("Setup_ID_measuredGRF*.xml"))
            if len(measured_setups) != 1:
                raise ValueError(f"Expected one measured setup in {segment_dir}")
            measured_setup = measured_setups[0]
            measured_tool = _tool(measured_setup)
            estimated_tool = _tool(estimated_setup)
            if measured_tool.findtext("forces_to_exclude", "").strip():
                raise ValueError(f"Source measured setup already excludes forces: {measured_setup}")
            if estimated_tool.findtext("forces_to_exclude", "").strip():
                raise ValueError(f"Source estimated setup already excludes forces: {estimated_setup}")
            measured_cutoff = measured_tool.findtext("lowpass_cutoff_frequency_for_coordinates")
            estimated_cutoff = estimated_tool.findtext("lowpass_cutoff_frequency_for_coordinates")
            if measured_cutoff != estimated_cutoff:
                raise ValueError(f"Measured/estimated cutoff mismatch: {measured_setup} / {estimated_setup}")
            model, q, external_xml, mot = _estimated_sources(estimated_setup)
            measured_model = _value_path(measured_tool.findtext("model_file") or "", segment_dir)
            measured_q = _value_path(measured_tool.findtext("coordinates_file") or "", segment_dir)
            if _sha256(model) != _sha256(measured_model):
                raise ValueError(f"Measured/estimated models differ for {subject}/{trial}/segment_{_segment_number(segment_dir)}")
            q_difference = _mot_max_abs_difference(q, measured_q)
            if q_difference > 1e-12:
                raise ValueError(
                    f"Measured/estimated q differ by {q_difference:.6g} for "
                    f"{subject}/{trial}/segment_{_segment_number(segment_dir)}"
                )
            source_files += 5
            if not external_xml.is_file() or not mot.is_file():
                missing_files.extend([str(external_xml), str(mot)])
    if missing_files:
        raise FileNotFoundError("Missing estimated force sources: " + ", ".join(missing_files[:10]))
    if total_frames != 861_064:
        raise ValueError(f"Source cohort has {total_frames} frames; expected 861064")
    if setup_count != 541:
        raise ValueError(f"Source cohort has {setup_count} ID segments; expected 541")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Nomuscles output is already non-empty: {output_root}")
    return {
        "status": "ready",
        "trials": len(selected),
        "per_subject": counts,
        "frames": total_frames,
        "segments": setup_count,
        "estimated_inputs_per_segment_verified": source_files,
        "output_root": str(output_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--pipeline-root", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subject", choices=tuple(EXPECTED_COUNTS))
    parser.add_argument("--trial")
    parser.add_argument("--forces-to-exclude", choices=("Actuators", "Muscles"), default="Actuators")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not args.source_dir or not args.pipeline_root or not args.index:
        parser.error("--source-dir, --pipeline-root, and --index are required")
    if args.preflight:
        report = preflight_inputs(
            source_root=args.source_dir.resolve(),
            pipeline_root=args.pipeline_root.resolve(),
            index_path=args.index.resolve(),
            output_root=args.output_dir.resolve(),
        )
        print(json.dumps(report, indent=2))
        return 0
    if not args.subject:
        parser.error("--subject is required unless --preflight is used")
    records, failures = run_subject(
        source_root=args.source_dir.resolve(),
        pipeline_root=args.pipeline_root.resolve(),
        index_path=args.index.resolve(),
        output_root=args.output_dir.resolve(),
        subject=args.subject,
        trial_filter=args.trial,
        forces_to_exclude=args.forces_to_exclude,
        resume=args.resume,
    )
    print(json.dumps({"requested": len(records), "succeeded": len(records) - failures, "failed": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

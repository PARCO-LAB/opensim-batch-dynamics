#!/usr/bin/env python3
"""Extract measured-GRF GroundLink torque targets with saved ID setups."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from opensim_batch_dynamics.inverse_dynamics_no_grf import (  # noqa: E402
    _prepare_model_for_inverse_dynamics,
    _run_opensim_tool,
    export_torque_csv_from_id_sto,
    parse_mot,
)
from opensim_batch_dynamics.mot_to_csv import _lowpass_butterworth_4th  # noqa: E402
from opensim_batch_dynamics.segments import (  # noqa: E402
    discover_segment_mot_paths,
    keep_segment_time,
    merge_csv_segments,
)

TARGET_COORDINATES = (
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
)
H1_FORCE_CUTOFF_HZ = 12.0
FORCE_THRESHOLD_N = 20.0
_MOT_TOKEN = re.compile(r"\S+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hardlink_or_copy(source: str, destination: str) -> str:
    """Reuse immutable pipeline artifacts when possible; copy across filesystems."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _ensure_local_geometry(source: Path, destination: Path) -> None:
    """Put a real Geometry directory beside each copied OpenSim model."""
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Geometry artifact path already exists and is not a directory: {destination}")
        return
    if source.is_dir():
        shutil.copytree(source, destination, copy_function=_hardlink_or_copy)


def _ensure_local_coordinates(source: Path, output_dir: Path) -> Path:
    """Keep the exact input IK MOT beside the setup that consumes it."""
    try:
        source.resolve().relative_to(output_dir.resolve())
    except ValueError:
        destination = output_dir / "coordinates_input.mot"
        if not destination.exists():
            _hardlink_or_copy(str(source), str(destination))
        return destination
    return source


def _numeric_array(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


def _load_measured_wrenches(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[dict[str, int | str]]]:
    """Load and prepare one GroundLink force file on its native 250 Hz grid."""
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        data = {name: loaded[name] for name in loaded.files}
        loaded.close()
    elif isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object:
        data = loaded.item()
    else:
        raise ValueError(f"Expected a GroundLink object-array force file: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary in GroundLink force file: {path}")
    cop = _numeric_array(data.get("CoP", data.get("cop")))
    raw_grf = _numeric_array(data.get("GRF", data.get("grf", data.get("force"))))
    if raw_grf.shape != cop.shape or raw_grf.ndim != 3 or raw_grf.shape[1:] != (2, 3):
        raise ValueError(f"Expected CoP/GRF with shape (T,2,3); got {cop.shape}/{raw_grf.shape}")
    if not np.all(np.isfinite(cop)) or not np.all(np.isfinite(raw_grf)):
        raise ValueError(f"Non-finite measured GRF or CoP in {path}")

    grf = raw_grf * 1000.0  # GroundLink stores GRF in kN.
    free_moment = np.zeros((len(grf), 2), dtype=float)
    moment_key = next(
        (key for key in ("free_moment", "freeMoment", "FreeMoment", "M") if key in data),
        None,
    )
    moment_source = "zero_no_free_moment_field"
    if moment_key is not None:
        moment = _numeric_array(data[moment_key])
        if moment.shape == (len(grf), 2, 3):
            free_moment = moment[:, :, 2]
        elif moment.shape == (len(grf), 2):
            free_moment = moment
        else:
            raise ValueError(
                f"Free moment {moment_key!r} must have shape (T,2,3) or (T,2); got {moment.shape}"
            )
        unit_key = next((key for key in ("free_moment_unit", "freeMomentUnit", "M_unit") if key in data), None)
        unit = str(data[unit_key]).strip().lower() if unit_key else "nm"
        if unit in {"knm", "kn*m", "kn·m"}:
            free_moment = free_moment * 1000.0
        elif unit not in {"nm", "n*m", "n·m"}:
            raise ValueError(f"Unknown free-moment unit {unit!r} in {path}")
        if not np.all(np.isfinite(free_moment)):
            raise ValueError(f"Non-finite free moment in {path}")
        moment_source = f"{moment_key}:{unit}"

    raw_contact = grf[:, :, 2] >= FORCE_THRESHOLD_N
    grf[~raw_contact] = 0.0
    cop[~raw_contact] = 0.0
    free_moment[~raw_contact] = 0.0
    sample_rate = 250.0
    for foot in range(2):
        for axis in range(3):
            grf[:, foot, axis] = _lowpass_butterworth_4th(
                grf[:, foot, axis], sample_rate, H1_FORCE_CUTOFF_HZ
            )
        free_moment[:, foot] = _lowpass_butterworth_4th(
            free_moment[:, foot], sample_rate, H1_FORCE_CUTOFF_HZ
        )

    # Filter CoP within each loaded interval, never across the zero-valued
    # flight/swing periods. Very short contact intervals cannot support the
    # zero-phase filter and retain their measured CoP.
    short_cop_runs: list[dict[str, int | str]] = []
    for foot in range(2):
        active = raw_contact[:, foot]
        edges = np.diff(np.r_[False, active, False].astype(np.int8))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for start, stop in zip(starts, stops):
            run_length = int(stop - start)
            if run_length < 3:
                short_cop_runs.append({
                    "foot": "l" if foot == 0 else "r",
                    "start_index": int(start),
                    "stop_index_exclusive": int(stop),
                    "samples": run_length,
                })
                continue
            for axis in range(3):
                values = cop[start:stop, foot, axis]
                if run_length >= 16:
                    filtered = _lowpass_butterworth_4th(
                        values, sample_rate, H1_FORCE_CUTOFF_HZ
                    )
                else:
                    # Keep the specified 4th-order filter for 3–15 sample
                    # contacts by shortening filtfilt's edge padding.
                    from scipy.signal import butter, filtfilt

                    b, a = butter(4, H1_FORCE_CUTOFF_HZ, btype="low", fs=sample_rate)
                    filtered = filtfilt(b, a, values, padlen=min(15, run_length - 1))
                cop[start:stop, foot, axis] = filtered
    return cop, grf, free_moment, moment_source, short_cop_runs


def _resample_wrenches(
    cop_source: np.ndarray,
    grf_source: np.ndarray,
    moment_source: np.ndarray,
    times: np.ndarray,
    cop_shift_m: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_times = np.arange(len(grf_source), dtype=float) / 250.0
    if not times.size or times[0] < source_times[0] - 1e-8 or times[-1] > source_times[-1] + 1e-8:
        raise ValueError("Measured force file does not cover the complete IK time range.")

    def interpolate(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [np.interp(times, source_times, values[:, foot, axis])
             for foot in range(2) for axis in range(values.shape[2])],
            axis=1,
        ).reshape(len(times), 2, values.shape[2])

    grf = interpolate(grf_source)
    cop = interpolate(cop_source)
    moment = np.stack(
        [np.interp(times, source_times, moment_source[:, foot]) for foot in range(2)], axis=1
    )
    low_force = grf[:, :, 2] < FORCE_THRESHOLD_N
    grf[low_force] = 0.0
    cop[low_force] = 0.0
    moment[low_force] = 0.0
    if cop_shift_m is not None:
        shift = np.asarray(cop_shift_m, dtype=float)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError(f"CoP shift must contain two finite values in metres; got {cop_shift_m!r}")
        valid_cop = (~low_force) & np.any(cop[:, :, :2] != 0.0, axis=2)
        cop[valid_cop, :2] -= shift[None, :]
    # OpenSim's ground CoP is on the ground plane in this model.
    cop[:, :, 2] = 0.0
    return cop, grf, moment


def _load_cop_shifts(path: Path | None) -> dict[tuple[str, str], tuple[float, float]] | None:
    """Load one constant CoP shift per trial from a CSV with dx_m/dy_m columns."""
    if path is None:
        return None
    shifts: dict[tuple[str, str], tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"subject", "trial", "dx_m", "dy_m"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CoP shift CSV must contain columns {sorted(required)}: {path}")
        for row in reader:
            key = (row["subject"].strip(), row["trial"].strip())
            if not all(key):
                raise ValueError(f"Empty subject/trial key in CoP shift CSV: {path}")
            if key in shifts:
                raise ValueError(f"Duplicate CoP shift for {key[0]}/{key[1]} in {path}")
            try:
                shift = (float(row["dx_m"]), float(row["dy_m"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid CoP shift for {key[0]}/{key[1]} in {path}") from exc
            if not np.all(np.isfinite(shift)):
                raise ValueError(f"Non-finite CoP shift for {key[0]}/{key[1]} in {path}")
            shifts[key] = shift
    return shifts


def _mot_data_layout(lines: list[str], path: Path) -> tuple[int, list[str], list[int]]:
    end_header = next(
        (index for index, line in enumerate(lines) if line.strip().lower() == "endheader"),
        None,
    )
    if end_header is None or end_header + 1 >= len(lines):
        raise ValueError(f"Invalid OpenSim MOT header: {path}")
    columns = lines[end_header + 1].split()
    data_lines = [index for index in range(end_header + 2, len(lines)) if lines[index].strip()]
    return end_header, columns, data_lines


def _shift_mot_add_tcode(path: Path, dx: float, dy: float) -> int:
    """Add the Phase C trial translation to valid measured CoP tokens in place."""
    if path.is_symlink():
        raise ValueError(f"Measured force MOT must be a regular copied file: {path}")
    shift = np.asarray([dx, dy], dtype=float)
    if shift.shape != (2,) or not np.all(np.isfinite(shift)):
        raise ValueError(f"T_code must contain two finite values in metres; got {shift!r}")
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    _, columns, data_lines = _mot_data_layout(lines, path)
    indexes: dict[str, int] = {}
    for side in ("l", "r"):
        for suffix in ("vz", "px", "py"):
            name = f"ground_force_calcn_{side}_{suffix}"
            if name not in columns:
                raise ValueError(f"Missing MOT column {name} in {path}")
            indexes[name] = columns.index(name)
    shifted_points = 0
    for line_index in data_lines:
        line = lines[line_index]
        matches = list(_MOT_TOKEN.finditer(line))
        if len(matches) != len(columns):
            raise ValueError(f"MOT row/column mismatch in {path}:{line_index + 1}")
        replacements: list[tuple[int, int, str]] = []
        for side in ("l", "r"):
            fz = float(matches[indexes[f"ground_force_calcn_{side}_vz"]].group())
            x_match = matches[indexes[f"ground_force_calcn_{side}_px"]]
            y_match = matches[indexes[f"ground_force_calcn_{side}_py"]]
            x, y = float(x_match.group()), float(y_match.group())
            if fz <= FORCE_THRESHOLD_N or (x == 0.0 and y == 0.0):
                continue
            if dx != 0.0:
                replacements.append((x_match.start(), x_match.end(), f"{x + dx:.12f}"))
            if dy != 0.0:
                replacements.append((y_match.start(), y_match.end(), f"{y + dy:.12f}"))
            shifted_points += int(dx != 0.0 or dy != 0.0)
        for start, end, value in sorted(replacements, reverse=True):
            line = line[:start] + value + line[end:]
        lines[line_index] = line
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text("".join(lines), encoding="utf-8", newline="")
    os.replace(temporary, path)
    return shifted_points


def _verify_mot_add_tcode(source: Path, output: Path, dx: float, dy: float) -> int:
    """Verify every MOT token except valid contact px/py has remained identical."""
    source_lines = source.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    output_lines = output.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    _, columns, source_data_lines = _mot_data_layout(source_lines, source)
    _, output_columns, output_data_lines = _mot_data_layout(output_lines, output)
    if columns != output_columns or len(source_lines) != len(output_lines):
        raise ValueError(f"MOT structure changed during T_code correction: {source} / {output}")
    if source_data_lines != output_data_lines:
        raise ValueError(f"MOT row count changed during T_code correction: {source} / {output}")
    indexes = {
        name: columns.index(name)
        for side in ("l", "r")
        for name in (
            f"ground_force_calcn_{side}_vz",
            f"ground_force_calcn_{side}_px",
            f"ground_force_calcn_{side}_py",
        )
    }
    cop_indexes = {
        indexes[f"ground_force_calcn_{side}_{suffix}"]
        for side in ("l", "r")
        for suffix in ("px", "py")
    }
    changed_points = 0
    data_set = set(source_data_lines)
    for line_index, (source_line, output_line) in enumerate(zip(source_lines, output_lines)):
        source_matches = list(_MOT_TOKEN.finditer(source_line))
        output_matches = list(_MOT_TOKEN.finditer(output_line))
        if line_index not in data_set:
            if source_line != output_line:
                raise ValueError(f"Non-data MOT line changed at {source}:{line_index + 1}")
            continue
        if len(source_matches) != len(columns) or len(output_matches) != len(columns):
            raise ValueError(f"MOT row/column mismatch during T_code verification: {source}:{line_index + 1}")
        for side in ("l", "r"):
            fz = float(source_matches[indexes[f"ground_force_calcn_{side}_vz"]].group())
            x_index, y_index = indexes[f"ground_force_calcn_{side}_px"], indexes[f"ground_force_calcn_{side}_py"]
            x_token, y_token = source_matches[x_index].group(), source_matches[y_index].group()
            x, y = float(x_token), float(y_token)
            valid = fz > FORCE_THRESHOLD_N and not (x == 0.0 and y == 0.0)
            for index, original, delta in ((x_index, x_token, dx), (y_index, y_token, dy)):
                actual = output_matches[index].group()
                if valid and delta != 0.0:
                    expected = f"{float(original) + delta:.12f}"
                    if actual != expected:
                        raise ValueError(f"Unexpected shifted CoP token at {output}:{line_index + 1}")
                elif actual != original:
                    raise ValueError(f"Unexpected MOT token change at {output}:{line_index + 1}")
            for index, (original, actual) in enumerate(zip(source_matches, output_matches)):
                if index in cop_indexes:
                    continue
                if original.group() != actual.group():
                    raise ValueError(f"Non-CoP MOT token changed at {output}:{line_index + 1}")
            changed_points += int(valid and (dx != 0.0 or dy != 0.0))
    return changed_points


def _find_unique(pattern: str, root: Path, description: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"Expected one {description} matching {pattern!r} under {root}, found {len(matches)}")
    return matches[0]


def _write_measured_load_files(
    output_dir: Path,
    trial_name: str,
    times: np.ndarray,
    grf: np.ndarray,
    cop: np.ndarray,
    free_moment: np.ndarray,
) -> tuple[Path, Path]:
    mot_path = output_dir / f"{trial_name}_grf_measured.mot"
    xml_path = output_dir / f"{trial_name}_external_forces_measured.xml"
    labels = ["time"]
    for side in ("l", "r"):
        body = f"calcn_{side}"
        labels.extend([f"ground_force_{body}_v{axis}" for axis in "xyz"])
        labels.extend([f"ground_force_{body}_p{axis}" for axis in "xyz"])
        labels.extend([f"ground_force_{body}_m{axis}" for axis in "xyz"])
    with mot_path.open("w", encoding="utf-8") as handle:
        handle.write(f"name {trial_name}_grf_measured\n")
        handle.write(f"datacolumns {len(labels)}\ndatarows {len(times)}\n")
        handle.write(f"range {times[0]:.10g} {times[-1]:.10g}\nendheader\n")
        handle.write(" ".join(labels) + "\n")
        for i, timestamp in enumerate(times):
            row = [float(timestamp)]
            for foot in range(2):
                row.extend(grf[i, foot].tolist())
                row.extend(cop[i, foot].tolist())
                row.extend((0.0, 0.0, float(free_moment[i, foot])))
            handle.write(" ".join(f"{value:.12g}" for value in row) + "\n")

    root = ET.Element("OpenSimDocument", {"Version": "40000"})
    loads = ET.SubElement(root, "ExternalLoads", {"name": trial_name})
    objects = ET.SubElement(loads, "objects")
    for side in ("l", "r"):
        body = f"calcn_{side}"
        force = ET.SubElement(objects, "ExternalForce", {"name": f"ForcePlate_{body}"})
        for name, value in (
            ("applied_to_body", body),
            ("force_expressed_in_body", "ground"),
            ("point_expressed_in_body", "ground"),
            ("force_identifier", f"ground_force_{body}_v"),
            ("point_identifier", f"ground_force_{body}_p"),
            ("torque_identifier", f"ground_force_{body}_m"),
        ):
            ET.SubElement(force, name).text = value
    ET.SubElement(loads, "datafile").text = mot_path.name
    ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)
    return mot_path, xml_path


def _patch_original_setup(
    setup_path: Path,
    destination: Path,
    model_path: Path,
    ik_path: Path,
    external_loads_path: Path,
    output_sto: Path,
    cutoff_override_hz: float | None = None,
    forces_to_exclude: str | None = None,
) -> float:
    """Clone an ID setup, retaining every numeric/tool option from the source."""
    tree = ET.parse(setup_path)
    tool = tree.getroot().find("InverseDynamicsTool")
    if tool is None:
        raise ValueError(f"No InverseDynamicsTool in {setup_path}")
    setup_directory = destination.parent.resolve()
    replacements = {
        "model_file": os.path.relpath(model_path.resolve(), setup_directory),
        "coordinates_file": os.path.relpath(ik_path.resolve(), setup_directory),
        "external_loads_file": os.path.relpath(external_loads_path.resolve(), setup_directory),
        "results_directory": ".",
        "output_gen_force_file": output_sto.name,
    }
    for tag, value in replacements.items():
        element = tool.find(tag)
        if element is None:
            raise ValueError(f"Missing {tag!r} in original ID setup {setup_path}")
        element.text = value
    if forces_to_exclude is not None:
        excluded = tool.find("forces_to_exclude")
        if excluded is None:
            excluded = ET.SubElement(tool, "forces_to_exclude")
        excluded.text = forces_to_exclude
    cutoff_text = tool.findtext("lowpass_cutoff_frequency_for_coordinates")
    if cutoff_text is None:
        raise ValueError(f"Original ID setup has no coordinate cutoff: {setup_path}")
    cutoff = float(cutoff_text)
    if not math.isfinite(cutoff):
        raise ValueError(f"Invalid coordinate cutoff in {setup_path}: {cutoff_text}")
    if cutoff_override_hz is not None:
        if not math.isfinite(cutoff_override_hz):
            raise ValueError(f"Invalid coordinate cutoff override: {cutoff_override_hz}")
        tool.find("lowpass_cutoff_frequency_for_coordinates").text = str(cutoff_override_hz)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return cutoff


def _clone_setup_with_forces_to_exclude(
    setup_path: Path,
    destination: Path,
    forces_to_exclude: str,
) -> None:
    """Copy an ID setup changing only its force-exclusion setting."""
    tree = ET.parse(setup_path)
    tool = tree.getroot().find("InverseDynamicsTool")
    if tool is None:
        raise ValueError(f"No InverseDynamicsTool in {setup_path}")
    excluded = tool.find("forces_to_exclude")
    if excluded is None:
        excluded = ET.SubElement(tool, "forces_to_exclude")
    excluded.text = forces_to_exclude
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    tree.write(temporary, encoding="utf-8", xml_declaration=True)
    temporary.replace(destination)


def _read_target_sto(
    sto_path: Path,
    model_path: Path,
    output_path: Path,
    segment_index: int,
    reference_times: np.ndarray,
) -> None:
    aligned_csv = output_path.with_name("id_aligned_all_dofs.csv")
    _, _, dofs, missing = export_torque_csv_from_id_sto(
        sto_path, model_path, aligned_csv, reference_times
    )
    missing_targets = sorted(set(TARGET_COORDINATES).intersection(missing))
    if missing_targets:
        raise ValueError(f"ID output {sto_path} is missing target coordinates {missing_targets}")
    _, rows = _read_frame_csv(aligned_csv)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time", "segment", *[f"{name}_tau_target" for name in TARGET_COORDINATES]])
        for frame, row in enumerate(rows):
            writer.writerow([
                frame,
                row["time"],
                segment_index,
                *(row[f"{name}_tau"] for name in TARGET_COORDINATES),
            ])


def _max_abs_id_csv_difference(
    reference_path: Path,
    comparison_path: Path,
    reference_suffix: str,
    comparison_suffix: str,
    subject: str,
    trial: str,
    segment_index: int,
) -> float:
    reference_fields, reference_rows = _read_frame_csv(reference_path)
    comparison_fields, comparison_rows = _read_frame_csv(comparison_path)
    if len(reference_rows) != len(comparison_rows):
        raise ValueError(f"Identical-load ID row counts differ for {subject}/{trial}")
    errors: list[float] = []
    for name in TARGET_COORDINATES:
        reference_name = f"{name}_tau_{reference_suffix}"
        comparison_name = f"{name}_tau_{comparison_suffix}"
        if reference_name not in reference_fields or comparison_name not in comparison_fields:
            raise ValueError(
                f"Identical-load ID is missing {name} for {subject}/{trial}/segment_{segment_index}"
            )
        reference_values = np.asarray([float(row[reference_name]) for row in reference_rows])
        comparison_values = np.asarray([float(row[comparison_name]) for row in comparison_rows])
        errors.append(float(np.max(np.abs(reference_values - comparison_values))))
    return max(errors)


def _run_measured_segment(
    setup_path: Path,
    source_model: Path,
    ik_path: Path,
    output_dir: Path,
    trial_name: str,
    times: np.ndarray,
    grf: np.ndarray,
    cop: np.ndarray,
    free_moment: np.ndarray,
    segment_index: int,
    cutoff_override_hz: float | None = None,
    forces_to_exclude: str | None = None,
) -> tuple[Path, float, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = _prepare_model_for_inverse_dynamics(source_model, output_dir, trial_name)
    geometry = source_model.parent / "Geometry"
    _ensure_local_geometry(geometry, output_dir / "Geometry")
    ik_path = _ensure_local_coordinates(ik_path, output_dir)
    grf_mot, external_xml = _write_measured_load_files(
        output_dir, trial_name, times, grf, cop, free_moment
    )
    setup = output_dir / f"Setup_ID_measuredGRF_{trial_name}.xml"
    sto = output_dir / f"{trial_name}_id_measuredGRF.sto"
    cutoff = _patch_original_setup(
        setup_path,
        setup,
        model_path,
        ik_path,
        external_xml,
        sto,
        cutoff_override_hz=cutoff_override_hz,
        forces_to_exclude=forces_to_exclude,
    )
    _run_opensim_tool(setup, working_directory=setup.parent)
    if not sto.is_file():
        raise FileNotFoundError(f"Measured-GRF ID did not create {sto}")
    target_csv = output_dir / "target_segment.csv"
    _read_target_sto(sto, model_path, target_csv, segment_index, times)
    return target_csv, cutoff, model_path


def _filter_ik_mot_explicit_order4(source: Path, destination: Path) -> tuple[Path, np.ndarray]:
    """Apply the encoder's 4th-order, 12 Hz filter to one continuous IK segment."""
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    end_header = next((i for i, line in enumerate(lines) if line.strip().lower() == "endheader"), None)
    if end_header is None:
        raise ValueError(f"IK motion has no endheader: {source}")
    label_row = next((i for i in range(end_header + 1, len(lines)) if lines[i].strip()), None)
    if label_row is None:
        raise ValueError(f"IK motion has no coordinate labels: {source}")
    labels, parsed_rows = parse_mot(source)
    matrix = np.asarray(parsed_rows, dtype=float)
    time_col = labels.index("time")
    times = matrix[:, time_col]
    if len(times) < 2 or np.any(~np.isfinite(matrix)):
        raise ValueError(f"IK segment is too short to infer its sample rate or is non-finite: {source}")
    steps = np.diff(times)
    if np.any(steps <= 0.0):
        raise ValueError(f"IK segment times are not strictly increasing: {source}")
    # Match convert_mot_to_model_csv exactly: infer the rate from the median
    # positive time step, then filter the original samples without regridding.
    sample_rate = 1.0 / float(np.median(steps))
    effective_cutoff = min(H1_FORCE_CUTOFF_HZ, 0.8 * 0.5 * sample_rate)
    for column, name in enumerate(labels):
        if column != time_col:
            matrix[:, column] = _lowpass_butterworth_4th(
                matrix[:, column], sample_rate, effective_cutoff
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for line in lines[: label_row + 1]:
            handle.write(line + "\n")
        for row in matrix:
            handle.write(" ".join(f"{value:.17g}" for value in row) + "\n")
    return destination, times


def _write_coordinate_segment_csv(mot_path: Path, output_path: Path) -> Path:
    labels, rows = parse_mot(mot_path)
    missing = sorted(set(TARGET_COORDINATES).difference(labels))
    if missing:
        raise ValueError(f"Filtered IK motion {mot_path} is missing coordinates {missing}")
    time_index = labels.index("time")
    coordinate_indices = [labels.index(name) for name in TARGET_COORDINATES]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", *TARGET_COORDINATES])
        for row in rows:
            writer.writerow([row[time_index], *(row[index] for index in coordinate_indices)])
    return output_path


def _run_estimated_segment_explicit_order4(
    setup_path: Path,
    source_segment_dir: Path,
    model_path: Path,
    ik_path: Path,
    output_dir: Path,
    trial_name: str,
    times: np.ndarray,
    segment_index: int,
    external_load_override: Path | None = None,
    forces_to_exclude: str | None = None,
) -> tuple[Path, Path | None, Path | None]:
    """Recompute tau-hat on the explicit 4th-order IK with saved estimated GRF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = model_path.parent / "Geometry"
    _ensure_local_geometry(geometry, output_dir / "Geometry")
    ik_path = _ensure_local_coordinates(ik_path, output_dir)
    external_xml = output_dir / f"{trial_name}_external_forces_estimated.xml"
    if external_load_override is None:
        estimated_mot = _find_unique("*_grf_estimated.mot", source_segment_dir, "estimated-GRF MOT")
        estimated_xml_source = _find_unique(
            "*_external_forces_estimated.xml", source_segment_dir, "estimated external-loads XML"
        )
        estimated_mot_copy = output_dir / estimated_mot.name
        if estimated_mot.resolve() != estimated_mot_copy.resolve():
            shutil.copy2(estimated_mot, estimated_mot_copy)
        tree = ET.parse(estimated_xml_source)
        source_xml_for_id = estimated_xml_source
    else:
        estimated_mot = None
        estimated_xml_source = None
        tree = ET.parse(external_load_override)
        datafile_node = tree.getroot().find(".//datafile")
        if datafile_node is None or not datafile_node.text:
            raise ValueError(f"External-loads XML has no datafile: {external_load_override}")
        datafile_path = Path(datafile_node.text)
        source_load = datafile_path if datafile_path.is_absolute() else external_load_override.parent / datafile_path
        local_load = output_dir / source_load.name
        if source_load.resolve() != local_load.resolve():
            shutil.copy2(source_load, local_load)
        source_xml_for_id = external_load_override
    datafile = tree.getroot().find(".//datafile")
    if datafile is None:
        raise ValueError(f"External-loads XML has no datafile: {source_xml_for_id}")
    if external_load_override is None:
        datafile.text = estimated_mot_copy.name
    else:
        datafile.text = Path(datafile.text).name
    tree.write(external_xml, encoding="utf-8", xml_declaration=True)

    setup = output_dir / f"Setup_ID_estimatedGRF_{trial_name}.xml"
    sto = output_dir / f"{trial_name}_id_estimatedGRF_explicit_order4.sto"
    _patch_original_setup(
        setup_path,
        setup,
        model_path,
        ik_path,
        external_xml,
        sto,
        cutoff_override_hz=-1.0,
        forces_to_exclude=forces_to_exclude,
    )
    _run_opensim_tool(setup, working_directory=setup.parent)
    if not sto.is_file():
        raise FileNotFoundError(f"Estimated-GRF ID did not create {sto}")

    aligned_csv = output_dir / "id_aligned_all_dofs.csv"
    _, _, _, missing = export_torque_csv_from_id_sto(sto, model_path, aligned_csv, times)
    missing_targets = sorted(set(TARGET_COORDINATES).intersection(missing))
    if missing_targets:
        raise ValueError(f"Estimated ID output is missing targets {missing_targets}")
    _, rows = _read_frame_csv(aligned_csv)
    estimate_csv = output_dir / "estimated_segment.csv"
    with estimate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "time", "segment", *[f"{name}_tau_hat" for name in TARGET_COORDINATES]])
        for frame, row in enumerate(rows):
            writer.writerow([
                frame,
                row["time"],
                segment_index,
                *(row[f"{name}_tau"] for name in TARGET_COORDINATES),
            ])
    return estimate_csv, estimated_mot, estimated_xml_source


def _read_frame_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _load_h1_mask(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields, rows = _read_frame_csv(path)
    required = {"time", "valid"}
    if not required.issubset(fields):
        raise ValueError(f"H1 labels are missing {sorted(required - set(fields))}: {path}")
    times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    valid = np.asarray([int(float(row["valid"])) != 0 for row in rows], dtype=bool)
    if "segment" in fields:
        segments = np.asarray([int(float(row["segment"])) for row in rows], dtype=int)
    else:
        segments = np.zeros(len(rows), dtype=int)
    return times, segments, valid


def _segment_number(path: Path) -> int:
    import re

    match = re.search(r"segment_(\d+)", str(path))
    return int(match.group(1)) if match else -1


def _discover_segment_setup_paths(artifact: Path, segment_count: int) -> list[Path]:
    """Find saved estimated-GRF ID setups for segmented and single-segment trials."""
    id_root = artifact / "ID_estimatedGRF"
    segment_setups = sorted(
        id_root.glob("segments/segment_*/Setup_ID_estimatedGRF*.xml"),
        key=_segment_number,
    )
    if segment_setups or segment_count != 1:
        return segment_setups
    # The pipeline writes a sole segment directly in ID_estimatedGRF; only
    # multi-segment trials get the segments/segment_N subdirectory layout.
    return sorted(id_root.glob("Setup_ID_estimatedGRF*.xml"))


def _write_execution_manifest(path: Path, records: list[dict[str, str | int | float]]) -> None:
    fields = [
        "subject", "trial", "protocol", "status", "frames", "cutoff_hz",
        "moment_source", "output_csv", "metadata_json", "output_sha256",
        "metadata_sha256", "source_hashes_json", "source_details_json", "error",
    ]
    temp_path = path.with_suffix(path.suffix + ".partial")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temp_path.replace(path)


def _run_target_trial(
    record: dict[str, str],
    *,
    dataset_root: Path,
    pipeline_root: Path,
    label_root: Path,
    index_path: Path,
    output_root: Path,
    protocol: str,
    verify_identical_loads: bool = False,
    forces_to_exclude: str | None = None,
    cop_shift_m: tuple[float, float] | None = None,
    nomuscles_root: Path | None = None,
) -> dict[str, str | int | float]:
    subject, trial = record["subject"], record["trial"]
    label_relative = record.get("file", "").strip()
    if not label_relative:
        raise ValueError("Included H1 record has no label file path")
    label_path = label_root / label_relative
    if not label_path.is_file():
        raise FileNotFoundError(f"H1 label file not found: {label_path}")
    label_times, label_segments, valid = _load_h1_mask(label_path)

    final_csv = _find_unique(
        f"{subject}_*_{trial}_stageii.csv", pipeline_root / subject, "pipeline trial CSV"
    )
    final_fields, final_rows = _read_frame_csv(final_csv)
    required_fields = {"time", *TARGET_COORDINATES, *(f"{name}_tau" for name in TARGET_COORDINATES)}
    if not required_fields.issubset(final_fields):
        raise ValueError(f"Pipeline CSV is missing required q/tau columns: {final_csv}")
    pipeline_times = np.asarray([float(row["time"]) for row in final_rows], dtype=float)
    if pipeline_times.size != label_times.size or not np.allclose(
        pipeline_times, label_times, rtol=0.0, atol=1.1e-5
    ):
        raise ValueError(
            f"Pipeline timestamps do not match the H1 label rows for {subject}/{trial}"
        )
    if len(valid) != len(final_rows):
        raise ValueError(f"H1 mask length differs from the pipeline CSV for {subject}/{trial}")

    nomuscles_rows: list[dict[str, str]] | None = None
    nomuscles_targets_path: Path | None = None
    if nomuscles_root is not None:
        nomuscles_targets_path = nomuscles_root / subject / trial / "torque_targets.csv"
        nomuscles_fields, nomuscles_rows = _read_frame_csv(nomuscles_targets_path)
        required_nomuscles = {
            "frame", "time", "segment", "valid", "cop_invalid",
            *[f"{name}_tau_hat" for name in TARGET_COORDINATES],
        }
        if not required_nomuscles.issubset(nomuscles_fields):
            raise ValueError(f"Nomuscles target table is missing required columns: {nomuscles_targets_path}")
        if len(nomuscles_rows) != len(final_rows):
            raise ValueError(f"Nomuscles/pipeline frame counts differ for {subject}/{trial}")
        for index, baseline in enumerate(nomuscles_rows):
            if int(float(baseline["frame"])) != index:
                raise ValueError(f"Nomuscles frame index differs at {subject}/{trial}/{index}")
            if int(float(baseline["segment"])) != int(label_segments[index]):
                raise ValueError(f"Nomuscles segment differs at {subject}/{trial}/{index}")
            if abs(float(baseline["time"]) - pipeline_times[index]) > 1.1e-5:
                raise ValueError(f"Nomuscles time differs at {subject}/{trial}/{index}")
            expected_valid = int(bool(valid[index]) and not int(float(baseline["cop_invalid"])))
            if int(float(baseline["valid"])) != expected_valid:
                raise ValueError(f"Nomuscles valid/cop_invalid disagree at {subject}/{trial}/{index}")
    if cop_shift_m is not None and nomuscles_rows is None:
        raise ValueError("Applying per-trial CoP shifts requires --nomuscles-root to preserve tau-hat and cop_invalid")

    force_matches = sorted((dataset_root / "force" / subject).glob(f"{subject}_*_{trial}.npy"))
    if len(force_matches) != 1:
        raise ValueError(
            f"Expected one measured force file for {subject}/{trial}, found {len(force_matches)}"
        )
    force_path = force_matches[0]

    artifact_candidates = list(
        (pipeline_root / subject).glob(f"{final_csv.stem}_*/results")
    )
    if len(artifact_candidates) != 1:
        if not artifact_candidates:
            raise FileNotFoundError(f"AddBiomechanics results not found for {subject}/{trial}")
        raise ValueError(
            f"Ambiguous AddBiomechanics results for {subject}/{trial}: {artifact_candidates}"
        )
    artifact = artifact_candidates[0]
    if artifact is None:
        raise FileNotFoundError(f"AddBiomechanics results not found for {subject}/{trial}")
    model_path = artifact / "Models" / "match_markers_but_ignore_physics.osim"
    ik_path = artifact / "IK" / f"{artifact.parent.name}_ik.mot"
    if not model_path.is_file() or not ik_path.is_file():
        raise FileNotFoundError(f"Persistent BSM model or IK motion missing for {subject}/{trial}")
    segment_paths = discover_segment_mot_paths(ik_path)
    segment_setup_paths = _discover_segment_setup_paths(artifact, len(segment_paths))
    if len(segment_setup_paths) != len(segment_paths):
        raise ValueError(
            f"Expected one original ID setup per IK segment for {subject}/{trial}; "
            f"found {len(segment_setup_paths)} setups and {len(segment_paths)} IK segments"
        )

    cop_source, grf_source, free_moment_source, moment_source, short_cop_runs = (
        _load_measured_wrenches(force_path)
    )
    temp_root = output_root / subject / trial / "segments"
    temp_root.mkdir(parents=True, exist_ok=True)
    segment_target_csvs: list[Path] = []
    coordinate_segment_csvs: list[Path] = []
    segment_hat_csvs: list[Path] = []
    estimated_sources: list[tuple[Path, Path]] = []
    cop_invalid_by_segment: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cutoffs: list[float] = []
    identity_max_abs_error: float | None = None
    for segment_index, (segment_path, setup_path) in enumerate(
        zip(segment_paths, segment_setup_paths)
    ):
        labels, rows = parse_mot(segment_path)
        time_index = labels.index("time")
        segment_times = np.asarray([row[time_index] for row in rows], dtype=float)
        segment_output = temp_root / f"segment_{segment_index}"
        if protocol == "explicit_order4":
            coordinates_path, explicit_times = _filter_ik_mot_explicit_order4(
                segment_path, segment_output / "coordinates_explicit_order4.mot"
            )
            if not np.array_equal(explicit_times, segment_times):
                raise ValueError(f"Explicitly filtered coordinates changed IK timestamps: {segment_path}")
            coordinate_segment_csvs.append(
                _write_coordinate_segment_csv(
                    coordinates_path, segment_output / "encoder_coordinates_segment.csv"
                )
            )
        elif protocol == "original_id_setup":
            coordinates_path = segment_path
        else:
            raise ValueError(f"Unknown extraction protocol: {protocol}")
        cop, grf, free_moment = _resample_wrenches(
            cop_source, grf_source, free_moment_source, segment_times,
            cop_shift_m=cop_shift_m,
        )
        invalid_cop = (grf[:, :, 2] >= FORCE_THRESHOLD_N) & np.all(cop[:, :, :2] == 0.0, axis=2)
        cop_invalid_by_segment[segment_index] = (segment_times, np.any(invalid_cop, axis=1))
        target_csv, cutoff, prepared_model = _run_measured_segment(
            setup_path,
            model_path,
            coordinates_path,
            segment_output,
            f"{subject}_{trial}_segment_{segment_index}",
            segment_times,
            grf,
            cop,
            free_moment,
            segment_index,
            cutoff_override_hz=-1.0 if protocol == "explicit_order4" else None,
            forces_to_exclude=forces_to_exclude,
        )
        segment_target_csvs.append(target_csv)
        cutoffs.append(cutoff)
        if protocol == "explicit_order4":
            estimated_csv, estimated_mot, estimated_xml = _run_estimated_segment_explicit_order4(
                setup_path,
                setup_path.parent,
                prepared_model,
                coordinates_path,
                segment_output / "estimated_id",
                f"{subject}_{trial}_segment_{segment_index}",
                segment_times,
                segment_index,
                forces_to_exclude=forces_to_exclude,
            )
            segment_hat_csvs.append(estimated_csv)
            if estimated_mot is not None and estimated_xml is not None:
                estimated_sources.append((estimated_mot, estimated_xml))
            if verify_identical_loads:
                measured_xml = segment_output / f"{subject}_{trial}_segment_{segment_index}_external_forces_measured.xml"
                identity_csv, _, _ = _run_estimated_segment_explicit_order4(
                    setup_path,
                    setup_path.parent,
                    prepared_model,
                    coordinates_path,
                    segment_output / "identity_check",
                    f"{subject}_{trial}_segment_{segment_index}_identity",
                    segment_times,
                    segment_index,
                    external_load_override=measured_xml,
                    forces_to_exclude=forces_to_exclude,
                )
                target_fields, target_values = _read_frame_csv(target_csv)
                identity_fields, identity_values = _read_frame_csv(identity_csv)
                if len(target_values) != len(identity_values):
                    raise ValueError(f"Identical-load ID row counts differ for {subject}/{trial}")
                check_errors = []
                for name in TARGET_COORDINATES:
                    target_name = f"{name}_tau_target"
                    identity_name = f"{name}_tau_hat"
                    if target_name not in target_fields or identity_name not in identity_fields:
                        raise ValueError(f"Identical-load ID is missing {name} for {subject}/{trial}")
                    target_array = np.asarray([float(row[target_name]) for row in target_values])
                    identity_array = np.asarray([float(row[identity_name]) for row in identity_values])
                    check_errors.append(float(np.max(np.abs(target_array - identity_array))))
                segment_identity_error = max(check_errors)
                identity_max_abs_error = max(
                    identity_max_abs_error or 0.0, segment_identity_error
                )
                if segment_identity_error > 1e-8:
                    raise ValueError(
                        f"Same-load ID moments differ by {segment_identity_error:.6g} N·m "
                        f"for {subject}/{trial}/segment_{segment_index}"
                    )
        elif protocol == "original_id_setup" and verify_identical_loads:
            identity_csv, identity_cutoff, _ = _run_measured_segment(
                setup_path,
                model_path,
                coordinates_path,
                segment_output / "identity_check",
                f"{subject}_{trial}_segment_{segment_index}_identity",
                segment_times,
                grf,
                cop,
                free_moment,
                segment_index,
                forces_to_exclude=forces_to_exclude,
            )
            if identity_cutoff != cutoff:
                raise ValueError(
                    f"Identical-load ID cutoff changed from {cutoff} to {identity_cutoff} "
                    f"for {subject}/{trial}/segment_{segment_index}"
                )
            segment_identity_error = _max_abs_id_csv_difference(
                target_csv,
                identity_csv,
                "target",
                "target",
                subject,
                trial,
                segment_index,
            )
            identity_max_abs_error = max(
                identity_max_abs_error or 0.0, segment_identity_error
            )
            if segment_identity_error > 1e-8:
                raise ValueError(
                    f"Same-load ID moments differ by {segment_identity_error:.6g} N·m "
                    f"for {subject}/{trial}/segment_{segment_index}"
                )
    if any(cutoff != cutoffs[0] for cutoff in cutoffs[1:]):
        raise ValueError(f"Original ID coordinate cutoffs differ across segments: {cutoffs}")

    merged_targets = output_root / subject / trial / "target_moments_by_segment.csv"
    merge_csv_segments(segment_target_csvs, merged_targets)
    target_fields, target_rows = _read_frame_csv(merged_targets)
    target_times = np.asarray([float(row["time"]) for row in target_rows], dtype=float)
    if target_times.shape != pipeline_times.shape or not np.allclose(
        target_times, pipeline_times, rtol=0.0, atol=1.1e-5
    ):
        raise ValueError(
            f"Measured ID timestamps/segments differ from pipeline CSV for {subject}/{trial}"
        )
    estimated_rows: list[dict[str, str]] | None = None
    if protocol == "explicit_order4":
        merged_coordinates = output_root / subject / trial / "encoder_coordinates_by_segment.csv"
        merge_csv_segments(coordinate_segment_csvs, merged_coordinates)
        coordinate_fields, coordinate_rows = _read_frame_csv(merged_coordinates)
        coordinate_times = np.asarray([float(row["time"]) for row in coordinate_rows])
        if coordinate_times.shape != pipeline_times.shape or not np.allclose(
            coordinate_times, pipeline_times, rtol=0.0, atol=1.1e-5
        ):
            raise ValueError(f"Filtered q timestamps/segments differ from encoder CSV for {subject}/{trial}")
        missing_coordinates = sorted(set(TARGET_COORDINATES).difference(coordinate_fields))
        if missing_coordinates:
            raise ValueError(f"Filtered q is missing encoder coordinates {missing_coordinates}")
        q_max_abs_error = max(
            float(np.max(np.abs(
                np.asarray([float(row[name]) for row in coordinate_rows])
                - np.asarray([float(row[name]) for row in final_rows])
            )))
            for name in TARGET_COORDINATES
        )
        if q_max_abs_error > 5e-12:
            raise ValueError(
                f"Filtered q differs from encoder CSV by {q_max_abs_error:.6g} rad "
                f"for {subject}/{trial}"
            )
        merged_estimates = output_root / subject / trial / "estimated_moments_by_segment.csv"
        merge_csv_segments(segment_hat_csvs, merged_estimates)
        estimated_fields, estimated_rows = _read_frame_csv(merged_estimates)
        estimated_times = np.asarray([float(row["time"]) for row in estimated_rows], dtype=float)
        expected_hat_fields = {f"{name}_tau_hat" for name in TARGET_COORDINATES}
        if not expected_hat_fields.issubset(estimated_fields):
            raise ValueError(f"Explicit ID output is missing estimated target columns for {subject}/{trial}")
        if estimated_times.shape != pipeline_times.shape or not np.allclose(
            estimated_times, pipeline_times, rtol=0.0, atol=1.1e-5
        ):
            raise ValueError(f"Recomputed tau-hat timestamps differ from pipeline for {subject}/{trial}")
    coordinate_names = [name.removesuffix("_tau_target") for name in target_fields if name.endswith("_tau_target")]
    if tuple(coordinate_names) != TARGET_COORDINATES:
        raise ValueError(f"Target ID coordinates are {coordinate_names}, expected {TARGET_COORDINATES}")
    for row in target_rows:
        row_segments = int(float(row["segment"]))
        index = int(float(row["frame"]))
        if row_segments != label_segments[index]:
            raise ValueError(f"H1 segment labels mismatch after merging {subject}/{trial}")

    trial_output = output_root / subject / trial / "torque_targets.csv"
    trial_output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = trial_output.with_suffix(".csv.partial")
    output_fields = ["frame", "time", "segment", "valid", "cop_invalid"]
    output_fields += [f"{name}_tau_target" for name in TARGET_COORDINATES]
    output_fields += [f"{name}_tau_hat" for name in TARGET_COORDINATES]
    new_cop_invalid_frames = 0
    valid_output_frames = 0
    cop_invalid_output_frames = 0
    with partial_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for index, (target_row, pipeline_row) in enumerate(zip(target_rows, final_rows)):
            segment_index = int(label_segments[index])
            invalid_times, invalid_frames = cop_invalid_by_segment[segment_index]
            nearest = int(np.argmin(np.abs(invalid_times - float(pipeline_row["time"]))))
            if abs(float(invalid_times[nearest]) - float(pipeline_row["time"])) > 1.1e-5:
                raise ValueError(f"Measured CoP validity times do not align at {subject}/{trial}/{index}")
            computed_invalid = bool(invalid_frames[nearest])
            if nomuscles_rows is not None:
                baseline = nomuscles_rows[index]
                baseline_invalid = bool(int(float(baseline["cop_invalid"])))
                if baseline_invalid and not computed_invalid:
                    raise ValueError(f"An existing cop_invalid frame became valid at {subject}/{trial}/{index}")
                output_invalid = baseline_invalid or computed_invalid
                output_valid = int(bool(int(float(baseline["valid"]))) and not computed_invalid)
            else:
                output_invalid = computed_invalid
                output_valid = int(bool(valid[index]) and not computed_invalid)
            new_cop_invalid_frames += int(computed_invalid and not (
                nomuscles_rows is not None and int(float(nomuscles_rows[index]["cop_invalid"]))
            ))
            valid_output_frames += output_valid
            cop_invalid_output_frames += int(output_invalid)
            output_row: dict[str, str | int | float] = {
                "frame": nomuscles_rows[index]["frame"] if nomuscles_rows is not None else index,
                "time": nomuscles_rows[index]["time"] if nomuscles_rows is not None else float(pipeline_row["time"]),
                "segment": nomuscles_rows[index]["segment"] if nomuscles_rows is not None else segment_index,
                "valid": output_valid,
                "cop_invalid": int(output_invalid),
            }
            for name in TARGET_COORDINATES:
                target_value = float(target_row[f"{name}_tau_target"])
                if not math.isfinite(target_value):
                    raise ValueError(f"Non-finite measured target {name} at frame {index}")
                output_row[f"{name}_tau_target"] = target_value
                if nomuscles_rows is not None:
                    hat_value = float(nomuscles_rows[index][f"{name}_tau_hat"])
                elif estimated_rows is None:
                    hat_value = float(pipeline_row[f"{name}_tau"])
                else:
                    hat_value = float(estimated_rows[index][f"{name}_tau_hat"])
                if not math.isfinite(hat_value):
                    raise ValueError(f"Non-finite estimated moment {name} at frame {index}")
                output_row[f"{name}_tau_hat"] = hat_value
            writer.writerow(output_row)
    partial_output.replace(trial_output)

    source_hashes = {
        "h1_index_sha256": _sha256(index_path),
        "h1_labels_sha256": _sha256(label_path),
        "force_sha256": _sha256(force_path),
        "pipeline_csv_sha256": _sha256(final_csv),
        "model_sha256": _sha256(model_path),
        "ik_sha256": _sha256(ik_path),
        "setup_sha256": [_sha256(path) for path in segment_setup_paths],
    }
    if nomuscles_targets_path is not None:
        source_hashes["nomuscles_torque_targets_sha256"] = _sha256(nomuscles_targets_path)
    source_paths = {
        "h1_index": str(index_path),
        "h1_labels": str(label_path),
        "measured_force": str(force_path),
        "pipeline_csv": str(final_csv),
        "model": str(model_path),
        "ik": str(ik_path),
        "original_id_setups": [str(path) for path in segment_setup_paths],
    }
    if nomuscles_targets_path is not None:
        source_paths["nomuscles_torque_targets"] = str(nomuscles_targets_path)
    if estimated_sources:
        source_hashes["estimated_force_sha256"] = [
            _sha256(estimated_mot) for estimated_mot, _ in estimated_sources
        ]
        source_hashes["estimated_external_xml_sha256"] = [
            _sha256(xml_path) for _, xml_path in estimated_sources
        ]
        source_paths["estimated_force"] = [str(path) for path, _ in estimated_sources]
        source_paths["estimated_external_xml"] = [str(path) for _, path in estimated_sources]
    source_details = {
        "h1_index": {"path": source_paths["h1_index"], "sha256": source_hashes["h1_index_sha256"]},
        "h1_labels": {"path": source_paths["h1_labels"], "sha256": source_hashes["h1_labels_sha256"]},
        "measured_force": {"path": source_paths["measured_force"], "sha256": source_hashes["force_sha256"]},
        "pipeline_csv": {"path": source_paths["pipeline_csv"], "sha256": source_hashes["pipeline_csv_sha256"]},
        "model": {"path": source_paths["model"], "sha256": source_hashes["model_sha256"]},
        "ik": {"path": source_paths["ik"], "sha256": source_hashes["ik_sha256"]},
        "original_id_setups": [
            {"path": path, "sha256": digest}
            for path, digest in zip(source_paths["original_id_setups"], source_hashes["setup_sha256"])
        ],
    }
    if nomuscles_targets_path is not None:
        source_details["nomuscles_torque_targets"] = {
            "path": str(nomuscles_targets_path),
            "sha256": source_hashes["nomuscles_torque_targets_sha256"],
        }
    if estimated_sources:
        source_details["estimated_force"] = [
            {"path": path, "sha256": digest}
            for path, digest in zip(source_paths["estimated_force"], source_hashes["estimated_force_sha256"])
        ]
        source_details["estimated_external_xml"] = [
            {"path": path, "sha256": digest}
            for path, digest in zip(source_paths["estimated_external_xml"], source_hashes["estimated_external_xml_sha256"])
        ]
    metadata = {
        "subject": subject,
        "trial": trial,
        "protocol": "original_id_setup",
        "forces_to_exclude": forces_to_exclude or "",
        "frames": len(final_rows),
        "coordinate_cutoff_hz": cutoffs[0],
        "effective_coordinate_cutoff_hz": -1.0 if protocol == "explicit_order4" else cutoffs[0],
        "explicit_kinematic_filter": "4th-order zero-phase Butterworth, 12 Hz" if protocol == "explicit_order4" else None,
        "kinematics_vs_encoder_max_abs_error_rad": q_max_abs_error if protocol == "explicit_order4" else None,
        "identical_loads_max_abs_error_nm": identity_max_abs_error,
        "measured_force_filter": "zero-phase 4th-order Butterworth, 12 Hz, 250 Hz",
        "estimated_force_filter_provenance": (
            "Saved pipeline tau-hat is retained for original_id_setup. Its estimated GRFs "
            "come from the pipeline ID/contact estimator with the saved coordinate cutoff; "
            "the saved external-force MOT is not post-filtered by this extractor."
        ),
        "cop_filter_scope": "separate continuous contact runs; runs shorter than 3 samples preserved",
        "force_threshold_n": FORCE_THRESHOLD_N,
        "force_alignment": "linear interpolation to existing segment IK times after filtering; no extrapolation",
        "load_application": "calcn_l/calcn_r; force, point and moment expressed in ground",
        "free_moment_source": moment_source,
        "short_cop_contact_runs_preserved": short_cop_runs,
        "valid_frames": valid_output_frames,
        "cop_invalid_frames": cop_invalid_output_frames,
        "new_cop_invalid_frames": new_cop_invalid_frames,
        "tau_hat_source": "nomuscles torque_targets.csv, copied unchanged" if nomuscles_rows is not None else "pipeline CSV",
        "target_id_regenerated": True,
        "output_csv": str(trial_output),
        "geometry_source": str(model_path.parent / "Geometry"),
        "source_paths": source_paths,
        "source_details": source_details,
        "source_hashes": source_hashes,
    }
    if cop_shift_m is not None:
        metadata["cop_shift_m"] = [float(cop_shift_m[0]), float(cop_shift_m[1])]
        metadata["cop_shift_method"] = "B2_median_CoP_minus_midfoot_Fz_gt_100N"
    metadata["protocol"] = protocol
    artifact_root = trial_output.parent
    metadata["artifact_files"] = sorted(
        str(path.relative_to(artifact_root))
        for path in artifact_root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "Geometry" not in path.relative_to(artifact_root).parts[:-1]
    )
    metadata["geometry_directories"] = sorted(
        str(path.relative_to(artifact_root))
        for path in artifact_root.rglob("Geometry")
        if path.is_dir() and not path.is_symlink()
    )
    metadata_path = trial_output.with_name("run_metadata.json")
    metadata_partial = metadata_path.with_suffix(".json.partial")
    metadata_partial.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata_partial.replace(metadata_path)
    metadata_hash = _sha256(metadata_path)
    output_hash = _sha256(trial_output)
    return {
        "subject": subject,
        "trial": trial,
        "protocol": protocol,
        "status": "ok",
        "frames": len(final_rows),
        "cutoff_hz": cutoffs[0],
        "moment_source": moment_source,
        "output_csv": str(trial_output),
        "metadata_json": str(metadata_path),
        "output_sha256": output_hash,
        "metadata_sha256": metadata_hash,
        "source_hashes_json": json.dumps(source_hashes, sort_keys=True, separators=(",", ":")),
        "source_details_json": json.dumps(source_details, sort_keys=True, separators=(",", ":")),
        "error": "",
    }


def _resume_output_is_current(
    metadata: dict[str, object],
    fields: set[str],
    row_count: int,
    *,
    expected_frames: int,
    protocol: str,
    expected_shift: list[float] | None,
    nomuscles_targets_path: Path | None = None,
) -> bool:
    expected_outputs = {
        "time", "segment", "valid", "cop_invalid",
        *[f"{name}_tau_target" for name in TARGET_COORDINATES],
        *[f"{name}_tau_hat" for name in TARGET_COORDINATES],
    }
    if (
        metadata.get("protocol") != protocol
        or metadata.get("cop_shift_m") != expected_shift
        or row_count != expected_frames
        or not expected_outputs.issubset(fields)
    ):
        return False
    if nomuscles_targets_path is not None:
        source_hashes = metadata.get("source_hashes")
        if not isinstance(source_hashes, dict):
            return False
        return (
            metadata.get("target_id_regenerated") is True
            and metadata.get("tau_hat_source") == "nomuscles torque_targets.csv, copied unchanged"
            and source_hashes.get("nomuscles_torque_targets_sha256") == _sha256(nomuscles_targets_path)
        )
    return True


def extract_groundlink_targets(
    *,
    dataset_root: Path,
    pipeline_root: Path,
    label_root: Path,
    output_root: Path,
    index_path: Path | None = None,
    limit: int | None = None,
    protocol: str = "original_id_setup",
    subject: str | None = None,
    trial: str | None = None,
    resume: bool = False,
    verify_identical_loads: bool = False,
    forces_to_exclude: str | None = None,
    cop_shifts: dict[tuple[str, str], tuple[float, float]] | None = None,
    nomuscles_root: Path | None = None,
) -> tuple[list[dict[str, str | int | float]], int]:
    index_path = index_path or (label_root / "index.csv")
    with index_path.open(newline="", encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    included = [row for row in index if row.get("included") == "1"]
    if len(included) != 206:
        raise ValueError(f"H1 index must select exactly 206 trials; found {len(included)}")
    expected_subjects = {"s001": 33, "s002": 33, "s003": 36, "s004": 27, "s005": 35, "s006": 42}
    counts = {subject: sum(row["subject"] == subject for row in included) for subject in expected_subjects}
    if counts != expected_subjects:
        raise ValueError(f"Unexpected included-trial distribution: {counts}")
    if subject is not None:
        if subject not in expected_subjects:
            raise ValueError(f"--subject must be one of {sorted(expected_subjects)}, got {subject!r}")
        included = [row for row in included if row["subject"] == subject]
    if trial is not None:
        if subject is None:
            raise ValueError("--trial requires --subject")
        included = [row for row in included if row["trial"] == trial]
        if len(included) != 1:
            raise ValueError(f"Expected one included trial for {subject}/{trial}, found {len(included)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        included = included[:limit]
    if cop_shifts is not None:
        if nomuscles_root is None:
            raise ValueError("--cop-shifts-csv requires --nomuscles-root to preserve tau-hat and cop_invalid")
        missing_shifts = [
            f"{row['subject']}/{row['trial']}"
            for row in included
            if (row["subject"], row["trial"]) not in cop_shifts
        ]
        if missing_shifts:
            raise ValueError(
                "CoP shift CSV has no shift for selected trials: "
                + ", ".join(missing_shifts[:10])
            )
    if protocol not in {"original_id_setup", "explicit_order4"}:
        raise ValueError(f"Unknown extraction protocol: {protocol}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        output_root / f"execution_manifest_{subject}.csv"
        if subject is not None else output_root / "execution_manifest.csv"
    )
    completed: dict[tuple[str, str, str], dict[str, str]] = {}
    if resume:
        prior_manifests = [output_root / "execution_manifest.csv", manifest_path]
        for prior_path in dict.fromkeys(prior_manifests):
            if not prior_path.is_file():
                continue
            with prior_path.open(newline="", encoding="utf-8") as handle:
                for prior in csv.DictReader(handle):
                    key = (prior.get("subject", ""), prior.get("trial", ""), prior.get("protocol", ""))
                    if prior.get("status") != "ok" or key[2] != protocol:
                        continue
                    completed[key] = prior
    results: list[dict[str, str | int | float]] = []
    failures = 0
    for row in included:
        key = (row["subject"], row["trial"], protocol)
        previous = completed.get(key)
        output_csv = output_root / row["subject"] / row["trial"] / "torque_targets.csv"
        metadata_path = output_csv.with_name("run_metadata.json")
        if resume and previous is not None and output_csv.is_file() and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                expected_shift = list(cop_shifts[key]) if cop_shifts is not None else None
                fields, rows = _read_frame_csv(output_csv)
                baseline_targets = (
                    nomuscles_root / row["subject"] / row["trial"] / "torque_targets.csv"
                    if nomuscles_root is not None else None
                )
                if _resume_output_is_current(
                    metadata,
                    set(fields),
                    len(rows),
                    expected_frames=int(previous.get("frames", 0)),
                    protocol=protocol,
                    expected_shift=expected_shift,
                    nomuscles_targets_path=baseline_targets,
                ):
                    results.append(previous)
                    _write_execution_manifest(manifest_path, results)
                    print(
                        f"[{len(results)}/{len(included)}] {row['subject']}/{row['trial']}: "
                        "resumed existing success",
                        flush=True,
                    )
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        try:
            result = _run_target_trial(
                row,
                dataset_root=dataset_root,
                pipeline_root=pipeline_root,
                label_root=label_root,
                index_path=index_path,
                output_root=output_root,
                protocol=protocol,
                verify_identical_loads=verify_identical_loads,
                forces_to_exclude=forces_to_exclude,
                cop_shift_m=cop_shifts.get(key) if cop_shifts is not None else None,
                nomuscles_root=nomuscles_root,
            )
        except Exception as exc:
            failures += 1
            result = {
                "subject": row["subject"],
                "trial": row["trial"],
                "protocol": protocol,
                "status": "failed",
                "frames": 0,
                "cutoff_hz": "",
                "moment_source": "",
                "output_csv": "",
                "metadata_json": "",
                "output_sha256": "",
                "metadata_sha256": "",
                "source_hashes_json": "",
                "source_details_json": "",
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }
        results.append(result)
        _write_execution_manifest(manifest_path, results)
        print(
            f"[{len(results)}/{len(included)}] {row['subject']}/{row['trial']}: "
            f"{result['status']} {result.get('error', '')}",
            flush=True,
        )
    return results, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract six GroundLink torque targets with each trial's saved ID setup."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--index", type=Path, help="H1 trial index CSV; label paths are relative to its directory.")
    labels.add_argument("--label-root", type=Path, help="Directory containing index.csv and the H1 labels.")
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("original_id_setup", "explicit_order4"),
        default="original_id_setup",
        help=(
            "Use the saved ID cutoff and pipeline tau-hat, or explicitly filter q and "
            "recalculate both ID passes."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N included trials for a scheduler smoke test.",
    )
    parser.add_argument(
        "--subject", choices=("s001", "s002", "s003", "s004", "s005", "s006"),
        help="Process one subject; writes an isolated execution_manifest_<subject>.csv.",
    )
    parser.add_argument("--trial", help="Limit a subject run to one included trial (for integration checks).")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse prior successful trials with an existing output CSV and matching metadata.",
    )
    parser.add_argument(
        "--verify-identical-loads", action="store_true",
        help="On this run, repeat the estimated-ID path with measured loads and assert numerical identity.",
    )
    parser.add_argument(
        "--forces-to-exclude", choices=("Actuators", "Muscles"), default=None,
        help="Set InverseDynamicsTool forces_to_exclude for every ID pass.",
    )
    parser.add_argument(
        "--cop-shifts-csv", type=Path, default=None,
        help="Optional per-trial CoP translations in metres; CSV columns: subject,trial,dx_m,dy_m.",
    )
    parser.add_argument(
        "--nomuscles-root", type=Path, default=None,
        help="Existing _nomuscles targets used to preserve tau-hat and cop_invalid during CoP correction.",
    )
    parser.add_argument(
        "--cop-shift-source", choices=("none", "tcode"), default="none",
        help="For Phase C, recompute the pipeline's constant T_code and rerun measured ID only.",
    )
    parser.add_argument(
        "--tcode-reference-csv", type=Path, default=None,
        help="B1-bis code_transform_by_trial.csv used only as an independent numerical reference.",
    )
    parser.add_argument(
        "--smplx-model-dir", type=Path, default=None,
        help="Directory containing the subject-specific SMPLX_FEMALE/MALE.npz models.",
    )
    parser.add_argument(
        "--marker-yaml", type=Path, default=ROOT / "assets" / "smpl2ab" / "bsm_markers_smplx.yaml",
        help="BSM SMPL-X marker vertex map used by the pipeline.",
    )
    parser.add_argument(
        "--tcode-preflight-only", action="store_true",
        help="Recompute and verify T_code for all 206 trials without creating correction outputs.",
    )
    parser.add_argument(
        "--tcode-preflight-output", type=Path, default=None,
        help="CSV path for the read-only full-cohort T_code audit.",
    )
    args = parser.parse_args()
    if args.tcode_preflight_only and args.cop_shift_source != "tcode":
        parser.error("--tcode-preflight-only requires --cop-shift-source tcode")
    index_path = args.index.expanduser().resolve() if args.index else None
    label_root = index_path.parent if index_path else args.label_root.expanduser().resolve()
    index_path = index_path or (label_root / "index.csv")
    output_dir = args.output_dir.expanduser().resolve()
    try:
        if args.cop_shift_source == "tcode":
            if args.cop_shifts_csv is not None:
                parser.error("--cop-shifts-csv cannot be combined with --cop-shift-source tcode")
            if args.nomuscles_root is None or args.tcode_reference_csv is None or args.smplx_model_dir is None:
                parser.error(
                    "--nomuscles-root, --tcode-reference-csv, and --smplx-model-dir are required "
                    "with --cop-shift-source tcode"
                )
            from rerun_groundlink_id_nomuscles import preflight_tcode_cohort, run_tcode_subject

            if args.tcode_preflight_only:
                if args.tcode_preflight_output is None:
                    parser.error("--tcode-preflight-output is required with --tcode-preflight-only")
                if args.subject is not None or args.trial is not None:
                    parser.error("T_code preflight always covers all 206 trials; omit --subject and --trial")
                result = preflight_tcode_cohort(
                    source_root=args.nomuscles_root.expanduser().resolve(),
                    dataset_root=args.dataset_root.expanduser().resolve(),
                    index_path=index_path,
                    tcode_reference_csv=args.tcode_reference_csv.expanduser().resolve(),
                    smplx_model_dir=args.smplx_model_dir.expanduser().resolve(),
                    marker_yaml=args.marker_yaml.expanduser().resolve(),
                    audit_csv=args.tcode_preflight_output.expanduser().resolve(),
                )
                print(json.dumps(result, indent=2))
                return 0
            if args.subject is None:
                parser.error("--subject is required with --cop-shift-source tcode (one subject per HPC job)")

            records, failures = run_tcode_subject(
                source_root=args.nomuscles_root.expanduser().resolve(),
                dataset_root=args.dataset_root.expanduser().resolve(),
                index_path=index_path,
                tcode_reference_csv=args.tcode_reference_csv.expanduser().resolve(),
                smplx_model_dir=args.smplx_model_dir.expanduser().resolve(),
                marker_yaml=args.marker_yaml.expanduser().resolve(),
                output_root=output_dir,
                subject=args.subject,
                trial_filter=args.trial,
                resume=args.resume,
            )
            print(json.dumps({
                "requested": len(records),
                "succeeded": sum(record["status"] == "ok" for record in records),
                "failed": failures,
                "manifest": str(output_dir / f"execution_manifest_{args.subject}.csv"),
                "output_root": str(output_dir),
            }, indent=2))
            return 1 if failures else 0
        records, failures = extract_groundlink_targets(
            dataset_root=args.dataset_root.expanduser().resolve(),
            pipeline_root=args.pipeline_root.expanduser().resolve(),
            label_root=label_root,
            index_path=index_path,
            output_root=output_dir,
            limit=args.limit,
            protocol=args.protocol,
            subject=args.subject,
            trial=args.trial,
            resume=args.resume,
            verify_identical_loads=args.verify_identical_loads,
            forces_to_exclude=args.forces_to_exclude,
            cop_shifts=_load_cop_shifts(
                args.cop_shifts_csv.expanduser().resolve() if args.cop_shifts_csv else None
            ),
            nomuscles_root=args.nomuscles_root.expanduser().resolve() if args.nomuscles_root else None,
        )
    except Exception as exc:
        parser.exit(2, f"Extraction preflight failed: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "requested": len(records),
        "succeeded": sum(record["status"] == "ok" for record in records),
        "failed": failures,
        "manifest": str(
            output_dir
            / (f"execution_manifest_{args.subject}.csv" if args.subject else "execution_manifest.csv")
        ),
        "output_root": str(output_dir),
    }, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""EXO_RECORD trial -> OpenSim IK -> ID with the measured force plate -> unified CSV.

Inputs (Vicon lab frame: Z up, millimetres):
  <subject-root>/aligned/marker/<trial>.csv       100 Hz, <MARKER>_X/_Y/_Z columns
  <subject-root>/aligned/forceplate/<trial>.csv   1000 Hz, Fx..Fz [N], Mx..Mz [N*mm]
                                                  about the plate centre, Cx..Cz [mm];
                                                  F and M are the action on the plate.
  Scaled subject model (Y up): <subject-root>/<subject>_opensim/calibration/model_opensim.osim

Everything written by this script is in the OpenSim frame (Y up, metres).
The single plate measures the total wrench of both feet. The per-foot split is a
lever rule on the foot positions; it preserves the total wrench exactly, so pelvis
residuals do not depend on it, but foot-to-foot sharing (leg torques) is assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Plate is 900 x 900 mm with its corner at the lab origin; moments are reported
# about its centre on the top surface (checked in DATASET-BSM conversion_audit.json).
PLATE_SIZE_M = 0.9
MARKER_ALIASES = {"LASI": "LFWT", "RASI": "RFWT", "LPSI": "LBWT", "RPSI": "RBWT"}
FEET = ("calcn_l", "calcn_r")
LIFT_THRESHOLD_M = 0.03
UNLOADED_FZ_N = 20.0
IK_WARMUP_SECONDS = 1.0
IK_INITIAL_PRIOR_WEIGHT = 0.0001
IK_COORDINATE_PRIOR_WEIGHT = 0.001
# Target the torso coordinates that carried the largest posterior torque error;
# shoulder/elbow neutral targets degraded marker fit on the same validation.
TORSO_NEUTRAL_COORDINATES = (
    "thorax_twist",
    "lumbar_twist",
    "thorax_bending",
    "thorax_extension",
    "lumbar_bending",
    "lumbar_extension",
)
TORSO_NEUTRAL_PRIOR_MULTIPLIER = 10.0
# Thorax and lumbar are redundant for the trunk markers; "lumbar" pins only the
# lumbar split and leaves the total trunk motion to the thorax coordinates.
TORSO_PRIOR_CHOICES = {
    "all": TORSO_NEUTRAL_COORDINATES,
    "lumbar": ("lumbar_twist", "lumbar_bending", "lumbar_extension"),
    "none": (),
}
WEAKLY_OBSERVED_COORDINATES = (
    "thorax_twist",
    "lumbar_twist",
    "scapula_abduction_r",
    "scapula_elevation_r",
    "scapula_upward_rot_r",
    "scapula_abduction_l",
    "scapula_elevation_l",
    "scapula_upward_rot_l",
    "pro_sup_r",
    "pro_sup_l",
    "wrist_flexion_r",
    "wrist_deviation_r",
    "wrist_flexion_l",
    "wrist_deviation_l",
)
WEAK_COORDINATES_TO_SMOOTH = WEAKLY_OBSERVED_COORDINATES + (
    "thorax_bending",
    "thorax_extension",
    "lumbar_bending",
    "lumbar_extension",
    "head_bending",
)


def lab_to_osim(v: np.ndarray) -> np.ndarray:
    """Vicon (x, y, z) Z-up -> OpenSim (x, z, -y) Y-up."""
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def load_markers(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return times [s], marker names, positions (T, N, 3) in lab metres.

    Drops Vicon subject prefixes ("NewSubject:C7" -> "C7").
    """
    data = np.genfromtxt(path, delimiter=",", names=True, deletechars="")
    columns = [c[:-2] for c in data.dtype.names if c.endswith("_X")]
    pos = np.stack([np.stack([data[f"{c}_{a}"] for a in "XYZ"], axis=-1) for c in columns], axis=1)
    return data["timestamp"] / 1000.0, [c.split(":")[-1] for c in columns], pos / 1000.0


def plate_wrench(
    path: Path, times: np.ndarray, cutoff_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lab-frame force on the subject [N], CoP [m] and vertical free moment [N*m].

    F and M are low-pass filtered at the plate rate before CoP and free moment are
    derived, then sampled at the marker times.
    """
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = data["timestamp"] / 1000.0
    rate = 1.0 / float(np.median(np.diff(t)))
    sos = butter(4, cutoff_hz, fs=rate, output="sos")
    f = sosfiltfilt(sos, np.stack([data["Fx"], data["Fy"], data["Fz"]], axis=1), axis=0)
    m = sosfiltfilt(sos, np.stack([data["Mx"], data["My"], data["Mz"]], axis=1), axis=0) / 1000.0
    f, m = (np.stack([np.interp(times, t, x[:, i]) for i in range(3)], axis=1) for x in (f, m))
    loaded = np.abs(f[:, 2]) > UNLOADED_FZ_N
    fz = np.where(loaded, f[:, 2], 1.0)
    # About the centre o on the surface: M = (C - o) x F + Tz * ez.
    rx, ry = -m[:, 1] / fz, m[:, 0] / fz
    tz = m[:, 2] - (rx * f[:, 1] - ry * f[:, 0])
    cop = np.stack([rx, ry, np.zeros_like(rx)], axis=1) + [PLATE_SIZE_M / 2, PLATE_SIZE_M / 2, 0.0]
    force = np.where(loaded[:, None], -f, 0.0)
    return force, cop, np.where(loaded, -tz, 0.0)


def split_feet(
    force: np.ndarray,
    cop: np.ndarray,
    free_moment: np.ndarray,
    foot_l: np.ndarray,
    foot_r: np.ndarray,
    contact_l: np.ndarray,
    contact_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split one plate wrench between two feet (lab frame, Z up).

    Right fraction r = projection of the CoP on the left->right foot-centre segment,
    clipped to [0, 1]; a foot out of contact gets nothing. Each foot keeps its own
    centre shifted by the CoP's off-segment offset, so sum(f_i * p_i) = CoP and the
    total force and moment are preserved exactly.
    Returns force, point, free moment (each T, 2, 3; order left, right) and r (T,).
    """
    # ponytail: lever rule assumes each foot's CoP sits at its centre; a
    # minimum-joint-torque split is the upgrade if leg torques must be sharper.
    d = foot_r[:, :2] - foot_l[:, :2]
    r = np.einsum("ij,ij->i", cop[:, :2] - foot_l[:, :2], d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
    r = np.clip(r, 0.0, 1.0)
    r = np.where(contact_r & ~contact_l, 1.0, np.where(contact_l & ~contact_r, 0.0, r))
    fractions = np.stack([1.0 - r, r], axis=1)[:, :, None]
    centres = np.stack([foot_l, foot_r], axis=1) * [1.0, 1.0, 0.0]
    offset = cop - np.einsum("ik,ikj->ij", fractions[:, :, 0], centres)
    moment = np.zeros_like(force)
    moment[:, 2] = free_moment
    return (
        fractions * force[:, None, :],
        centres + offset[:, None, :],
        fractions * moment[:, None, :],
        r,
    )


def run_ik_parallel(
    model: Path,
    trc: Path,
    marker_names: list[str],
    times: np.ndarray,
    out_mot: Path,
    workers: int,
    warmup_seconds: float = IK_WARMUP_SECONDS,
    coordinate_prior_weight: float = IK_COORDINATE_PRIOR_WEIGHT,
    coordinate_prior_file: Path | None = None,
    neutral_coordinates: tuple[str, ...] = TORSO_NEUTRAL_COORDINATES,
) -> Path:
    """Run overlapping OpenSim IK chunks, trimming each chunk's warm-up frames.

    Each OpenSim process receives only its own TRC slice, so it does not solve all
    preceding frames before reaching its assigned time range. The coordinate prior
    is limited to weakly observed upper-body coordinates.
    """
    import opensim

    if workers < 1:
        raise ValueError("workers must be at least 1")
    if times.size == 0 or not np.all(np.diff(times) > 0):
        raise ValueError("IK times must be non-empty and strictly increasing")
    if warmup_seconds < 0 or coordinate_prior_weight < 0:
        raise ValueError("warm-up and coordinate prior weight must be non-negative")

    model_obj = opensim.Model(str(model))
    coordinate_set = model_obj.getCoordinateSet()
    model_markers = {
        model_obj.getMarkerSet().get(i).getName()
        for i in range(model_obj.getMarkerSet().getSize())
    }
    usable_markers = [name for name in marker_names if name in model_markers]
    if not usable_markers:
        raise ValueError("No trial markers are present in the OpenSim model")

    tool = opensim.InverseKinematicsTool()
    tasks = tool.getIKTaskSet()
    for name in usable_markers:
        task = opensim.IKMarkerTask()
        task.setName(name)
        task.setWeight(1.0)
        task.setApply(True)
        tasks.cloneAndAppend(task)
    if coordinate_prior_weight:
        state = model_obj.initSystem()
        prior_names = WEAK_COORDINATES_TO_SMOOTH if coordinate_prior_file else WEAKLY_OBSERVED_COORDINATES
        for name in prior_names:
            try:
                coordinate = coordinate_set.get(name)
            except Exception:
                continue
            if coordinate.isDependent(state):
                continue
            task = opensim.IKCoordinateTask()
            task.setName(name)
            task.setValue(coordinate.getDefaultValue())
            task_weight = coordinate_prior_weight
            if coordinate_prior_file and name in neutral_coordinates:
                task_weight *= TORSO_NEUTRAL_PRIOR_MULTIPLIER
            task.setWeight(task_weight)
            task.setApply(True)
            if coordinate_prior_file:
                task.setValueType(opensim.IKCoordinateTask.FromFile)
            tasks.cloneAndAppend(task)
    else:
        prior_names = ()
        state = model_obj.initSystem()
    tool.set_model_file(str(model))
    if coordinate_prior_file:
        tool.set_coordinate_file(str(coordinate_prior_file.resolve()))
    tool.setResultsDir(str(out_mot.parent))
    tool.set_report_errors(False)
    half_dt = 0.5 * float(np.median(np.diff(times)))
    chunks = np.array_split(times, min(workers, times.size))
    parts, setups, trc_parts, keep_from_times = [], [], [], []
    for i, chunk in enumerate(chunks):
        part = out_mot.with_name(f"{out_mot.stem}_part{i:02d}.mot")
        start_index = int(np.searchsorted(times, chunk[0] - warmup_seconds, side="left"))
        end_index = int(np.searchsorted(times, chunk[-1], side="right"))
        warmup_start = float(times[start_index] - half_dt)
        trc_part = out_mot.with_name(f"{out_mot.stem}_part{i:02d}.trc")
        write_trc_slice(trc, trc_part, start_index, end_index, times.size)
        tool.set_marker_file(str(trc_part))
        tool.setName(part.stem)
        tool.set_time_range(0, warmup_start)
        tool.set_time_range(1, float(chunk[-1] + half_dt))
        tool.set_output_motion_file(str(part))
        setups.append(part.with_suffix(".xml"))
        trc_parts.append(trc_part)
        tool.printToXML(str(setups[-1]))
        parts.append(part)
        keep_from_times.append(float(chunk[0] - half_dt))
    command = [shutil.which("opensim-cmd") or "opensim-cmd", "-o", "error", "run-tool"]
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(lambda xml: subprocess.run([*command, str(xml)], capture_output=True, text=True), setups):
            if result.returncode != 0:
                raise RuntimeError(f"IK chunk failed:\n{result.stdout}\n{result.stderr}")
    merge_mot_parts(parts, out_mot, times.size, keep_from_times)
    for part, setup, trc_part in zip(parts, setups, trc_parts):
        part.unlink()
        setup.unlink()
        trc_part.unlink()
    return out_mot


def write_trc_slice(
    source: Path,
    output: Path,
    start_index: int,
    end_index: int,
    expected_rows: int,
) -> Path:
    """Write a contiguous TRC frame slice, updating its frame-count metadata."""
    lines = source.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError(f"Invalid TRC header: {source}")
    rows = lines[6:]
    if len(rows) != expected_rows:
        raise ValueError(f"TRC contains {len(rows)} frames, expected {expected_rows}")
    if start_index < 0 or end_index > expected_rows or start_index >= end_index:
        raise ValueError("Invalid TRC slice bounds")
    selected_rows = rows[start_index:end_index]
    metadata = lines[2].split()
    if len(metadata) < 8:
        raise ValueError(f"Invalid TRC frame metadata: {source}")
    metadata[2] = str(len(selected_rows))
    metadata[6] = str(start_index + 1)
    metadata[7] = str(expected_rows)
    lines[2] = "\t".join(metadata)
    output.write_text("\n".join(lines[:6] + selected_rows) + "\n", encoding="utf-8")
    return output


def unwrap_angles(ik_mot: Path) -> Path:
    """Remove 360-degree wraps that unclamped IK coordinates take between frames.

    A wrap is the same pose, but differentiating it gives huge accelerations. Only
    jumps above 180 per frame are touched, so translations in metres never change.
    """
    lines = ik_mot.read_text(encoding="utf-8").splitlines()
    endheader = lines.index("endheader")
    values = np.loadtxt(lines[endheader + 2 :], ndmin=2)
    values[:, 1:] = np.unwrap(values[:, 1:], period=360.0, axis=0)
    output_lines = lines[: endheader + 2]
    output_lines.extend("\t".join(f"{value:.10f}" for value in row) for row in values)
    ik_mot.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return ik_mot


def smooth_weak_coordinate_trajectories(
    ik_mot: Path,
    coordinate_names: tuple[str, ...] | list[str],
    sample_rate_hz: float,
    cutoff_hz: float,
    strength: float = 1.0,
) -> Path:
    """Low-pass only weakly observed coordinate columns in an IK motion."""
    if sample_rate_hz <= 0 or cutoff_hz <= 0 or cutoff_hz >= sample_rate_hz / 2:
        raise ValueError("cutoff_hz must be between zero and the Nyquist frequency")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be between zero and one")
    lines = ik_mot.read_text(encoding="utf-8").splitlines()
    endheader = lines.index("endheader")
    labels = lines[endheader + 1].split()
    values = np.loadtxt(lines[endheader + 2 :], ndmin=2)
    columns = {name: i for i, name in enumerate(labels)}
    selected = [columns[name] for name in coordinate_names if name in columns]
    if selected:
        sos = butter(4, cutoff_hz, fs=sample_rate_hz, output="sos")
        original = values[:, selected].copy()
        filtered = sosfiltfilt(sos, original, axis=0)
        values[:, selected] = original + strength * (filtered - original)
    output_lines = lines[: endheader + 2]
    output_lines.extend("\t".join(f"{value:.10f}" for value in row) for row in values)
    ik_mot.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return ik_mot


def set_coordinate_prior_targets(ik_mot: Path, targets: dict[str, float]) -> Path:
    """Set selected coordinate-prior columns to fixed targets in motion-file units."""
    lines = ik_mot.read_text(encoding="utf-8").splitlines()
    endheader = lines.index("endheader")
    labels = lines[endheader + 1].split()
    values = np.loadtxt(lines[endheader + 2 :], ndmin=2)
    columns = {name: i for i, name in enumerate(labels)}
    missing = sorted(set(targets) - set(columns))
    if missing:
        raise ValueError(f"Coordinate prior is missing target columns: {', '.join(missing)}")
    for name, value in targets.items():
        values[:, columns[name]] = value
    output_lines = lines[: endheader + 2]
    output_lines.extend("\t".join(f"{value:.10f}" for value in row) for row in values)
    ik_mot.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return ik_mot


def merge_mot_parts(
    parts: list[Path],
    out_mot: Path,
    expected_rows: int,
    keep_from_times: list[float] | None = None,
) -> Path:
    """Merge IK chunks, discarding warm-up rows and repeated boundary frames."""
    if keep_from_times is not None and len(keep_from_times) != len(parts):
        raise ValueError("keep_from_times must have one entry per IK part")
    header, rows, last_time = [], [], -np.inf
    for part_index, part in enumerate(parts):
        lines = part.read_text().splitlines()
        start = lines.index("endheader") + 2
        header = header or lines[:start]
        for line in lines[start:]:
            t = float(line.split()[0])
            if keep_from_times is not None and t < keep_from_times[part_index] - 1e-6:
                continue
            if t > last_time + 1e-6:
                rows.append(line)
                last_time = t
    if len(rows) != expected_rows:
        raise RuntimeError(f"IK produced {len(rows)} frames, expected {expected_rows}")
    header = [f"nRows={len(rows)}" if line.startswith("nRows=") else line for line in header]
    out_mot.write_text("\n".join(header + rows) + "\n")
    return out_mot


def mot_from_bridge_npz(npz_path: Path, model_path: Path, out_mot: Path) -> Path:
    """Rebuild an OpenSim IK .mot from DATASET-BSM's *_nimble_q_from_opensim_ik.npz.

    Inverts the bridge mapping in *_nimble_bridge_audit.json (q_N = q_OS by name,
    rad/m, two left-scapula signs flipped, knee_*_beta dropped); the dropped
    coordinates are recomputed from the model's CoordinateCouplerConstraints.
    """
    import opensim

    bridge = np.load(npz_path)
    names = [str(n) for n in bridge["dof_names"]]
    q = dict(zip(names, np.asarray(bridge["q_rad_m"], dtype=float).T))
    for name in ("scapula_elevation_l", "scapula_upward_rot_l"):
        q[name] = -q[name]
    model = opensim.Model(str(model_path))
    constraints = model.getConstraintSet()
    for i in range(constraints.getSize()):
        coupler = opensim.CoordinateCouplerConstraint.safeDownCast(constraints.get(i))
        if coupler is None:
            continue
        function = coupler.getFunction()
        independent = q[coupler.getIndependentCoordinateNames().get(0)]
        q[coupler.getDependentCoordinateName()] = float(coupler.getPropertyByName("scale_factor").toString()) * np.array(
            [function.calcValue(opensim.Vector(1, float(x))) for x in independent]
        )
    coordinates = model.getCoordinateSet()
    labels, columns = ["time"], [np.asarray(bridge["time_s"], dtype=float)]
    for i in range(coordinates.getSize()):
        coordinate = coordinates.get(i)
        values = q[coordinate.getName()]
        rotational = coordinate.getMotionType() == opensim.Coordinate.Rotational
        labels.append(coordinate.getName())
        columns.append(np.degrees(values) if rotational else values)
    data = np.column_stack(columns)
    header = f"Coordinates\nversion=1\nnRows={data.shape[0]}\nnColumns={data.shape[1]}\ninDegrees=yes\nendheader"
    np.savetxt(out_mot, data, fmt="%.10f", delimiter="\t", header=header + "\n" + "\t".join(labels), comments="")
    return out_mot


def ik_quality_metrics(
    ik_mot: Path,
    model_path: Path,
    marker_names: list[str],
    marker_positions_osim: np.ndarray,
) -> dict[str, float | int]:
    """Measure rotational jumps and RMS marker fit for an IK motion file."""
    import opensim

    lines = ik_mot.read_text().splitlines()
    endheader = lines.index("endheader")
    labels = lines[endheader + 1].split()
    values = np.loadtxt(lines[endheader + 2 :], ndmin=2)
    if values.shape[0] != marker_positions_osim.shape[0]:
        raise ValueError("IK motion and marker data have different frame counts")
    column = {name: i for i, name in enumerate(labels)}
    model = opensim.Model(str(model_path))
    coordinates = [model.getCoordinateSet().get(i) for i in range(model.getCoordinateSet().getSize())]
    rotational = [
        coordinate.getName()
        for coordinate in coordinates
        if coordinate.getMotionType() == opensim.Coordinate.Rotational and coordinate.getName() in column
    ]
    rotational_columns = [column[name] for name in rotational]
    jump_frames = int(
        np.sum(np.any(np.abs(np.diff(values[:, rotational_columns], axis=0)) > 10.0, axis=1))
    ) if rotational_columns else 0

    model_marker_names = {
        model.getMarkerSet().get(i).getName()
        for i in range(model.getMarkerSet().getSize())
    }
    selected = [(i, name) for i, name in enumerate(marker_names) if name in model_marker_names]
    if not selected:
        raise ValueError("No measured markers are present in the OpenSim model")
    marker_pairs = [(i, model.getMarkerSet().get(name)) for i, name in selected]
    state = model.initSystem()
    errors_squared: list[float] = []
    in_degrees = any(line.strip().lower() == "indegrees=yes" for line in lines[:endheader])
    for frame, row in enumerate(values):
        for coordinate in coordinates:
            name = coordinate.getName()
            if name not in column:
                continue
            value = float(row[column[name]])
            if in_degrees and coordinate.getMotionType() == opensim.Coordinate.Rotational:
                value = float(np.deg2rad(value))
            coordinate.setValue(state, value, False)
        model.realizePosition(state)
        for marker_index, marker in marker_pairs:
            target = marker_positions_osim[frame, marker_index]
            if not np.all(np.isfinite(target)):
                continue
            actual_vec = marker.getLocationInGround(state)
            actual = np.array([actual_vec.get(i) for i in range(3)])
            errors_squared.append(float(np.sum((actual - target) ** 2)))
    marker_rms_mm = 1000.0 * float(np.sqrt(np.mean(errors_squared))) if errors_squared else float("nan")
    return {"frames_with_rotational_jump_gt_10deg": jump_frames, "marker_rms_mm": marker_rms_mm}


def pelvis_residuals(torque_csv: Path, max_force_n: float, com_height_m: float) -> dict[str, dict[str, float]]:
    """Hicks et al. 2015: RMS force < 5% max|F_ext|, RMS moment < 1% of h_COM * max|F_ext|."""
    data = np.genfromtxt(torque_csv, delimiter=",", names=True)
    report = {}
    for dof in ("pelvis_tx", "pelvis_ty", "pelvis_tz", "pelvis_tilt", "pelvis_list", "pelvis_rotation"):
        values = data[f"{dof}_tau"]
        limit = 0.05 * max_force_n if dof.endswith(("_tx", "_ty", "_tz")) else 0.01 * com_height_m * max_force_n
        rms = float(np.sqrt(np.mean(values**2)))
        report[dof] = {"rms": rms, "max_abs": float(np.max(np.abs(values))), "limit_rms": limit, "pass": rms < limit}
    return report


def main() -> int:
    from opensim_batch_dynamics.final_csv_export import export_final_csv
    from opensim_batch_dynamics.inverse_dynamics_no_grf import (
        export_torque_csv_from_id_sto,
        run_inverse_dynamics_with_measured_grf,
    )
    from opensim_batch_dynamics.mot_to_csv import convert_mot_to_model_csv
    from opensim_batch_dynamics.trc_export import write_trc

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject-root", type=Path, required=True, help="e.g. ~/nas/MAEVE/EXO_RECORD/s5")
    parser.add_argument("--trial", required=True, help="e.g. posterior_reaching_1")
    parser.add_argument("--model", type=Path, help="default: <subject-root>/<subject>_opensim/calibration/model_opensim.osim")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/exo_record"))
    parser.add_argument("--q-npz", type=Path, help="reuse DATASET-BSM IK: <trial>/nimble_ik/<trial>_nimble_q_from_opensim_ik.npz")
    parser.add_argument("--ik-workers", type=int, default=16, help="parallel IK processes")
    parser.add_argument(
        "--ik-temporal-workers",
        type=int,
        default=1,
        help="workers for the final marker-constrained IK pass; 1 preserves the continuous trajectory",
    )
    parser.add_argument("--ik-warmup-s", type=float, default=IK_WARMUP_SECONDS, help="overlap before each IK chunk")
    parser.add_argument(
        "--ik-regularization-weight",
        type=float,
        default=IK_COORDINATE_PRIOR_WEIGHT,
        help="IKCoordinateTask weight for the second-pass temporal coordinate prior (0 disables)",
    )
    parser.add_argument(
        "--torso-prior",
        choices=sorted(TORSO_PRIOR_CHOICES),
        default="lumbar",
        help="trunk coordinates pulled toward the neutral pose in the second IK pass; "
        "the others track the smoothed first pass",
    )
    parser.add_argument(
        "--plate-force-offset",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("FX", "FY"),
        help="lab-frame horizontal force on the subject [N] subtracted from the plate (zero offset); "
        "use the mean horizontal force of a static trial",
    )
    parser.add_argument(
        "--plate-cop-offset",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("DX", "DY"),
        help="lab-frame shift [m] added to the plate CoP; use COM - CoP of a static trial",
    )
    parser.add_argument("--cutoff-hz", type=float, default=6.0, help="same low-pass for plate and coordinates")
    args = parser.parse_args()
    neutral_coordinates = TORSO_PRIOR_CHOICES[args.torso_prior]

    started = time.monotonic()
    root = args.subject_root.expanduser().resolve()
    subject = root.name
    model = (args.model or root / f"{subject}_opensim/calibration/model_opensim.osim").resolve()
    out = (args.output_dir / subject / args.trial).resolve()
    out.mkdir(parents=True, exist_ok=True)

    times, names, lab = load_markers(root / "aligned/marker" / f"{args.trial}.csv")
    names = [MARKER_ALIASES.get(n, n) for n in names]
    rate = 1.0 / float(np.median(np.diff(times)))

    ik_mot = out / f"{args.trial}_ik.mot"
    ik_runtime = None
    if args.q_npz:
        mot_from_bridge_npz(args.q_npz.expanduser(), model, ik_mot)
    elif ik_mot.exists():
        print(f"Reusing IK (delete to recompute): {ik_mot}")
    else:
        trc = write_trc(lab_to_osim(lab), names, out / f"{args.trial}.trc", rate, t_start=float(times[0]))
        ik_started = time.monotonic()
        pass1_mot = out / f"{args.trial}_ik_pass1.mot"
        if pass1_mot.exists():
            print(f"Reusing first IK pass (delete to recompute): {pass1_mot}")
        else:
            run_ik_parallel(
                model,
                trc,
                names,
                times,
                pass1_mot,
                args.ik_workers,
                warmup_seconds=args.ik_warmup_s,
                coordinate_prior_weight=IK_INITIAL_PRIOR_WEIGHT,
            )
        if args.ik_regularization_weight:
            coordinate_prior = out / f"{args.trial}_ik_coordinate_prior.mot"
            shutil.copyfile(pass1_mot, coordinate_prior)
            unwrap_angles(coordinate_prior)
            smooth_weak_coordinate_trajectories(
                coordinate_prior,
                WEAK_COORDINATES_TO_SMOOTH,
                sample_rate_hz=rate,
                cutoff_hz=args.cutoff_hz,
            )
            import opensim

            prior_model = opensim.Model(str(model))
            coordinate_set = prior_model.getCoordinateSet()
            neutral_targets = {}
            for name in neutral_coordinates:
                coordinate = coordinate_set.get(name)
                value = coordinate.getDefaultValue()
                if coordinate.getMotionType() == opensim.Coordinate.Rotational:
                    value = np.degrees(value)
                neutral_targets[name] = float(value)
            set_coordinate_prior_targets(coordinate_prior, neutral_targets)
            run_ik_parallel(
                model,
                trc,
                names,
                times,
                ik_mot,
                args.ik_temporal_workers,
                warmup_seconds=args.ik_warmup_s,
                coordinate_prior_weight=args.ik_regularization_weight,
                coordinate_prior_file=coordinate_prior,
                neutral_coordinates=neutral_coordinates,
            )
        else:
            shutil.copyfile(pass1_mot, ik_mot)
        ik_runtime = time.monotonic() - ik_started
    unwrap_angles(ik_mot)

    force, cop, free_moment = plate_wrench(root / "aligned/forceplate" / f"{args.trial}.csv", times, args.cutoff_hz)
    # ponytail: rigid lab-frame calibration from static trials; a per-channel plate
    # zero/tilt model is the upgrade if the offsets depend on load or posture.
    loaded = force[:, 2] > 0.0
    force[loaded, :2] -= args.plate_force_offset
    cop[loaded, :2] += args.plate_cop_offset
    index = {n: i for i, n in enumerate(names)}
    feet = []
    for side in "LR":
        heel, toe = lab[:, index[f"{side}HEE"]], lab[:, index[f"{side}TOE"]]
        centre = 0.5 * (heel + toe)
        height = np.minimum(heel[:, 2], toe[:, 2])
        on_plate = np.all((centre[:, :2] >= 0.0) & (centre[:, :2] <= PLATE_SIZE_M), axis=1)
        feet.append((centre, on_plate & (height < np.nanmin(height) + LIFT_THRESHOLD_M)))
    (foot_l, contact_l), (foot_r, contact_r) = feet
    grf, point, moment, right_fraction = split_feet(force, cop, free_moment, foot_l, foot_r, contact_l, contact_r)

    _, sto, _, _, wrench_csv = run_inverse_dynamics_with_measured_grf(
        model,
        ik_mot,
        out,
        args.trial,
        times,
        lab_to_osim(grf),
        lab_to_osim(point),
        contact_body_names=FEET,
        cutoff_hz=args.cutoff_hz,
        free_moment_nm=lab_to_osim(moment),
        z_up_gravity=False,
        forces_to_exclude="Actuators",
    )
    dof_csv = out / f"{args.trial}_dof.csv"
    convert_mot_to_model_csv(ik_mot, model, dof_csv, cutoff_hz=args.cutoff_hz)
    torque_csv, *_ = export_torque_csv_from_id_sto(sto, model, out / f"{args.trial}_torque.csv", times)
    final = export_final_csv(
        dof_csv,
        torque_csv,
        out / f"{args.trial}.csv",
        wrench_csv,
        model_path=model,
        excluded_dofs=("knee_angle_r_beta", "knee_angle_l_beta"),
    )

    max_force = float(np.max(np.linalg.norm(force, axis=1)))
    # ponytail: mean pelvis height stands in for COM height in the Hicks moment limit.
    com_height = float(np.mean(np.genfromtxt(dof_csv, delimiter=",", names=True)["pelvis_ty"]))
    summary = {
        "trial": args.trial,
        "model": str(model),
        "frame": "OpenSim, Y up, metres",
        "cutoff_hz": args.cutoff_hz,
        "ik_warmup_seconds": args.ik_warmup_s,
        "ik_initial_coordinate_prior_weight": IK_INITIAL_PRIOR_WEIGHT,
        "ik_coordinate_prior_weight": args.ik_regularization_weight,
        "ik_torso_neutral_prior_weight": args.ik_regularization_weight * TORSO_NEUTRAL_PRIOR_MULTIPLIER,
        "ik_torso_prior": args.torso_prior,
        "ik_torso_neutral_coordinates": list(neutral_coordinates),
        "plate_force_offset_lab_n": list(args.plate_force_offset),
        "plate_cop_offset_lab_m": list(args.plate_cop_offset),
        "ik_weak_coordinate_smoothing_cutoff_hz": args.cutoff_hz,
        "ik_initial_workers": args.ik_workers,
        "ik_temporal_workers": args.ik_temporal_workers,
        "ik_runtime_seconds": ik_runtime,
        "runtime_seconds": time.monotonic() - started,
        "ik_quality": ik_quality_metrics(ik_mot, model, names, lab_to_osim(lab)),
        "frames": final.frames,
        "mass_kg": final.mass_kg,
        "mean_vertical_force_over_weight": float(np.mean(force[:, 2]) / (final.mass_kg * 9.80665)),
        "right_fraction": {k: float(f(right_fraction)) for k, f in (("mean", np.mean), ("min", np.min), ("max", np.max))},
        "frames_single_support": int(np.sum(contact_l ^ contact_r)),
        "frames_no_foot_contact": int(np.sum(~contact_l & ~contact_r)),
        "free_moment_abs_max_nm": float(np.max(np.abs(free_moment))),
        "pelvis_residuals": pelvis_residuals(torque_csv, max_force, com_height),
        "final_csv": str(final.output_csv_path),
    }
    (out / f"{args.trial}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

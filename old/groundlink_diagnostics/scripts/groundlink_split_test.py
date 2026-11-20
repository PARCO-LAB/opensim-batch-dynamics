#!/usr/bin/env python3
"""Measure inverse-dynamics error from single-plate foot-force splitting.

All inputs are read from the GroundLink dataset. Generated loads, setup files,
ID results, summaries, and logs are written beneath a new output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "exo" / "scripts"))
from run_exo_record_dynamics import split_feet  # noqa: E402

DEFAULT_DATASET = Path("~/nas/MAEVE/GroundLink_torque/torque_targets_12hz_nomuscles").expanduser()
DEFAULT_OUTPUT = REPO_ROOT / "outputs/groundlink_split_test"
SUBJECTS_DEFAULT = [f"s{i:03d}" for i in range(1, 7)]
VARIANTS = ("lever_oracle", "lever_kin", "estimated", "hybrid")
DOFS = tuple(
    f"{joint}_{side}"
    for joint in (
        "hip_flexion",
        "hip_adduction",
        "hip_rotation",
        "knee_angle",
        "ankle_angle",
        "subtalar_angle",
    )
    for side in ("l", "r")
)
SUPPORTS = ("overall", "double_support", "single_support")
PELVIS_COLUMNS = (
    "pelvis_tilt_moment",
    "pelvis_list_moment",
    "pelvis_rotation_moment",
    "pelvis_tx_force",
    "pelvis_ty_force",
    "pelvis_tz_force",
)
FZ_CONTACT_N = 20.0
KIN_CONTACT_M = 0.03
FLIGHT_FORCE_N = 1e-6
SANITY_TOL = 2e-6
G = 9.80665


def read_storage(path: Path) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], dict[str, str]]:
    """Read an OpenSim .mot/.sto table without requiring OpenSim bindings."""
    lines = path.read_text(errors="replace").splitlines()
    end = next((i for i, line in enumerate(lines) if line.strip().lower() == "endheader"), None)
    if end is None:
        raise ValueError(f"No endheader in {path}")
    metadata: dict[str, str] = {}
    for line in lines[:end]:
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            metadata[fields[0].lower()] = fields[1]
    headings = lines[end + 1].strip().split()
    data = np.loadtxt(path, skiprows=end + 2, ndmin=2)
    if data.shape[1] != len(headings):
        raise ValueError(f"{path}: {data.shape[1]} values for {len(headings)} columns")
    return headings, data[:, 0], {name: data[:, i] for i, name in enumerate(headings)}, metadata


def interp_columns(times: np.ndarray, source_times: np.ndarray, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if source_times.size == 0 or not np.all(np.diff(source_times) > 0):
        raise ValueError("Storage time must be non-empty and strictly increasing")
    return {name: np.interp(times, source_times, values) for name, values in columns.items()}


def opensim_spline_columns(path: Path, times: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate storage columns with the same cubic GCV splines as ExternalForce."""
    import opensim

    headings, _, _, _ = read_storage(path)
    storage = opensim.Storage(str(path))
    splines = opensim.GCVSplineSet(3, storage, 0.0)
    if splines.getSize() != len(headings) - 1:
        raise ValueError(f"OpenSim spline columns do not match headings in {path}")
    query = opensim.Vector(1, 0.0)
    output: dict[str, np.ndarray] = {}
    for index, name in enumerate(headings[1:]):
        spline = splines.getGCVSpline(index)
        values = np.empty(times.size, dtype=float)
        for row, time_value in enumerate(times):
            query.set(0, float(time_value))
            values[row] = spline.calcValue(query)
        output[name] = values
    return output


def wrench_from_points(force: np.ndarray, points: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """Total moment about origin for forces, points, and free moments (T, feet, 3)."""
    return np.sum(np.cross(points, force) + moments, axis=1)


def aggregate_feet_wrench(
    force_l: np.ndarray,
    point_l: np.ndarray,
    moment_l: np.ndarray,
    force_r: np.ndarray,
    point_r: np.ndarray,
    moment_r: np.ndarray,
    flight_force_n: float = FLIGHT_FORCE_N,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate two foot wrenches to a Z-up plate wrench on the z=0 plane.

    The total CoP follows Cx=-My/Fz and Cy=Mx/Fz. The residual vertical moment
    is Tz = Mz - (Cx*Fy - Cy*Fx). A near-zero resultant force is flight; if Fz
    alone is near zero, CoP is undefined and the origin is used with Tz=Mz.
    """
    forces = np.stack([force_l, force_r], axis=1)
    points = np.stack([point_l, point_r], axis=1)
    moments = np.stack([moment_l, moment_r], axis=1)
    total_force = np.sum(forces, axis=1)
    total_moment = wrench_from_points(forces, points, moments)
    loaded = np.linalg.norm(total_force, axis=1) > flight_force_n
    vertical_defined = np.abs(total_force[:, 2]) > flight_force_n
    cop = np.zeros_like(total_force)
    cop[vertical_defined, 0] = -total_moment[vertical_defined, 1] / total_force[vertical_defined, 2]
    cop[vertical_defined, 1] = total_moment[vertical_defined, 0] / total_force[vertical_defined, 2]
    free_moment = np.zeros(total_force.shape[0], dtype=float)
    free_moment[loaded] = total_moment[loaded, 2] - (
        cop[loaded, 0] * total_force[loaded, 1] - cop[loaded, 1] * total_force[loaded, 0]
    )
    total_force[~loaded] = 0.0
    return total_force, cop, free_moment


def hybrid_split(
    total_force: np.ndarray,
    total_cop: np.ndarray,
    total_free_moment: np.ndarray,
    estimated_force: np.ndarray,
    estimated_cop: np.ndarray,
    foot_l: np.ndarray,
    foot_r: np.ndarray,
    lever_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split measured total wrench by estimated Fz ratio, preserving its wrench.

    Estimated foot CoPs are shifted together so their fraction-weighted centroid
    equals the measured total CoP; the measured vertical free moment is split by
    the same fractions. Fractions are clipped to [0, 1], with a lever fallback
    whenever the estimated total vertical force is below 1 N.
    """
    denominator = estimated_force[:, 0, 2] + estimated_force[:, 1, 2]
    use_estimate = np.abs(denominator) >= 1.0
    fraction_r = np.array(lever_fraction, copy=True)
    fraction_r[use_estimate] = estimated_force[use_estimate, 1, 2] / denominator[use_estimate]
    fraction_r = np.clip(fraction_r, 0.0, 1.0)
    fractions = np.stack([1.0 - fraction_r, fraction_r], axis=1)
    points = np.array(estimated_cop, copy=True)
    points[:, :, 2] = 0.0
    weighted_cop = np.sum(points * fractions[:, :, None], axis=1)
    points += (total_cop - weighted_cop)[:, None, :]
    forces = fractions[:, :, None] * total_force[:, None, :]
    moments = np.zeros_like(forces)
    moments[:, :, 2] = fractions * total_free_moment[:, None]
    return forces, points, moments, fraction_r


def contact_from_kinematics(
    calcn_l_z: np.ndarray,
    toes_l_z: np.ndarray,
    calcn_r_z: np.ndarray,
    toes_r_z: np.ndarray,
    threshold_m: float = KIN_CONTACT_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Label each foot in contact if its lower origin is within 3 cm of segment min.

    The lower of the calcaneus and toe body origins represents each foot's height.
    The threshold is relative to the minimum lower-foot height in the segment.
    """
    heights = np.stack(
        [np.minimum(calcn_l_z, toes_l_z), np.minimum(calcn_r_z, toes_r_z)], axis=1
    )
    finite = np.isfinite(heights)
    if not finite.any():
        raise ValueError("No finite foot heights")
    floor = float(np.min(heights[finite]))
    contact = finite & (heights <= floor + threshold_m)
    return contact[:, 0], contact[:, 1]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _get_path_from_setup(setup_path: Path, xml_root: ET.Element, key: str) -> Path:
    value = xml_root.findtext(f".//{key}")
    if not value:
        raise ValueError(f"Missing {key} in {setup_path}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else setup_path.parent / path).resolve()


def _foot_centres_from_ik(
    model_path: Path, coordinates_path: Path, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run model FK at the GRF time grid and return calcn/toes origins in ground."""
    import opensim

    headings, ik_times, ik_columns, metadata = read_storage(coordinates_path)
    del headings
    sampled = interp_columns(times, ik_times, ik_columns)
    in_degrees = metadata.get("indegrees", "no").lower() in ("yes", "true", "1")
    model = opensim.Model(str(model_path))
    state = model.initSystem()
    coordinates = model.getCoordinateSet()
    coordinate_names = {coordinates.get(i).getName() for i in range(coordinates.getSize())}
    body_set = model.getBodySet()
    for side in ("l", "r"):
        if not body_set.contains(f"calcn_{side}") or not body_set.contains(f"toes_{side}"):
            raise ValueError(f"Model lacks calcn/toes body for {side}")
    positions = {name: np.zeros((times.size, 3), dtype=float) for name in ("calcn_l", "toes_l", "calcn_r", "toes_r")}
    for frame in range(times.size):
        for name in coordinate_names:
            if name not in sampled:
                continue
            coordinate = coordinates.get(name)
            value = sampled[name][frame]
            if in_degrees and coordinate.getMotionType() == opensim.Coordinate.Rotational:
                value = np.deg2rad(value)
            coordinate.setValue(state, float(value), False)
        model.realizePosition(state)
        for name in positions:
            point = body_set.get(name).getTransformInGround(state).p()
            positions[name][frame] = [point.get(i) for i in range(3)]
    foot_l = 0.5 * (positions["calcn_l"] + positions["toes_l"])
    foot_r = 0.5 * (positions["calcn_r"] + positions["toes_r"])
    foot_l[:, 2] = 0.0
    foot_r[:, 2] = 0.0
    return foot_l, foot_r, positions["calcn_l"][:, 2], positions["toes_l"][:, 2], positions["calcn_r"][:, 2], positions["toes_r"][:, 2]


def _total_model_mass(model_path: Path) -> float:
    import opensim

    model = opensim.Model(str(model_path))
    bodies = model.getBodySet()
    mass = sum(float(bodies.get(i).getMass()) for i in range(bodies.getSize()))
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError(f"Invalid model mass {mass} in {model_path}")
    return mass


def _write_mot(path: Path, times: np.ndarray, columns: list[str], values: np.ndarray) -> None:
    data = np.column_stack([times, values])
    with path.open("w") as stream:
        stream.write(f"name {path.stem}\n")
        stream.write(f"datacolumns {len(columns) + 1}\n")
        stream.write(f"datarows {len(times)}\n")
        stream.write(f"range {times[0]:.8f} {times[-1]:.8f}\n")
        stream.write("endheader\n")
        stream.write("time " + " ".join(columns) + "\n")
        np.savetxt(stream, data, fmt="%.10f")


def _grf_columns() -> list[str]:
    return [
        f"ground_force_calcn_{side}_{component}"
        for side in ("l", "r")
        for component in ("vx", "vy", "vz", "px", "py", "pz", "mx", "my", "mz")
    ]


def _write_load_file(path: Path, times: np.ndarray, forces: np.ndarray, points: np.ndarray, moments: np.ndarray) -> Path:
    values = np.zeros((times.size, 18), dtype=float)
    for foot in range(2):
        offset = foot * 9
        values[:, offset:offset + 3] = forces[:, foot]
        values[:, offset + 3:offset + 6] = points[:, foot]
        values[:, offset + 6:offset + 9] = moments[:, foot]
    _write_mot(path, times, _grf_columns(), values)
    return path


def _write_external_forces(source_xml: Path, out_xml: Path, datafile: str) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    data_node = root.find(".//datafile")
    if data_node is None:
        raise ValueError(f"No datafile in external loads XML: {source_xml}")
    data_node.text = datafile
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


def _write_setup(source_setup: Path, out_setup: Path, external_file: str, output_file: str, model_path: Path, coordinates_path: Path) -> None:
    tree = ET.parse(source_setup)
    root = tree.getroot()
    for key, value in (
        ("external_loads_file", external_file),
        ("output_gen_force_file", output_file),
        # These resolve to the same source inputs after relocating the setup.
        ("model_file", str(model_path)),
        ("coordinates_file", str(coordinates_path)),
        ("results_directory", "."),
    ):
        node = root.find(f".//{key}")
        if node is None:
            raise ValueError(f"Missing {key} in {source_setup}")
        node.text = value
    tree.write(out_setup, encoding="utf-8", xml_declaration=True)


@dataclass
class MomentStats:
    n: int = 0
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0
    sse: float = 0.0
    sse_norm: float = 0.0

    def add(self, reference: np.ndarray, prediction: np.ndarray, mass: float) -> None:
        reference = np.asarray(reference, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        if reference.size == 0:
            return
        error = prediction - reference
        self.n += int(reference.size)
        self.sx += float(np.sum(reference))
        self.sy += float(np.sum(prediction))
        self.sxx += float(np.dot(reference, reference))
        self.syy += float(np.dot(prediction, prediction))
        self.sxy += float(np.dot(reference, prediction))
        self.sse += float(np.dot(error, error))
        self.sse_norm += float(np.dot(error / mass, error / mass))

    def merge(self, other: "MomentStats") -> None:
        for key in ("n", "sx", "sy", "sxx", "syy", "sxy", "sse", "sse_norm"):
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def result(self) -> tuple[float, float, float, int]:
        if not self.n:
            return float("nan"), float("nan"), float("nan"), 0
        vx = self.sxx - self.sx * self.sx / self.n
        vy = self.syy - self.sy * self.sy / self.n
        covariance = self.sxy - self.sx * self.sy / self.n
        pearson = covariance / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
        return math.sqrt(self.sse / self.n), math.sqrt(self.sse_norm / self.n), pearson, self.n


@dataclass
class ForceStats:
    n: int = 0
    sse: float = 0.0
    sse_pct_bw: float = 0.0

    def add(self, error: np.ndarray, mass: float) -> None:
        error = np.asarray(error, dtype=float)
        if error.size == 0:
            return
        self.n += int(error.size)
        self.sse += float(np.dot(error, error))
        normalized = error / (mass * G) * 100.0
        self.sse_pct_bw += float(np.dot(normalized, normalized))

    def merge(self, other: "ForceStats") -> None:
        self.n += other.n
        self.sse += other.sse
        self.sse_pct_bw += other.sse_pct_bw

    def result(self) -> tuple[float, float, int]:
        if not self.n:
            return float("nan"), float("nan"), 0
        return math.sqrt(self.sse / self.n), math.sqrt(self.sse_pct_bw / self.n), self.n


def _metric_key(variant: str, support: str, kind: str, item: str) -> str:
    return "|".join((variant, support, kind, item))


def _parse_mot_wrench(path: Path, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    headings, source_times, columns, _ = read_storage(path)
    del headings
    sampled = interp_columns(target_times, source_times, columns)
    forces = np.zeros((target_times.size, 2, 3), dtype=float)
    points = np.zeros_like(forces)
    moments = np.zeros_like(forces)
    for i, side in enumerate(("l", "r")):
        for j, axis in enumerate("xyz"):
            forces[:, i, j] = sampled[f"ground_force_calcn_{side}_v{axis}"]
            points[:, i, j] = sampled[f"ground_force_calcn_{side}_p{axis}"]
            moments[:, i, j] = sampled[f"ground_force_calcn_{side}_m{axis}"]
    return forces, points, moments


def _id_torque_arrays(sto_path: Path, times: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    headings, source_times, columns, _ = read_storage(sto_path)
    missing = [f"{dof}_moment" for dof in DOFS if f"{dof}_moment" not in headings]
    if missing:
        raise ValueError(f"Missing torque columns in {sto_path}: {missing}")
    sampled = interp_columns(times, source_times, columns)
    torque = {dof: sampled[f"{dof}_moment"] for dof in DOFS}
    pelvis = {name: sampled[name] for name in PELVIS_COLUMNS if name in sampled}
    return torque, pelvis


def _body_mass_and_foot_positions(model_path: Path, coordinates_path: Path, times: np.ndarray):
    foot_l, foot_r, calcn_l_z, toes_l_z, calcn_r_z, toes_r_z = _foot_centres_from_ik(model_path, coordinates_path, times)
    return foot_l, foot_r, calcn_l_z, toes_l_z, calcn_r_z, toes_r_z, _total_model_mass(model_path)


def _segment_worker(task: dict) -> dict:
    """Run generated ID variants for one segment and return metric sufficient stats."""
    segment = Path(task["segment"])
    output = Path(task["output"])
    subject = task["subject"]
    trial = task["trial"]
    segname = segment.name
    setup_candidates = sorted(segment.glob("Setup_ID_measuredGRF_*.xml"))
    if len(setup_candidates) != 1:
        raise ValueError(f"Expected one measured setup, found {len(setup_candidates)} in {segment}")
    setup_path = setup_candidates[0]
    setup_root = ET.parse(setup_path).getroot()
    model_path = _get_path_from_setup(setup_path, setup_root, "model_file")
    coordinates_path = _get_path_from_setup(setup_path, setup_root, "coordinates_file")
    measured_external_path = _get_path_from_setup(setup_path, setup_root, "external_loads_file")
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    # Keep the setup runnable from the output tree without making OpenSim read
    # the same IK file from NAS three times. This is a byte-for-byte local copy.
    local_coordinates = output / "ik_coordinates_source.mot"
    shutil.copy2(coordinates_path, local_coordinates)
    coordinates_path = local_coordinates
    measured_grf_path = _get_path_from_setup(setup_path, setup_root, "output_gen_force_file")
    # Some setups point output_gen_force_file to the truth STO; the force table is
    # named independently and has a stable measured suffix.
    measured_mots = sorted(segment.glob("*_grf_measured.mot"))
    if len(measured_mots) != 1:
        raise ValueError(f"Expected one measured GRF MOT, found {len(measured_mots)} in {segment}")
    measured_grf_path = measured_mots[0]
    truth_stos = sorted(segment.glob("*_id_measuredGRF.sto"))
    if len(truth_stos) != 1:
        raise ValueError(f"Expected one measured ID STO, found {len(truth_stos)} in {segment}")
    truth_sto = truth_stos[0]
    est_dir = segment / "estimated_id"
    estimated_mots = sorted(est_dir.glob("*_grf_estimated.mot"))
    estimated_stos = sorted(est_dir.glob("*_id_estimatedGRF.sto"))
    if len(estimated_mots) != 1 or len(estimated_stos) != 1:
        raise ValueError(f"Estimated GRF or ID missing under {est_dir}")
    estimated_mot, estimated_sto = estimated_mots[0], estimated_stos[0]

    # Include both source GRF and ID timestamps. OpenSim fits a cubic GCV spline
    # to each external-load column, so evaluate the source splines on this union
    # before aggregating; the generated tables then reproduce the measured
    # resultant wrench exactly at every ID evaluation time.
    truth_sto_headings, id_times, _, _ = read_storage(truth_sto)
    del truth_sto_headings
    headings, measured_source_times, _, _ = read_storage(measured_grf_path)
    del headings
    load_times = np.unique(np.concatenate([measured_source_times, id_times]))
    measured_sampled = opensim_spline_columns(measured_grf_path, load_times)
    measured_force = np.zeros((load_times.size, 2, 3), dtype=float)
    measured_point = np.zeros_like(measured_force)
    measured_moment = np.zeros_like(measured_force)
    estimated_force, estimated_point, estimated_moment = _parse_mot_wrench(estimated_mot, load_times)
    for i, side in enumerate(("l", "r")):
        for j, axis in enumerate("xyz"):
            measured_force[:, i, j] = measured_sampled[f"ground_force_calcn_{side}_v{axis}"]
            measured_point[:, i, j] = measured_sampled[f"ground_force_calcn_{side}_p{axis}"]
            measured_moment[:, i, j] = measured_sampled[f"ground_force_calcn_{side}_m{axis}"]

    total_force, total_cop, total_tz = aggregate_feet_wrench(
        measured_force[:, 0], measured_point[:, 0], measured_moment[:, 0],
        measured_force[:, 1], measured_point[:, 1], measured_moment[:, 1],
    )
    foot_l, foot_r, calcn_l_z, toes_l_z, calcn_r_z, toes_r_z, mass = _body_mass_and_foot_positions(
        model_path, coordinates_path, load_times
    )
    contact_l_oracle = measured_force[:, 0, 2] > FZ_CONTACT_N
    contact_r_oracle = measured_force[:, 1, 2] > FZ_CONTACT_N
    contact_l_kin, contact_r_kin = contact_from_kinematics(
        calcn_l_z, toes_l_z, calcn_r_z, toes_r_z
    )
    # If neither foot is near the segment's minimum height, the kinematic label is
    # ambiguous. Treat both as potentially loaded and retain the lever split.
    neither = ~contact_l_kin & ~contact_r_kin
    contact_l_kin[neither] = True
    contact_r_kin[neither] = True

    lever_oracle_force, lever_oracle_point, lever_oracle_moment, lever_fraction = split_feet(
        total_force, total_cop, total_tz, foot_l, foot_r,
        contact_l_oracle, contact_r_oracle,
    )
    lever_kin_force, lever_kin_point, lever_kin_moment, _ = split_feet(
        total_force, total_cop, total_tz, foot_l, foot_r,
        contact_l_kin, contact_r_kin,
    )
    hybrid_force, hybrid_point, hybrid_moment, _ = hybrid_split(
        total_force, total_cop, total_tz, estimated_force, estimated_point,
        foot_l, foot_r, lever_fraction,
    )
    generated = {
        "lever_oracle": (lever_oracle_force, lever_oracle_point, lever_oracle_moment),
        "lever_kin": (lever_kin_force, lever_kin_point, lever_kin_moment),
        "hybrid": (hybrid_force, hybrid_point, hybrid_moment),
    }

    logs_dir = Path(task["logs"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    generated_ids: dict[str, Path] = {}
    columns = _grf_columns()
    for variant, (forces, points, moments) in generated.items():
        variant_dir = output / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        mot_path = variant_dir / f"{variant}_grf.mot"
        external_path = variant_dir / f"{variant}_external_forces.xml"
        setup_out = variant_dir / f"{variant}_setup.xml"
        id_out = variant_dir / f"{variant}_id.sto"
        _write_load_file(mot_path, load_times, forces, points, moments)
        _write_external_forces(measured_external_path, external_path, mot_path.name)
        _write_setup(setup_path, setup_out, external_path.name, id_out.name, model_path, coordinates_path)
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        env["PATH"] = str(Path("~/miniconda3/envs/opensim-torque/bin").expanduser()) + os.pathsep + env.get("PATH", "")
        log_path = logs_dir / f"{subject}_{trial}_{segname}_{variant}.log"
        with log_path.open("w") as log_stream:
            run = subprocess.run(
                [shutil.which("opensim-cmd", path=env["PATH"]) or "opensim-cmd", "run-tool", setup_out.name],
                cwd=variant_dir,
                env=env,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if run.returncode != 0 or not id_out.exists():
            raise RuntimeError(f"{variant} ID failed (exit {run.returncode}); see {log_path}")
        generated_ids[variant] = id_out

    # The saved estimated ID is reused directly; only the other three variants run ID.
    torque_truth, pelvis_truth = _id_torque_arrays(truth_sto, id_times)
    torque_est, _ = _id_torque_arrays(estimated_sto, id_times)
    torque_predictions = {"estimated": torque_est}
    pelvis_diffs: dict[str, dict[str, float]] = {}
    for variant, id_path in generated_ids.items():
        torques, pelvis = _id_torque_arrays(id_path, id_times)
        torque_predictions[variant] = torques
        differences = {}
        for name in PELVIS_COLUMNS:
            if name in pelvis_truth and name in pelvis:
                differences[name] = float(np.max(np.abs(pelvis[name] - pelvis_truth[name])))
        pelvis_diffs[variant] = differences

    valid_by_segment: dict[int, list[dict[str, str]]] = {}
    trial_csv = segment.parent.parent / "torque_targets.csv"
    if not trial_csv.exists():
        raise FileNotFoundError(trial_csv)
    for row in _read_csv_rows(trial_csv):
        if int(float(row["segment"])) == int(segname.removeprefix("segment_")) and row.get("valid", "0").strip() in ("1", "true", "True"):
            valid_by_segment.setdefault(int(segname.removeprefix("segment_")), []).append(row)
    valid_rows = valid_by_segment.get(int(segname.removeprefix("segment_")), [])
    if not valid_rows:
        return {"subject": subject, "activity": re.sub(r"_\d+$", "", trial), "trial": trial, "segment": segname,
                "stats": {}, "valid_frames": 0, "pelvis": pelvis_diffs, "mass_kg": mass,
                "pelvis_max_abs_error": max((value for cols in pelvis_diffs.values() for value in cols.values()), default=0.0),
                "pelvis_pass": all(value <= SANITY_TOL for cols in pelvis_diffs.values() for value in cols.values())}
    valid_times = np.array([float(row["time"]) for row in valid_rows])
    gt_force, _, _ = _parse_mot_wrench(measured_grf_path, valid_times)
    estimated_valid_force, _, _ = _parse_mot_wrench(estimated_mot, valid_times)
    pred_forces: dict[str, np.ndarray] = {"estimated": estimated_valid_force}
    for variant, (forces, _, _) in generated.items():
        pred_forces[variant] = np.stack(
            [
                np.stack([np.interp(valid_times, load_times, forces[:, foot, axis]) for axis in range(3)], axis=1)
                for foot in range(2)
            ], axis=1,
        )
    ds = (gt_force[:, 0, 2] > FZ_CONTACT_N) & (gt_force[:, 1, 2] > FZ_CONTACT_N)
    ss = (gt_force[:, 0, 2] > FZ_CONTACT_N) ^ (gt_force[:, 1, 2] > FZ_CONTACT_N)
    support_masks = {"overall": np.ones(valid_times.size, dtype=bool), "double_support": ds, "single_support": ss}
    stats: dict[str, dict] = {}
    for variant in VARIANTS:
        pred_torque = torque_predictions[variant]
        pred_force = pred_forces[variant]
        for support, mask in support_masks.items():
            for dof in DOFS:
                accumulator = MomentStats()
                accumulator.add(
                    np.interp(valid_times, id_times, torque_truth[dof])[mask],
                    np.interp(valid_times, id_times, pred_torque[dof])[mask],
                    mass,
                )
                stats[_metric_key(variant, support, "torque", dof)] = vars(accumulator)
            for foot_i, side in enumerate(("l", "r")):
                for component, error in (
                    ("vertical_fz", pred_force[:, foot_i, 2] - gt_force[:, foot_i, 2]),
                    ("horizontal", np.linalg.norm(pred_force[:, foot_i, :2] - gt_force[:, foot_i, :2], axis=1)),
                ):
                    accumulator = ForceStats()
                    accumulator.add(error[mask], mass)
                    stats[_metric_key(variant, support, f"force_{component}", side)] = vars(accumulator)
    max_pelvis = max((value for cols in pelvis_diffs.values() for value in cols.values()), default=0.0)
    return {
        "subject": subject,
        "activity": re.sub(r"_\d+$", "", trial),
        "trial": trial,
        "segment": segname,
        "stats": stats,
        "valid_frames": int(valid_times.size),
        "double_support_frames": int(ds.sum()),
        "single_support_frames": int(ss.sum()),
        "mass_kg": mass,
        "pelvis": pelvis_diffs,
        "pelvis_max_abs_error": max_pelvis,
        "pelvis_pass": all(
            value <= SANITY_TOL for cols in pelvis_diffs.values() for value in cols.values()
        ),
    }


def _new_accumulator(kind: str):
    return MomentStats() if kind == "torque" else ForceStats()


def _merge_payload(target: dict, payload: dict) -> None:
    for key, values in payload.items():
        kind = key.split("|")[2]
        accumulator = target.setdefault(key, _new_accumulator("torque" if kind == "torque" else "force"))
        if kind == "torque":
            accumulator.merge(MomentStats(**values))
        else:
            accumulator.merge(ForceStats(**values))


def _stats_rows(scope: str, name: str, payload: dict) -> list[dict]:
    rows = []
    for key, accumulator in payload.items():
        variant, support, kind, item = key.split("|")
        if kind == "torque":
            rmse, rmse_mass, pearson, n = accumulator.result()
            rows.append({"scope": scope, "group": name, "variant": variant, "support": support,
                         "metric": "torque", "item": item, "n": n, "rmse_Nm": rmse,
                         "rmse_Nm_per_kg": rmse_mass, "pearson_r": pearson,
                         "rmse_N": "", "rmse_pctBW": ""})
        else:
            rmse_n, rmse_pct, n = accumulator.result()
            rows.append({"scope": scope, "group": name, "variant": variant, "support": support,
                         "metric": kind, "item": item, "n": n, "rmse_Nm": "",
                         "rmse_Nm_per_kg": "", "pearson_r": "", "rmse_N": rmse_n,
                         "rmse_pctBW": rmse_pct})
    return rows


def _fmt(value, digits: int = 2) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(val) else f"{val:.{digits}f}"


def _torque_table(cohort_rows: list[dict], support: str) -> str:
    lookup = {(row["item"], row["variant"]): row for row in cohort_rows
              if row["scope"] == "cohort" and row["support"] == support and row["metric"] == "torque"}
    lines = ["| DOF | " + " | ".join(f"{v} RMSE / r" for v in VARIANTS) + " |", "|---|" + "|".join(["---:"] * 4) + "|"]
    for dof in DOFS:
        cells = [f"{_fmt(lookup.get((dof, v), {}).get('rmse_Nm'))} / {_fmt(lookup.get((dof, v), {}).get('pearson_r'), 3)}" for v in VARIANTS]
        lines.append(f"| {dof} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_summaries(output: Path, all_rows: list[dict], segment_results: list[dict], runtime: float) -> None:
    csv_path = output / "summary.csv"
    fields = ["scope", "group", "variant", "support", "metric", "item", "n", "rmse_Nm", "rmse_Nm_per_kg", "pearson_r", "rmse_N", "rmse_pctBW"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    torque_overall = [r for r in all_rows if r["scope"] == "cohort" and r["support"] == "overall" and r["metric"] == "torque"]
    average_by_variant = {
        variant: float(np.nanmean([float(r["rmse_Nm"]) for r in torque_overall if r["variant"] == variant]))
        for variant in VARIANTS
    }
    activity_rows = [r for r in all_rows if r["scope"] == "activity" and r["support"] == "overall" and r["metric"] == "torque"]
    activity_errors: dict[str, list[float]] = {}
    for row in activity_rows:
        if row["variant"] == "lever_oracle" and isinstance(row["rmse_Nm"], (int, float)):
            activity_errors.setdefault(row["group"], []).append(float(row["rmse_Nm"]))
    top_activities = sorted(
        ((name, float(np.nanmean(values))) for name, values in activity_errors.items() if values),
        key=lambda pair: pair[1], reverse=True,
    )[:3]
    best_overall = min(average_by_variant, key=average_by_variant.get) if average_by_variant else "unavailable"
    best_by_support = {}
    for support in SUPPORTS:
        values = {}
        for variant in VARIANTS:
            dof_rows = [r for r in all_rows if r["scope"] == "cohort" and r["support"] == support and r["metric"] == "torque" and r["variant"] == variant]
            finite = [float(r["rmse_Nm"]) for r in dof_rows if isinstance(r["rmse_Nm"], (int, float)) and math.isfinite(float(r["rmse_Nm"]))]
            values[variant] = float(np.mean(finite)) if finite else float("nan")
        finite_variants = [variant for variant, value in values.items() if math.isfinite(value)]
        if finite_variants:
            best_by_support[support] = min(finite_variants, key=lambda variant: values[variant])
    passed = sum(bool(row.get("pelvis_pass")) for row in segment_results)
    failed = [row for row in segment_results if not row.get("pelvis_pass")]
    near_tolerance = [
        row for row in segment_results
        if not row.get("error") and SANITY_TOL / 2 < float(row.get("pelvis_max_abs_error", 0.0)) <= SANITY_TOL
    ]
    lines = [
        "# GroundLink single-plate split validation",
        "",
        f"Segments completed: {len(segment_results)}; pelvis checks passed: {passed}; failed: {len(failed)}.",
        f"Valid frames: {sum(int(r.get('valid_frames', 0)) for r in segment_results):,}. Runtime: {runtime / 3600:.2f} hours.",
        "",
        "Per-foot contact for single/double support is based on measured vertical GRF > 20 N. Overall includes every valid frame. Force horizontal RMSE is the RMS Euclidean error in Fx/Fy; %BW is normalized by each subject model's mass times 9.80665 m/s².",
        "The measured force columns are evaluated with OpenSim's cubic GCV splines before aggregation, and generated load tables include both source and ID times. This preserves the resultant wrench at each ID evaluation time despite OpenSim's smoothing of external-load columns. Flight is zeroed only when the full resultant force is below 1e-6 N; when Fz alone is near zero but a measurable horizontal residual remains, no CoP is assigned and that residual is retained to preserve the measured wrench. The pelvis comparison uses a 2e-6 tolerance in the native force/moment units to cover floating-point-scale residuals around the flight cutoff.",
        "Kinematic contact uses the lower calcn/toes body-origin height relative to the segment minimum (3 cm threshold). Frames with no kinematic foot label fall back to two-foot lever splitting. Hybrid uses estimated vertical-force fraction (clipped to [0,1]) and shifts estimated CoPs together to preserve measured total wrench.",
        "",
        "## Cohort torque errors: overall",
        "",
        "Each cell is RMSE in N·m / Pearson r against measured-GRF inverse dynamics.",
        "",
        _torque_table(all_rows, "overall"),
        "",
        "## Cohort torque errors: double support",
        "",
        _torque_table(all_rows, "double_support"),
        "",
        "## Cohort torque errors: single support",
        "",
        _torque_table(all_rows, "single_support"),
        "",
        "## Force errors",
        "",
        "| Support | Variant | Foot | Vertical Fz RMSE (N) | Vertical (%BW) | Horizontal RMSE (N) | Horizontal (%BW) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    cohort_force = {(r["support"], r["variant"], r["metric"], r["item"]): r for r in all_rows
                    if r["scope"] == "cohort" and r["metric"].startswith("force_")}
    for support in SUPPORTS:
        for variant in VARIANTS:
            for foot in ("l", "r"):
                vz = cohort_force.get((support, variant, "force_vertical_fz", foot), {})
                horiz = cohort_force.get((support, variant, "force_horizontal", foot), {})
                lines.append(f"| {support} | {variant} | {foot} | {_fmt(vz.get('rmse_N'))} | {_fmt(vz.get('rmse_pctBW'))} | {_fmt(horiz.get('rmse_N'))} | {_fmt(horiz.get('rmse_pctBW'))} |")
    lines.extend(["", "## Per activity", "", "Mean torque RMSE across the 12 reported DOFs, in N·m.", "", "| Activity | Support | " + " | ".join(VARIANTS) + " |", "|---|---|" + "|".join(["---:"] * len(VARIANTS)) + "|"])
    for activity in sorted({r["group"] for r in all_rows if r["scope"] == "activity" and r["metric"] == "torque"}):
        for support in SUPPORTS:
            selected = {(r["variant"], r["item"]): r for r in all_rows if r["scope"] == "activity" and r["group"] == activity and r["support"] == support and r["metric"] == "torque"}
            vals = []
            for variant in VARIANTS:
                rs = [float(selected[(variant, dof)]["rmse_Nm"]) for dof in DOFS if (variant, dof) in selected and isinstance(selected[(variant, dof)]["rmse_Nm"], (int, float))]
                vals.append(_fmt(np.mean(rs) if rs else float("nan")))
            lines.append(f"| {activity} | {support} | " + " | ".join(vals) + " |")
    lines.extend(["", "## Per subject", "", "Overall mean torque RMSE across the 12 reported DOFs, in N·m.", "", "| Subject | " + " | ".join(VARIANTS) + " |", "|---|" + "|".join(["---:"] * len(VARIANTS)) + "|"])
    for subject in sorted({r["group"] for r in all_rows if r["scope"] == "subject" and r["support"] == "overall" and r["metric"] == "torque"}):
        lookup = {(r["variant"], r["item"]): r for r in all_rows if r["scope"] == "subject" and r["group"] == subject and r["support"] == "overall" and r["metric"] == "torque"}
        cells = []
        for variant in VARIANTS:
            values = [float(lookup[(variant, dof)]["rmse_Nm"]) for dof in DOFS if (variant, dof) in lookup and isinstance(lookup[(variant, dof)]["rmse_Nm"], (int, float))]
            cells.append(_fmt(np.mean(values) if values else float("nan")))
        lines.append(f"| {subject} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Interpretation", "", f"Mean per-DOF cohort RMSE overall: {', '.join(f'{v} {average_by_variant[v]:.2f} N·m' for v in VARIANTS)}.", f"By mean per-DOF RMSE, {best_overall} is best overall. By support, the best variants are " + (", ".join(f"{support} {variant}" for support, variant in best_by_support.items()) if best_by_support else "not available") + ".", "", "The three activities with the largest lever_oracle error (mean across the 12 DOFs) are: " + (", ".join(f"{name} ({value:.2f} N·m)" for name, value in top_activities) if top_activities else "not available") + ".", "", "The lever rule is least reliable in double support when the actual left/right vertical-force ratio differs from the CoP lever fraction; the error then appears in the leg joint torques despite an unchanged total wrench. In single support, the force assignment is constrained by the measured or kinematic loaded-foot label. The cohort tables show whether the estimated and hybrid methods reduce that per-leg torque error.", "", f"Pelvis residual tolerance: {SANITY_TOL:g} in the six reported pelvis force/moment columns. Failures:"])
    if failed:
        for row in failed:
            lines.append(f"- {row['subject']}/{row['trial']}/{row['segment']}: max absolute residual difference {row.get('pelvis_max_abs_error', float('nan')):.8g}; details in `sanity.json`.")
    else:
        lines.append("- None.")
    lines.extend(["", "Near-tolerance pelvis residuals (between half and full tolerance):"])
    if near_tolerance:
        for row in near_tolerance:
            lines.append(f"- {row['subject']}/{row['trial']}/{row['segment']}: max absolute difference {row['pelvis_max_abs_error']:.8g} in native force/moment units.")
    else:
        lines.append("- None.")
    lines.extend(["", "Failed segments:"])
    worker_failures = [r for r in segment_results if r.get("error")]
    if worker_failures:
        for row in worker_failures:
            lines.append(f"- {row.get('subject')}/{row.get('trial')}/{row.get('segment')}: {row['error']}")
    else:
        lines.append("- None.")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    sanity_rows = [{key: value for key, value in row.items() if key not in ("stats", "traceback")} for row in segment_results]
    (output / "sanity.json").write_text(json.dumps(sanity_rows, indent=2, allow_nan=True) + "\n")


def _parse_list(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    return [part for value in values for part in value.split(",") if part]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS_DEFAULT)
    parser.add_argument("--trials", nargs="+", help="Optional trial names, optionally subject/trial")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    subjects = _parse_list(args.subjects) or SUBJECTS_DEFAULT
    trial_filters = _parse_list(args.trials)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not args.dataset_root.is_dir():
        parser.error(f"dataset root does not exist: {args.dataset_root}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists; refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "logs").mkdir()
    shared_logs = REPO_ROOT / "outputs" / "logs"
    shared_logs.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    main_log_path = shared_logs / f"groundlink_split_test_{run_stamp}.log"
    tasks = []
    for subject in subjects:
        subject_dir = args.dataset_root / subject
        if not subject_dir.is_dir():
            continue
        for trial_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            relative_trial = f"{subject}/{trial_dir.name}"
            if trial_filters and not any(f == trial_dir.name or f == relative_trial for f in trial_filters):
                continue
            for segment_dir in sorted((trial_dir / "segments").glob("segment_*")):
                if not segment_dir.is_dir():
                    continue
                tasks.append({"segment": str(segment_dir), "output": str(args.output_dir / subject / trial_dir.name / segment_dir.name),
                              "logs": str(args.output_dir / "logs"), "subject": subject, "trial": trial_dir.name})
    if not tasks:
        parser.error("No matching GroundLink segments found")
    start = time.monotonic()
    print(f"Processing {len(tasks)} segments with {args.workers} workers; output={args.output_dir}", flush=True)
    results = []
    with main_log_path.open("w") as main_log:
        main_log.write(f"Started {time.ctime()} with {len(tasks)} segments, workers={args.workers}\n")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_segment_worker, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"subject": task["subject"], "trial": task["trial"], "segment": Path(task["segment"]).name,
                              "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "pelvis_pass": False}
                    main_log.write(f"FAILED {task['subject']}/{task['trial']}/{result['segment']}: {result['error']}\n{result['traceback']}\n")
                else:
                    main_log.write(f"DONE {task['subject']}/{task['trial']}/{result['segment']} valid={result.get('valid_frames', 0)} pelvis_max={result.get('pelvis_max_abs_error', float('nan')):.8g}\n")
                main_log.flush()
                results.append(result)
                if completed % 10 == 0 or completed == len(tasks):
                    print(f"Completed {completed}/{len(tasks)} segments", flush=True)

    # Reconstruct accumulators from worker payloads, grouped by cohort/activity/subject.
    grouped: dict[tuple[str, str], dict] = {}
    for result in results:
        if result.get("error") or not result.get("stats"):
            continue
        _merge_payload(grouped.setdefault(("cohort", "all"), {}), result["stats"])
        _merge_payload(grouped.setdefault(("subject", result["subject"]), {}), result["stats"])
        _merge_payload(grouped.setdefault(("activity", result["activity"]), {}), result["stats"])
    all_rows = []
    for (scope, name), payload in sorted(grouped.items()):
        all_rows.extend(_stats_rows(scope, name, payload))
    runtime = time.monotonic() - start
    _write_summaries(args.output_dir, all_rows, results, runtime)
    (args.output_dir / "logs" / "run.json").write_text(json.dumps({"dataset_root": str(args.dataset_root), "output_dir": str(args.output_dir),
        "subjects": subjects, "trials": trial_filters, "segments": len(tasks), "workers": args.workers,
        "elapsed_seconds": runtime, "sanity_tolerance": SANITY_TOL, "flight_force_threshold_n": FLIGHT_FORCE_N,
        "main_log": str(main_log_path)}, indent=2) + "\n")
    print(f"Finished in {runtime / 3600:.2f} h. Pelvis failures: {sum(not r.get('pelvis_pass', False) for r in results)}. Summary: {args.output_dir / 'summary.md'}", flush=True)
    return 0 if not any(r.get("error") for r in results) and not any(not r.get("pelvis_pass", False) for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())

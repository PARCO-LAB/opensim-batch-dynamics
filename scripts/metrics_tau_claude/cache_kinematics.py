#!/usr/bin/env python3
"""Cache COM and foot/contact-point kinematics for GroundLink diagnostics.

This reads model and IK inputs only. It writes compressed NumPy caches below
the requested study scratch directory and never copies Geometry assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt


SOURCE_ROOT = Path("/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque")
TARGET_DIRNAME = "torque_targets_12hz_nomuscles_copfix_tcode"
SUBJECTS = ("s001", "s002", "s003", "s004", "s005", "s006")
BODY_NAMES = ("calcn_l", "toes_l", "calcn_r", "toes_r")


def lowpass(values: np.ndarray, fs_hz: float, cutoff_hz: float = 12.0) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.shape[0] < 16 or not math.isfinite(fs_hz) or fs_hz <= 0:
        return data.copy()
    cutoff = min(cutoff_hz, 0.49 * fs_hz)
    if cutoff <= 0:
        return data.copy()
    b, a = butter(4, cutoff / (fs_hz / 2.0))
    try:
        return filtfilt(b, a, data, axis=0)
    except ValueError:
        return data.copy()


def resolve_setup_inputs(setup_path: Path) -> tuple[Path, Path]:
    root = ET.parse(setup_path).getroot().find("InverseDynamicsTool")
    if root is None:
        raise ValueError(f"No InverseDynamicsTool in {setup_path}")
    model_node, coordinates_node = root.find("model_file"), root.find("coordinates_file")
    if model_node is None or not model_node.text or coordinates_node is None or not coordinates_node.text:
        raise ValueError(f"Missing model/coordinates path in {setup_path}")
    model = Path(model_node.text.strip()).expanduser()
    coordinates = Path(coordinates_node.text.strip()).expanduser()
    if not model.is_absolute():
        model = (setup_path.parent / model).resolve()
    if not coordinates.is_absolute():
        coordinates = (setup_path.parent / coordinates).resolve()
    if not model.is_file() or not coordinates.is_file():
        raise FileNotFoundError(f"Missing IK/model from {setup_path}: {coordinates}, {model}")
    return model, coordinates


def map_source_path(path: Path, torque_root: Path) -> Path:
    """Map the HPC data root in run_metadata.json to the selected local mirror."""
    path = path.expanduser()
    try:
        relative = path.relative_to(SOURCE_ROOT)
    except ValueError:
        return path
    return (torque_root / relative).resolve()


def compute_segment_cache(osim: object, skel: object, ik_path: Path) -> dict[str, np.ndarray]:
    import nimblephysics as nimble

    motion = nimble.biomechanics.OpenSimParser.loadMot(skel, str(ik_path))
    poses = np.asarray(motion.poses, dtype=float)
    times = np.asarray(motion.timestamps, dtype=float).reshape(-1)
    if poses.ndim != 2 or poses.shape[1] != times.size or times.size < 2:
        raise ValueError(f"Unexpected IK pose/time shapes: {poses.shape}, {times.shape}")
    dt = np.diff(times)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    if positive_dt.size == 0:
        raise ValueError(f"Cannot infer sample rate from {ik_path}")
    fs_hz = 1.0 / float(np.median(positive_dt))
    names = [skel.getBodyNode(i).getName() for i in range(skel.getNumBodyNodes())]
    missing = [name for name in BODY_NAMES if name not in names]
    if missing:
        raise ValueError(f"Missing contact bodies in {ik_path}: {missing}")
    body_nodes = [skel.getBodyNode(name) for name in BODY_NAMES]
    all_body_nodes = [skel.getBodyNode(i) for i in range(skel.getNumBodyNodes())]
    com = np.zeros((times.size, 3), dtype=float)
    body_positions = np.zeros((times.size, len(BODY_NAMES), 3), dtype=float)
    centroidal_angular_momentum = np.zeros((times.size, 3), dtype=float)
    # OpenSim rotational coordinates wrap at +/- pi. Unwrap before differentiating
    # so branch crossings do not become impossible angular velocities.
    continuous_poses = np.unwrap(poses, axis=1)
    pose_velocities = np.gradient(continuous_poses, times, axis=1, edge_order=2 if times.size > 2 else 1)

    markers_by_side: dict[str, list[tuple[object, np.ndarray]]] = {"l": [], "r": []}
    markers_map = getattr(osim, "markersMap", {})
    if isinstance(markers_map, dict):
        selected = set(BODY_NAMES)
        for body, offset in markers_map.values():
            name = body.getName()
            if name in selected:
                side = name[-1]
                markers_by_side[side].append((body, np.asarray(offset, dtype=float).reshape(3)))
    marker_positions = {
        side: np.zeros((len(entries), times.size, 3), dtype=float)
        for side, entries in markers_by_side.items()
    }
    for frame in range(times.size):
        skel.setPositions(poses[:, frame])
        skel.setVelocities(pose_velocities[:, frame])
        com[frame] = np.asarray(skel.getCOM(), dtype=float).reshape(3)
        centroidal_angular_momentum[frame] = sum(
            (np.asarray(body.getAngularMomentum(com[frame]), dtype=float).reshape(3) for body in all_body_nodes),
            start=np.zeros(3, dtype=float),
        )
        for index, body in enumerate(body_nodes):
            body_positions[frame, index] = np.asarray(body.getWorldTransform().translation(), dtype=float).reshape(3)
        for side, entries in markers_by_side.items():
            for point_index, (body, offset) in enumerate(entries):
                marker_positions[side][point_index, frame] = np.asarray(
                    body.getWorldTransform().multiply(offset), dtype=float
                ).reshape(3)

    com_filtered = lowpass(com, fs_hz)
    centroidal_angular_momentum_filtered = lowpass(centroidal_angular_momentum, fs_hz)
    centroidal_angular_momentum_rate = np.gradient(
        centroidal_angular_momentum_filtered, times, axis=0
    )
    body_filtered = lowpass(body_positions, fs_hz)
    body_velocity = np.gradient(body_filtered, times, axis=0)
    body_speed = np.linalg.norm(body_velocity, axis=2)
    body_height = body_filtered[:, :, 2]

    point_heights = []
    point_data: dict[str, np.ndarray] = {}
    for side in "lr":
        if marker_positions[side].shape[0]:
            positions = np.transpose(marker_positions[side], (1, 0, 2))
        else:
            indices = [BODY_NAMES.index(f"calcn_{side}"), BODY_NAMES.index(f"toes_{side}")]
            positions = body_positions[:, indices, :]
        filtered = lowpass(positions, fs_hz)
        point_heights.append(filtered[:, :, 2].reshape(-1))
        low_height = np.min(filtered[:, :, 2], axis=1)
        centroid = np.mean(filtered, axis=1)
        velocity = np.gradient(centroid, times, axis=0)
        point_data[f"det_low_height_{side}"] = low_height
        point_data[f"det_speed_{side}"] = np.linalg.norm(velocity, axis=1)
    floor_height = float(np.percentile(np.concatenate(point_heights), 1.0)) if point_heights else 0.0
    for side in "lr":
        point_data[f"det_low_height_{side}"] -= floor_height

    result = {
        "times": times,
        "com_positions": com_filtered,
        "centroidal_angular_momentum": centroidal_angular_momentum_filtered,
        "centroidal_angular_momentum_rate": centroidal_angular_momentum_rate,
        "body_positions": body_filtered,
        "body_heights_above_floor": body_height - floor_height,
        "body_speeds": body_speed,
        "floor_height": np.asarray([floor_height]),
        "body_names": np.asarray(BODY_NAMES),
        "mass_kg": np.asarray([float(skel.getMass())]),
        **point_data,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--cache-subdir", default="kinematics_cache", help="Versioned cache folder under phase_b")
    parser.add_argument("--subject", choices=SUBJECTS, default=None)
    parser.add_argument("--subject-index", type=int, default=None, help="Defaults to SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()
    if args.subject is None:
        import os
        index = args.subject_index
        if index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if index is None or not 0 <= index < len(SUBJECTS):
            parser.error("Specify --subject or a valid --subject-index/SLURM_ARRAY_TASK_ID")
        subject = SUBJECTS[index]
    else:
        subject = args.subject
    cache_root = args.study_root / "phase_b" / args.cache_subdir / subject
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "cache_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"Cache manifest already exists: {manifest_path}")

    import nimblephysics as nimble

    trials = sorted((args.torque_root / TARGET_DIRNAME / subject).glob("*/run_metadata.json"))
    if not trials:
        raise FileNotFoundError(f"No run_metadata.json files for {subject}")
    manifest_rows: list[dict[str, object]] = []
    for trial_index, metadata_path in enumerate(trials, start=1):
        metadata = json.loads(metadata_path.read_text())
        trial = str(metadata["trial"])
        segment_dirs = sorted((metadata_path.parent / "segments").glob("segment_*"))
        if not segment_dirs:
            raise FileNotFoundError(f"No segments in {metadata_path.parent}")
        setup_paths = []
        for segment_dir in segment_dirs:
            estimated_dir = segment_dir / "estimated_id"
            setup_paths.append(next(iter(estimated_dir.glob("Setup_ID_estimatedGRF_*.xml"))))
        source_model = Path(metadata["source_paths"]["model"]).expanduser()
        if not source_model.is_absolute():
            source_model = (args.torque_root / source_model).resolve()
        model_path = map_source_path(source_model, args.torque_root).resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing source model from {metadata_path}: {model_path}")
        osim = nimble.biomechanics.OpenSimParser.parseOsim(str(model_path), "", True)
        skel = osim.skeleton
        cache_arrays: dict[str, np.ndarray] = {}
        for segment_dir, setup_path in zip(segment_dirs, setup_paths):
            segment_index = int(segment_dir.name.split("_")[-1])
            _, ik_path = resolve_setup_inputs(setup_path)
            part = compute_segment_cache(osim, skel, ik_path)
            prefix = f"segment_{segment_index}__"
            cache_arrays.update({prefix + key: value for key, value in part.items()})
            manifest_rows.append({
                "subject": subject,
                "trial": trial,
                "segment": segment_index,
                "cache_file": str(cache_root / f"{trial}.npz"),
                "cache_prefix": prefix,
                "frames": len(part["times"]),
                "model_path": str(model_path),
            })
        cache_path = cache_root / f"{trial}.npz"
        if cache_path.exists():
            raise FileExistsError(cache_path)
        np.savez_compressed(cache_path, **cache_arrays)
        print(f"{subject}/{trial} segments={len(segment_dirs)} cached={cache_path}", flush=True)

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"subject={subject} trials={len(trials)} segments={len(manifest_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

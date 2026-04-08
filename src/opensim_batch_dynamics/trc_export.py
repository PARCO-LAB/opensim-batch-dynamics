from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _axis_rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for one axis rotation in degrees."""
    axis = axis.lower()
    angle_rad = math.radians(float(angle_deg))
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    raise ValueError(f"Invalid rotation axis '{axis}', expected one of x/y/z")


def apply_axis_rotations(points: np.ndarray, rotations_deg: dict[str, float] | None) -> np.ndarray:
    """Apply x/y/z rotations in order to a (T, N, 3) points array."""
    if rotations_deg is None:
        return points
    rotated = points.copy()
    for axis in ("x", "y", "z"):
        if axis not in rotations_deg:
            continue
        rot = _axis_rotation_matrix(axis, rotations_deg[axis])
        rotated = np.einsum("ij,tnj->tni", rot, rotated)
    return rotated


def _write_header(
    f,
    num_frames: int,
    num_markers: int,
    marker_names: list[str],
    frame_rate_hz: float,
    units: str,
) -> None:
    """Write a minimal TRC header accepted by OpenSim tools."""
    f.write(f"PathFileType  4\t(X/Y/Z) {Path.cwd()}\n")
    f.write(
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\t"
        "Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
    )
    f.write(
        f"{frame_rate_hz:.1f}\t{frame_rate_hz:.1f}\t{num_frames}\t{num_markers}\t"
        f"{units}\t{frame_rate_hz:.1f}\t1\t{num_frames}\n"
    )
    f.write("Frame#\tTime\t")
    for name in marker_names:
        f.write(f"{name}\t\t\t")
    f.write("\n\t\t")
    for i in range(1, num_markers + 1):
        f.write(f"X{i}\tY{i}\tZ{i}\t")
    f.write("\n\n")


def write_trc(
    marker_positions: np.ndarray,
    marker_names: list[str],
    output_path: str | Path,
    frame_rate_hz: float,
    t_start: float = 0.0,
    units: str = "m",
    rotations_deg: dict[str, float] | None = None,
    vertical_offset: float | None = None,
) -> Path:
    """Write marker trajectories to a TRC file."""
    if marker_positions.ndim != 3 or marker_positions.shape[2] != 3:
        raise ValueError(
            f"Expected marker_positions shape (T, N, 3), got {marker_positions.shape}"
        )
    if marker_positions.shape[1] != len(marker_names):
        raise ValueError(
            "Marker count mismatch between positions and marker names: "
            f"{marker_positions.shape[1]} vs {len(marker_names)}"
        )

    data = marker_positions.astype(np.float32, copy=True)
    data = apply_axis_rotations(data, rotations_deg)
    if vertical_offset is not None:
        data[:, :, 1] += float(-vertical_offset + 0.01)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    num_frames, num_markers, _ = data.shape
    with output.open("w", encoding="utf-8") as f:
        _write_header(f, num_frames, num_markers, marker_names, frame_rate_hz, units)
        for frame_idx in range(num_frames):
            time_s = (frame_idx / frame_rate_hz) + t_start
            f.write(f"{frame_idx + 1}\t{time_s:.8f}\t")
            for marker_idx in range(num_markers):
                x, y, z = data[frame_idx, marker_idx]
                f.write(f"{x:.5f}\t{y:.5f}\t{z:.5f}\t")
            f.write("\n")

    return output


def infer_trc_time_range(path_trc: str | Path) -> tuple[float, float]:
    """Read first and last timestamps from a TRC file."""
    path = Path(path_trc)
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    # OpenSim TRC usually has 6 header lines; data starts at index 6.
    data_lines = [line.strip() for line in lines[6:] if line.strip()]
    if not data_lines:
        raise ValueError(f"TRC has no data rows: {path}")

    def _time(row: str) -> float:
        parts = row.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid TRC row format: {row}")
        return float(parts[1])

    return _time(data_lines[0]), _time(data_lines[-1])

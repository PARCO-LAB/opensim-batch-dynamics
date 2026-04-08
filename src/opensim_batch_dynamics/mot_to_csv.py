from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MotionCsvSummary:
    """Summary returned after converting a .mot file into a model-ordered CSV."""

    input_rows: int
    model_dofs: int
    mapped_dofs: int
    missing_dofs: tuple[str, ...]
    output_csv_path: Path
    filter_cutoff_hz: float | None
    sample_rate_hz: float


def extract_coordinate_names_from_osim(osim_path: Path) -> list[str]:
    """Read coordinate names from an OpenSim model XML."""
    text = osim_path.read_text(encoding="utf-8", errors="ignore")
    coords = re.findall(r'<Coordinate name="([^"]+)"', text)
    if not coords:
        raise ValueError(f"No coordinates found in model: {osim_path}")
    return coords


def _canonical_mot_label(label: str) -> str:
    """
    Normalize OpenSim labels.

    Example:
    - '/jointset/hip_r/hip_flexion_r/value' -> 'hip_flexion_r'
    - 'pelvis_tilt' -> 'pelvis_tilt'
    """
    label = label.strip()
    if "/" not in label:
        return label
    parts = [part for part in label.split("/") if part]
    if not parts:
        return label
    if parts[-1].lower() == "value" and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def parse_mot(path: Path) -> tuple[list[str], list[list[float]]]:
    """Parse a .mot file with OpenSim-style header ending at 'endheader'."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    end_idx = None
    for idx, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            end_idx = idx
            break
    if end_idx is None:
        raise ValueError(f"Invalid .mot file (missing endheader): {path}")

    labels_line = None
    labels_idx = None
    for idx in range(end_idx + 1, len(lines)):
        if lines[idx].strip():
            labels_line = lines[idx].strip()
            labels_idx = idx
            break
    if labels_line is None or labels_idx is None:
        raise ValueError(f"Invalid .mot file (missing labels row): {path}")

    labels = labels_line.split()
    rows: list[list[float]] = []
    for line in lines[labels_idx + 1 :]:
        row_text = line.strip()
        if not row_text:
            continue
        values = row_text.split()
        if len(values) != len(labels):
            # Skip malformed rows instead of failing hard.
            continue
        rows.append([float(value) for value in values])

    if not rows:
        raise ValueError(f"Invalid .mot file (no numeric rows): {path}")

    return labels, rows


def parse_missing_fill(raw: str) -> float:
    """Parse CLI fill value; supports numeric values or 'nan'."""
    if raw.lower() == "nan":
        return math.nan
    return float(raw)


def _build_column_lookup(mot_labels: list[str]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for idx, label in enumerate(mot_labels):
        lookup[_canonical_mot_label(label)] = idx
    return lookup


def _infer_sample_rate(time_values: np.ndarray) -> float:
    if time_values.ndim != 1 or time_values.size < 2:
        raise ValueError("Time vector must have at least two samples.")
    diffs = np.diff(time_values)
    finite = diffs[np.isfinite(diffs)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        raise ValueError("Could not infer sample rate from time column.")
    dt = float(np.median(finite))
    return 1.0 / dt


def infer_cutoff_hz(filter_mode: str, mot_path: Path) -> float | None:
    """
    Infer low-pass cutoff from motion type.

    - walking -> 12 Hz
    - dynamic -> 30 Hz
    - auto -> 12 Hz for gait-like names, else 30 Hz
    - none -> no filtering
    """
    mode = filter_mode.lower()
    if mode == "none":
        return None
    if mode == "walking":
        return 12.0
    if mode == "dynamic":
        return 30.0
    if mode != "auto":
        raise ValueError(
            "Invalid filter_mode. Expected one of: auto, walking, dynamic, none."
        )

    trial_name = mot_path.stem.lower()
    gait_tokens = ("walk", "walking", "gait", "cammino", "camminata")
    if any(token in trial_name for token in gait_tokens):
        return 12.0
    return 30.0


def _lowpass_butterworth_4th(
    values: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float | None,
) -> np.ndarray:
    """
    Apply 4th-order zero-phase Butterworth low-pass filter.

    If cutoff_hz is None, the input is returned unchanged.
    """
    if cutoff_hz is None:
        return values.copy()
    if values.size < 20:
        # Too short to reliably apply filtfilt.
        return values.copy()

    try:
        from scipy.signal import butter, filtfilt
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for Butterworth filtering. "
            "Install it in your environment."
        ) from exc

    nyquist_hz = 0.5 * float(sample_rate_hz)
    effective_cutoff = min(float(cutoff_hz), 0.95 * nyquist_hz)
    if effective_cutoff <= 0:
        return values.copy()

    b, a = butter(4, effective_cutoff, btype="low", fs=float(sample_rate_hz))
    return filtfilt(b, a, values)


def _differentiate(time_values: np.ndarray, signal_values: np.ndarray) -> np.ndarray:
    """Differentiate with respect to time using centered finite differences."""
    edge_order = 2 if signal_values.size >= 3 else 1
    return np.gradient(signal_values, time_values, edge_order=edge_order)


def convert_mot_to_model_csv(
    mot_path: Path,
    model_path: Path,
    out_csv_path: Path,
    missing_fill: float = math.nan,
    include_time: bool = True,
    include_frame: bool = True,
    add_velocity: bool = True,
    add_acceleration: bool = True,
    filter_mode: str = "auto",
    cutoff_hz: float | None = None,
) -> MotionCsvSummary:
    """
    Convert .mot to CSV with columns ordered exactly as model DOFs.

    Every output row is a frame. Every output DOF column follows the order found in
    the target OpenSim model.
    """
    model_dofs = extract_coordinate_names_from_osim(model_path)
    labels, rows = parse_mot(mot_path)
    lookup = _build_column_lookup(labels)

    if "time" not in lookup:
        raise ValueError("Input .mot does not contain 'time' column.")

    row_matrix = np.asarray(rows, dtype=np.float64)
    time_values = row_matrix[:, lookup["time"]]
    sample_rate_hz = _infer_sample_rate(time_values)

    inferred_cutoff = cutoff_hz
    if inferred_cutoff is None:
        inferred_cutoff = infer_cutoff_hz(filter_mode=filter_mode, mot_path=mot_path)

    # Build one-time mapping DOF -> source column index (or None if missing).
    dof_to_col: dict[str, int | None] = {}
    for dof in model_dofs:
        dof_to_col[dof] = lookup.get(dof)

    # Precompute filtered positions (and optional derivatives) per DOF.
    positions_by_dof: dict[str, np.ndarray] = {}
    velocities_by_dof: dict[str, np.ndarray] = {}
    accelerations_by_dof: dict[str, np.ndarray] = {}

    for dof in model_dofs:
        source_col = dof_to_col[dof]
        if source_col is None:
            filled = np.full(row_matrix.shape[0], missing_fill, dtype=np.float64)
            positions_by_dof[dof] = filled
            if add_velocity:
                velocities_by_dof[dof] = np.full_like(filled, missing_fill)
            if add_acceleration:
                accelerations_by_dof[dof] = np.full_like(filled, missing_fill)
            continue

        raw_signal = row_matrix[:, source_col]
        filtered_signal = _lowpass_butterworth_4th(
            values=raw_signal,
            sample_rate_hz=sample_rate_hz,
            cutoff_hz=inferred_cutoff,
        )
        positions_by_dof[dof] = filtered_signal

        if add_velocity:
            velocities_by_dof[dof] = _differentiate(time_values, filtered_signal)
        if add_acceleration:
            base_for_acc = velocities_by_dof[dof] if add_velocity else _differentiate(
                time_values, filtered_signal
            )
            accelerations_by_dof[dof] = _differentiate(time_values, base_for_acc)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    header: list[str] = []
    if include_frame:
        header.append("frame")
    if include_time:
        header.append("time")
    header.extend(model_dofs)
    if add_velocity:
        header.extend([f"{dof}_vel" for dof in model_dofs])
    if add_acceleration:
        header.extend([f"{dof}_acc" for dof in model_dofs])

    with out_csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        for frame_idx in range(row_matrix.shape[0]):
            out_row: list[float | int] = []
            if include_frame:
                out_row.append(frame_idx)
            if include_time:
                out_row.append(time_values[frame_idx])
            for dof in model_dofs:
                out_row.append(float(positions_by_dof[dof][frame_idx]))
            if add_velocity:
                for dof in model_dofs:
                    out_row.append(float(velocities_by_dof[dof][frame_idx]))
            if add_acceleration:
                for dof in model_dofs:
                    out_row.append(float(accelerations_by_dof[dof][frame_idx]))
            writer.writerow(out_row)

    missing = tuple(dof for dof, source_col in dof_to_col.items() if source_col is None)
    mapped = len(model_dofs) - len(missing)
    return MotionCsvSummary(
        input_rows=row_matrix.shape[0],
        model_dofs=len(model_dofs),
        mapped_dofs=mapped,
        missing_dofs=missing,
        output_csv_path=out_csv_path,
        filter_cutoff_hz=inferred_cutoff,
        sample_rate_hz=sample_rate_hz,
    )

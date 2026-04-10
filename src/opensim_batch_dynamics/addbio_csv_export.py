from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mot_to_csv import extract_coordinate_names_from_osim, parse_mot


@dataclass(frozen=True)
class AddBiomechanicsCsvSummary:
    """Summary for the final BSM CSV export."""

    output_csv_path: Path
    model_path: Path
    mot_path: Path
    dof_names: tuple[str, ...]
    frames: int
    velocity_source: str


def _canonical_label(label: str) -> str:
    label = label.strip()
    if "/" not in label:
        return label
    parts = [part for part in label.split("/") if part]
    if not parts:
        return label
    if parts[-1].lower() == "value" and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def _differentiate(time_values: np.ndarray, signal_values: np.ndarray) -> np.ndarray:
    edge_order = 2 if signal_values.size >= 3 else 1
    return np.gradient(signal_values, time_values, edge_order=edge_order)


def _load_mot_signals(
    mot_path: Path,
    dof_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels, rows = parse_mot(mot_path)
    label_lookup = {_canonical_label(label): idx for idx, label in enumerate(labels)}
    time_values = np.asarray([row[0] for row in rows], dtype=np.float64)

    positions: list[np.ndarray] = []
    missing: list[str] = []
    for dof_name in dof_names:
        if dof_name not in label_lookup:
            positions.append(np.full(time_values.shape, np.nan, dtype=np.float64))
            missing.append(dof_name)
            continue
        column = label_lookup[dof_name]
        positions.append(np.asarray([row[column] for row in rows], dtype=np.float64))

    return time_values, np.asarray(positions, dtype=np.float64), missing


def export_addbiomechanics_csv(
    final_model_path: str | Path,
    final_mot_path: str | Path,
    output_csv_path: str | Path,
    use_subject_on_disk: object | None = None,
) -> AddBiomechanicsCsvSummary:
    """
    Export a model-ordered CSV from the final AddBiomechanics outputs.

    The first version uses the final ``.mot`` as the authoritative source for
    positions and derives velocities/accelerations with ``np.gradient``.
    """
    model_path = Path(final_model_path).resolve()
    mot_path = Path(final_mot_path).resolve()
    output_path = Path(output_csv_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dof_names = extract_coordinate_names_from_osim(model_path)
    time_values, position_columns, _ = _load_mot_signals(mot_path, dof_names)
    if time_values.size < 2:
        raise ValueError(f"Motion file has too few samples to differentiate: {mot_path}")

    position_matrix = position_columns.T
    velocity_matrix = np.vstack([
        _differentiate(time_values, position_matrix[:, idx]) for idx in range(position_matrix.shape[1])
    ]).T
    acceleration_matrix = np.vstack([
        _differentiate(time_values, velocity_matrix[:, idx]) for idx in range(velocity_matrix.shape[1])
    ]).T

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["frame", "time"]
        header.extend(dof_names)
        header.extend(f"{name}_vel" for name in dof_names)
        header.extend(f"{name}_acc" for name in dof_names)
        writer.writerow(header)

        for frame_idx, time_s in enumerate(time_values):
            row = [frame_idx, float(time_s)]
            row.extend(float(position_matrix[frame_idx, idx]) for idx in range(position_matrix.shape[1]))
            row.extend(float(velocity_matrix[frame_idx, idx]) for idx in range(velocity_matrix.shape[1]))
            row.extend(float(acceleration_matrix[frame_idx, idx]) for idx in range(acceleration_matrix.shape[1]))
            writer.writerow(row)

    return AddBiomechanicsCsvSummary(
        output_csv_path=output_path,
        model_path=model_path,
        mot_path=mot_path,
        dof_names=tuple(dof_names),
        frames=int(time_values.size),
        velocity_source="numerical_derivative_fallback",
    )

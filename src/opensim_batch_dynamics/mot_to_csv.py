from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MotionCsvSummary:
    """Summary returned after converting a .mot file into a model-ordered CSV."""

    input_rows: int
    model_dofs: int
    mapped_dofs: int
    missing_dofs: tuple[str, ...]
    output_csv_path: Path


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


def convert_mot_to_model_csv(
    mot_path: Path,
    model_path: Path,
    out_csv_path: Path,
    missing_fill: float = math.nan,
    include_time: bool = True,
    include_frame: bool = True,
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

    # Build one-time mapping DOF -> source column index (or None if missing).
    dof_to_col: dict[str, int | None] = {}
    for dof in model_dofs:
        dof_to_col[dof] = lookup.get(dof)

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    header: list[str] = []
    if include_frame:
        header.append("frame")
    if include_time:
        header.append("time")
    header.extend(model_dofs)

    with out_csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        for frame_idx, row in enumerate(rows):
            out_row: list[float | int] = []
            if include_frame:
                out_row.append(frame_idx)
            if include_time:
                out_row.append(row[lookup["time"]])
            for dof in model_dofs:
                source_col = dof_to_col[dof]
                out_row.append(row[source_col] if source_col is not None else missing_fill)
            writer.writerow(out_row)

    missing = tuple(dof for dof, source_col in dof_to_col.items() if source_col is None)
    mapped = len(model_dofs) - len(missing)
    return MotionCsvSummary(
        input_rows=len(rows),
        model_dofs=len(model_dofs),
        mapped_dofs=mapped,
        missing_dofs=missing,
        output_csv_path=out_csv_path,
    )

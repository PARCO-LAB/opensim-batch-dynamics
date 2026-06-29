from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .mot_to_csv import convert_mot_to_model_csv, extract_coordinate_names_from_osim


@dataclass(frozen=True)
class AddBiomechanicsCsvSummary:
    """Summary for the final BSM CSV export."""

    output_csv_path: Path
    model_path: Path
    mot_path: Path
    dof_names: tuple[str, ...]
    frames: int
    velocity_source: str


def export_addbiomechanics_csv(
    final_model_path: str | Path,
    final_mot_path: str | Path,
    output_csv_path: str | Path,
) -> AddBiomechanicsCsvSummary:
    """Export a model-ordered CSV from the final AddBiomechanics outputs."""
    model_path = Path(final_model_path).resolve()
    mot_path = Path(final_mot_path).resolve()
    output_path = Path(output_csv_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_summary = convert_mot_to_model_csv(
        mot_path=mot_path,
        model_path=model_path,
        out_csv_path=output_path,
        missing_fill=math.nan,
        include_time=True,
        include_frame=True,
        add_velocity=True,
        add_acceleration=True,
        filter_mode="none",
    )
    dof_names = extract_coordinate_names_from_osim(model_path)

    return AddBiomechanicsCsvSummary(
        output_csv_path=output_path,
        model_path=model_path,
        mot_path=mot_path,
        dof_names=tuple(dof_names),
        frames=int(convert_summary.input_rows),
        velocity_source="numerical_derivative_fallback",
    )

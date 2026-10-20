from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .mot_to_csv import convert_mot_to_model_csv, extract_coordinate_names_from_osim
from .segments import discover_segment_mot_paths, merge_csv_segments


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
    filter_mode: str = "none",
    cutoff_hz: float | None = None,
) -> AddBiomechanicsCsvSummary:
    """Export a model-ordered CSV from the final AddBiomechanics outputs."""
    model_path = Path(final_model_path).resolve()
    mot_path = Path(final_mot_path).resolve()
    output_path = Path(output_csv_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segment_paths = discover_segment_mot_paths(mot_path)
    if len(segment_paths) == 1:
        convert_summary = convert_mot_to_model_csv(
            mot_path=mot_path,
            model_path=model_path,
            out_csv_path=output_path,
            missing_fill=math.nan,
            include_time=True,
            include_frame=True,
            add_velocity=True,
            add_acceleration=True,
            filter_mode=filter_mode,
            cutoff_hz=cutoff_hz,
        )
        frames = int(convert_summary.input_rows)
        velocity_source = "numerical_derivative_fallback"
    else:
        # Differentiate and filter each AddBiomechanics segment independently.
        # This prevents a millimetric IK seam jump from becoming an impulsive
        # velocity/acceleration in the exported dynamics input.
        with tempfile.TemporaryDirectory(prefix="segmented-kinematics-") as tmpdir:
            segment_csvs: list[Path] = []
            for segment_index, segment_path in enumerate(segment_paths):
                segment_csv = Path(tmpdir) / f"segment_{segment_index}.csv"
                convert_mot_to_model_csv(
                    mot_path=segment_path,
                    model_path=model_path,
                    out_csv_path=segment_csv,
                    missing_fill=math.nan,
                    include_time=True,
                    include_frame=True,
                    add_velocity=True,
                    add_acceleration=True,
                    filter_mode=filter_mode,
                    cutoff_hz=cutoff_hz,
                )
                segment_csvs.append(segment_csv)
            _, frames = merge_csv_segments(segment_csvs, output_path)
        velocity_source = "segmented_numerical_derivative"
    dof_names = extract_coordinate_names_from_osim(model_path)

    return AddBiomechanicsCsvSummary(
        output_csv_path=output_path,
        model_path=model_path,
        mot_path=mot_path,
        dof_names=tuple(dof_names),
        frames=frames,
        velocity_source=velocity_source,
    )

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


REPARAMETERIZED_COORDINATES = ("shoulder_l_x", "shoulder_l_y")
COORDINATE_SUFFIXES = ("", "_vel", "_acc", "_tau")
CONVERTED_COLUMNS = tuple(
    f"{coordinate}{suffix}"
    for coordinate in REPARAMETERIZED_COORDINATES
    for suffix in COORDINATE_SUFFIXES
)


@dataclass(frozen=True)
class BsmCsvConversionSummary:
    input_path: Path
    output_path: Path
    rows: int
    converted_columns: tuple[str, ...]
    converted_position_ranges: dict[str, tuple[float, float]]


def _target_coordinate_limits(model_path: Path) -> dict[str, tuple[float, float, bool]]:
    """Validate the bsm2 shoulder convention and return its coordinate limits."""
    root = ET.parse(model_path).getroot()
    joint = next(
        (
            item
            for item in root.findall(".//JointSet/objects/CustomJoint")
            if item.attrib.get("name") == "GlenoHumeral_l"
        ),
        None,
    )
    if joint is None:
        raise ValueError(f"GlenoHumeral_l not found in target model: {model_path}")

    rotations = joint.findall("./SpatialTransform/TransformAxis")[:3]
    actual_axes = {
        (axis.findtext("coordinates") or "").strip():
        (axis.findtext("axis") or "").strip()
        for axis in rotations
    }
    expected_axes = {
        "shoulder_l_x": "-1 0 0",
        "shoulder_l_y": "0 -1 0",
        "shoulder_l_z": "0 0 1",
    }
    if actual_axes != expected_axes:
        raise ValueError(
            "Target model does not use the expected bsm2 left-shoulder axes: "
            f"expected {expected_axes}, found {actual_axes}"
        )

    limits: dict[str, tuple[float, float, bool]] = {}
    for coordinate in joint.findall("./coordinates/Coordinate"):
        name = coordinate.attrib.get("name", "")
        if name not in REPARAMETERIZED_COORDINATES:
            continue
        raw_range = (coordinate.findtext("range") or "").split()
        if len(raw_range) != 2:
            raise ValueError(f"Invalid range for {name} in target model: {raw_range}")
        clamped = (coordinate.findtext("clamped") or "false").strip().lower() == "true"
        limits[name] = (float(raw_range[0]), float(raw_range[1]), clamped)

    missing = sorted(set(REPARAMETERIZED_COORDINATES) - set(limits))
    if missing:
        raise ValueError(f"Target model is missing coordinates: {', '.join(missing)}")
    return limits


def _negate_numeric_cell(raw: str, *, column: str, row_number: int) -> tuple[str, float]:
    """Negate a numeric CSV cell while preserving its decimal representation."""
    token = raw.strip()
    if not token:
        raise ValueError(f"Empty value in column {column!r}, CSV row {row_number}")
    try:
        value = float(token)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value {raw!r} in column {column!r}, CSV row {row_number}"
        ) from exc

    converted = -value
    # A NaN has no meaningful sign. Keeping it unchanged also preserves its spelling.
    if math.isnan(value) or value == 0.0:
        return raw, converted

    leading = raw[: len(raw) - len(raw.lstrip())]
    trailing = raw[len(raw.rstrip()) :]
    if token.startswith("-"):
        token = token[1:]
    elif token.startswith("+"):
        token = "-" + token[1:]
    else:
        token = "-" + token
    return leading + token + trailing, converted


def convert_bsm_csv_to_bsm2(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_model_path: str | Path,
    overwrite: bool = False,
    allow_out_of_range: bool = False,
) -> BsmCsvConversionSummary:
    """
    Reparameterize a stage-II CSV from bsm.osim to bsm2.osim.

    The physical pose and generalized power are preserved by negating position,
    velocity, acceleration, and generalized force for shoulder_l_x/y. All other
    cells are copied unchanged.
    """
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    target_model = Path(target_model_path).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}")
    if not target_model.is_file():
        raise FileNotFoundError(f"Target model not found: {target_model}")
    if source == destination:
        raise ValueError(
            "Input and output must differ; in-place conversion is intentionally disabled"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite): {destination}")

    limits = _target_coordinate_limits(target_model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    row_count = 0
    observed: dict[str, list[float]] = {
        coordinate: [math.inf, -math.inf]
        for coordinate in REPARAMETERIZED_COORDINATES
    }

    try:
        with source.open("r", newline="", encoding="utf-8") as input_handle:
            reader = csv.reader(input_handle)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Input CSV is empty: {source}") from exc

            duplicates = sorted({name for name in header if header.count(name) > 1})
            if duplicates:
                raise ValueError(f"Duplicate CSV columns: {', '.join(duplicates)}")
            missing = [column for column in CONVERTED_COLUMNS if column not in header]
            if missing:
                raise ValueError(
                    "CSV is not a complete stage-II export; missing columns: "
                    + ", ".join(missing)
                )
            column_indices = {column: header.index(column) for column in CONVERTED_COLUMNS}

            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_handle:
                temp_path = Path(output_handle.name)
                writer = csv.writer(output_handle)
                writer.writerow(header)
                for row_number, row in enumerate(reader, start=2):
                    if len(row) != len(header):
                        raise ValueError(
                            f"CSV row {row_number} has {len(row)} cells; expected {len(header)}"
                        )
                    for column, index in column_indices.items():
                        row[index], converted = _negate_numeric_cell(
                            row[index], column=column, row_number=row_number
                        )
                        if column in observed and math.isfinite(converted):
                            observed[column][0] = min(observed[column][0], converted)
                            observed[column][1] = max(observed[column][1], converted)
                    writer.writerow(row)
                    row_count += 1

        if row_count == 0:
            raise ValueError(f"Input CSV has a header but no data rows: {source}")

        converted_ranges: dict[str, tuple[float, float]] = {}
        violations: list[str] = []
        for coordinate, values in observed.items():
            if not math.isfinite(values[0]):
                continue
            observed_range = (values[0], values[1])
            converted_ranges[coordinate] = observed_range
            lower, upper, clamped = limits[coordinate]
            if clamped and (
                observed_range[0] < lower - 1e-12 or observed_range[1] > upper + 1e-12
            ):
                violations.append(
                    f"{coordinate} converted range {observed_range} exceeds clamped "
                    f"target range ({lower}, {upper})"
                )
        if violations and not allow_out_of_range:
            raise ValueError(
                "; ".join(violations)
                + ". Use --allow-out-of-range only if downstream code does not clamp coordinates."
            )

        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during conversion: {destination}")
        temp_path.chmod(source.stat().st_mode & 0o777)
        os.replace(temp_path, destination)
        temp_path = None
        return BsmCsvConversionSummary(
            input_path=source,
            output_path=destination,
            rows=row_count,
            converted_columns=CONVERTED_COLUMNS,
            converted_position_ranges=converted_ranges,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

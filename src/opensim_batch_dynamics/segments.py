from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Sequence


DEFAULT_SEGMENT_EDGE_TRIM_S = 0.2


def _segment_number(path: Path) -> int:
    match = re.search(r"_segment_(\d+)_ik\.mot$", path.name)
    return int(match.group(1)) if match else -1


def discover_segment_mot_paths(final_mot_path: str | Path) -> tuple[Path, ...]:
    """Return numerically ordered AddBiomechanics IK segments for a merged MOT."""
    final_path = Path(final_mot_path).resolve()
    if re.search(r"_segment_\d+_ik\.mot$", final_path.name):
        return (final_path,)
    stem = final_path.stem
    base = stem[:-3] if stem.endswith("_ik") else stem
    candidates = sorted(
        final_path.parent.glob(f"{base}_segment_*_ik.mot"),
        key=_segment_number,
    )
    return tuple(path.resolve() for path in candidates) if candidates else (final_path,)


def keep_segment_time(
    time_s: float,
    start_s: float,
    end_s: float,
    segment_index: int,
    segment_count: int,
    edge_trim_s: float = DEFAULT_SEGMENT_EDGE_TRIM_S,
) -> bool:
    """Keep samples outside internal filter-edge windows."""
    tolerance = 1e-9
    if segment_index > 0 and time_s < start_s + edge_trim_s - tolerance:
        return False
    if segment_index + 1 < segment_count and time_s > end_s - edge_trim_s + tolerance:
        return False
    return True


def _effective_edge_trim(
    start_s: float,
    end_s: float,
    segment_index: int,
    segment_count: int,
    edge_trim_s: float,
) -> float:
    """Return a trim that cannot erase a very short segment.

    AddBiomechanics can emit a final (or internal) segment containing only a
    few samples after a discontinuity.  Applying the normal 200 ms filter
    margin to both sides of such a segment would remove every sample and
    abort an otherwise usable trial.  In that case retain the short segment
    unchanged; its timestamps still provide the correct uniform continuation
    of the preceding segment, while the usual margin remains in force for
    normal-length segments.
    """
    if edge_trim_s <= 0.0:
        return edge_trim_s
    trimmed_sides = int(segment_index > 0) + int(segment_index + 1 < segment_count)
    if trimmed_sides and (end_s - start_s) <= trimmed_sides * edge_trim_s + 1e-9:
        return 0.0
    return edge_trim_s


def merge_csv_segments(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    edge_trim_s: float = DEFAULT_SEGMENT_EDGE_TRIM_S,
) -> tuple[Path, int]:
    """Merge equal-schema segment CSVs after trimming internal segment edges."""
    paths = [Path(path).resolve() for path in input_paths]
    if not paths:
        raise ValueError("At least one segment CSV is required.")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_header: list[str] | None = None
    merged_rows: list[dict[str, str]] = []
    for segment_index, path in enumerate(paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or ())
            rows = list(reader)
        if not header or "time" not in header or not rows:
            raise ValueError(f"Invalid segment CSV: {path}")
        if expected_header is None:
            expected_header = header
        elif header != expected_header:
            raise ValueError(f"Segment CSV headers do not match: {path}")
        times = [float(row["time"]) for row in rows]
        start_s, end_s = times[0], times[-1]
        effective_edge_trim = _effective_edge_trim(
            start_s,
            end_s,
            segment_index,
            len(paths),
            edge_trim_s,
        )
        kept = [
            row
            for row, time_s in zip(rows, times)
            if keep_segment_time(
                time_s,
                start_s,
                end_s,
                segment_index,
                len(paths),
                effective_edge_trim,
            )
        ]
        if not kept:
            raise ValueError(
                f"Segment edge trim removed every row from {path}; "
                f"reduce edge_trim_s={edge_trim_s}."
            )
        merged_rows.extend(kept)

    assert expected_header is not None
    if "frame" in expected_header:
        for frame, row in enumerate(merged_rows):
            row["frame"] = str(frame)
    times = [float(row["time"]) for row in merged_rows]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Merged segment CSV timestamps are not strictly increasing.")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=expected_header)
        writer.writeheader()
        writer.writerows(merged_rows)
    return output, len(merged_rows)


def merge_storage_segments(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    edge_trim_s: float = DEFAULT_SEGMENT_EDGE_TRIM_S,
) -> tuple[Path, int]:
    """Merge OpenSim MOT/STO tables after trimming internal filter edges."""
    from .mot_to_csv import parse_mot

    paths = [Path(path).resolve() for path in input_paths]
    if not paths:
        raise ValueError("At least one segment storage file is required.")
    labels: list[str] | None = None
    merged: list[list[float]] = []
    for segment_index, path in enumerate(paths):
        current_labels, rows = parse_mot(path)
        if labels is None:
            labels = current_labels
        elif current_labels != labels:
            raise ValueError(f"Segment storage labels do not match: {path}")
        time_idx = current_labels.index("time")
        start_s, end_s = rows[0][time_idx], rows[-1][time_idx]
        effective_edge_trim = _effective_edge_trim(
            start_s,
            end_s,
            segment_index,
            len(paths),
            edge_trim_s,
        )
        kept = [
            row
            for row in rows
            if keep_segment_time(
                row[time_idx],
                start_s,
                end_s,
                segment_index,
                len(paths),
                effective_edge_trim,
            )
        ]
        if not kept:
            raise ValueError(f"Segment edge trim removed every row from {path}.")
        merged.extend(kept)

    assert labels is not None
    time_idx = labels.index("time")
    times = [row[time_idx] for row in merged]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Merged storage timestamps are not strictly increasing.")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(f"name {output.stem}\n")
        handle.write(f"datacolumns {len(labels)}\n")
        handle.write(f"datarows {len(merged)}\n")
        handle.write(f"range {times[0]:.10g} {times[-1]:.10g}\n")
        handle.write("endheader\n")
        handle.write(" ".join(labels) + "\n")
        for row in merged:
            values = (
                f"{value:.12g}" if math.isfinite(value) else "nan"
                for value in row
            )
            handle.write(" ".join(values) + "\n")
    return output, len(merged)

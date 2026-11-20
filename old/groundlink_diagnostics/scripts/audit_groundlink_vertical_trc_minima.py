#!/usr/bin/env python3
"""Report per-trial marker-coordinate minima from saved GroundLink TRCs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

SUBJECTS = ("s001", "s002", "s003", "s004", "s005", "s006")


def _read_trc_minima(path: Path) -> tuple[float, float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.strip().startswith("Frame#")), None)
    if header_index is None or header_index + 2 >= len(lines):
        raise ValueError(f"Invalid TRC header: {path}")
    z_min, y_min = float("inf"), float("inf")
    for line in lines[header_index + 2:]:
        tokens = line.split()
        if len(tokens) < 5:
            continue
        try:
            values = [float(value) for value in tokens]
        except ValueError:
            continue
        y_values = values[3::3]
        z_values = values[4::3]
        finite_y = [value for value in y_values if value == value and abs(value) != float("inf")]
        finite_z = [value for value in z_values if value == value and abs(value) != float("inf")]
        if finite_y:
            y_min = min(y_min, min(finite_y))
        if finite_z:
            z_min = min(z_min, min(finite_z))
    if not (z_min < float("inf") and y_min < float("inf")):
        raise ValueError(f"No finite marker coordinates in {path}")
    return y_min, z_min


def audit(source_root: Path, index_path: Path, output_root: Path) -> dict:
    with index_path.open(newline="", encoding="utf-8") as handle:
        selected = [row for row in csv.DictReader(handle) if row.get("included") == "1"]
    if len(selected) != 206:
        raise ValueError(f"Expected 206 included trials, got {len(selected)}")
    rows = []
    for index, item in enumerate(selected, start=1):
        subject, trial = item["subject"], item["trial"]
        metadata_path = source_root / subject / trial / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ik = Path(metadata["source_paths"]["ik"])
        trial_name = ik.stem.removesuffix("_ik")
        trc = ik.parents[2] / "trials" / trial_name / "markers.trc"
        min_y, min_z = _read_trc_minima(trc)
        rows.append({
            "subject": subject, "trial": trial, "markers_trc": str(trc),
            "minimum_marker_y_m": min_y, "minimum_marker_z_m": min_z,
        })
        print(f"[{index}/206] {subject}/{trial}: min Z={min_z:.6f} m", flush=True)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    csv_path = logs / "axis_min_vertical_by_trial.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    z_values = [row["minimum_marker_z_m"] for row in rows]
    y_values = [row["minimum_marker_y_m"] for row in rows]
    summary = {
        "trials": len(rows),
        "minimum_marker_z_m_median": statistics.median(z_values),
        "minimum_marker_z_m_min": min(z_values),
        "minimum_marker_z_m_max": max(z_values),
        "minimum_marker_y_m_median": statistics.median(y_values),
        "minimum_marker_y_m_min": min(y_values),
        "minimum_marker_y_m_max": max(y_values),
        "all_minimum_y_equal_0_01m_within_1e-8": all(abs(value - 0.01) <= 1e-8 for value in y_values),
        "csv": str(csv_path),
    }
    (logs / "axis_minima_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.source_dir, args.index, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

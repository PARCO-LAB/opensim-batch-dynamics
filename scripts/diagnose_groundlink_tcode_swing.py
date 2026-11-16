#!/usr/bin/env python3
"""Read-only detail for Phase C target differences in zero-Fz windows."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from diagnose_groundlink_cop_offsets import _column, _read_mot

COORDS = (
    "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
)
SIDES = ("l", "r")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnose(source_root: Path, output_root: Path, index_path: Path) -> dict[str, Any]:
    index_rows = _rows(index_path)
    trials = [(row["subject"], row["trial"]) for row in index_rows if row.get("included") == "1"]
    details: list[dict[str, Any]] = []
    total_samples = 0
    for subject, trial in trials:
        source_trial, output_trial = source_root / subject / trial, output_root / subject / trial
        source_rows, output_rows = _rows(source_trial / "torque_targets.csv"), _rows(output_trial / "torque_targets.csv")
        segments = sorted({int(float(row["segment"])) for row in output_rows})
        for segment in segments:
            source_segment = source_trial / "segments" / f"segment_{segment}"
            output_segment = output_trial / "segments" / f"segment_{segment}"
            source_segment_rows = _rows(source_segment / "target_segment.csv")
            output_segment_rows = _rows(output_segment / "target_segment.csv")
            if len(source_segment_rows) != len(output_segment_rows):
                raise ValueError(f"Segment frame count changed for {subject}/{trial}/segment_{segment}")
            for source_row, output_row in zip(source_segment_rows, output_segment_rows):
                if any(source_row[name] != output_row[name] for name in ("frame", "time", "segment")):
                    raise ValueError(f"Segment frame/time/segment changed for {subject}/{trial}/segment_{segment}")
            times = np.asarray([float(row["time"]) for row in output_segment_rows], dtype=float)
            source_mot = next(iter(sorted(source_segment.glob("*_grf_measured.mot"))))
            output_mot = next(iter(sorted(output_segment.glob("*_grf_measured.mot"))))
            source_columns, source_values = _read_mot(source_mot)
            output_columns, output_values = _read_mot(output_mot)
            mot_time = _column(output_columns, output_values, "time")
            fz = {
                side: np.interp(times, mot_time, _column(output_columns, output_values, f"ground_force_calcn_{side}_vz"))
                for side in SIDES
            }
            both_zero = (fz["l"] == 0.0) & (fz["r"] == 0.0)
            stable = both_zero.copy()
            for offset in range(1, 11):
                stable[offset:] &= both_zero[:-offset]
                stable[:-offset] &= both_zero[offset:]
            global_by_time = {
                round(float(row["time"]), 7): index
                for index, row in enumerate(output_rows)
                if int(float(row["segment"])) == segment
            }
            for local_index, segment_row in enumerate(output_segment_rows):
                if not stable[local_index]:
                    continue
                global_index = global_by_time.get(round(float(segment_row["time"]), 7))
                if global_index is None:
                    continue
                total_samples += 1
                for coord in COORDS:
                    before = float(source_rows[global_index][f"{coord}_tau_target"])
                    after = float(output_rows[global_index][f"{coord}_tau_target"])
                    delta = abs(after - before)
                    if delta <= 1e-6:
                        continue
                    first, last = max(0, local_index - 10), min(len(output_segment_rows), local_index + 11)
                    detail: dict[str, Any] = {
                        "subject": subject, "trial": trial, "segment": segment,
                        "frame": output_rows[global_index]["frame"], "time": output_rows[global_index]["time"],
                        "coord": coord, "tau_before_nm": before, "tau_after_nm": after,
                        "delta_nm": delta,
                        "fz_l_n": fz["l"][local_index], "fz_r_n": fz["r"][local_index],
                        "window_fz_l_min_n": float(np.min(fz["l"][first:last])),
                        "window_fz_l_max_n": float(np.max(fz["l"][first:last])),
                        "window_fz_r_min_n": float(np.min(fz["r"][first:last])),
                        "window_fz_r_max_n": float(np.max(fz["r"][first:last])),
                    }
                    for side in SIDES:
                        for suffix in ("vx", "vy", "vz", "px", "py"):
                            name = f"ground_force_calcn_{side}_{suffix}"
                            source_value = np.interp(times[local_index], mot_time, _column(source_columns, source_values, name))
                            output_value = np.interp(times[local_index], mot_time, _column(output_columns, output_values, name))
                            detail[f"{name}_before"] = float(source_value)
                            detail[f"{name}_after"] = float(output_value)
                    details.append(detail)
    details.sort(key=lambda row: float(row["delta_nm"]), reverse=True)
    fields = list(details[0]) if details else ["subject", "trial", "segment", "frame", "time", "coord", "delta_nm"]
    path = output_root / "logs" / "swing_target_diagnostic.csv"
    _write(path, fields, details[:100])
    summary = {"stable_zero_fz_frames": total_samples, "target_values_over_1e-6_nm": len(details), "max_abs_delta_nm": float(details[0]["delta_nm"]) if details else 0.0, "top_differences_csv": str(path)}
    print(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.source_dir, args.output_dir, args.index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

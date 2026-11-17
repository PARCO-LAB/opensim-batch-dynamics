#!/usr/bin/env python3
"""Recompute the GroundLink false-contact gate from measured/estimated MOTs.

The denominator is pooled valid foot-frames in measured flight (measured Fz <=
1 N), using the non-filtered measured GRF MOT. A false contact is estimated
calcn+toes vertical force > 20 N. Alignment is by target segment and time.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path


TARGET_DIRNAME = "torque_targets_12hz_nomuscles_copfix_tcode"
EXPECTED = {
    "s001": 0.245,
    "s002": 0.080,
    "s003": 0.199,
    "s004": 0.124,
    "s005": 0.106,
    "s006": 0.245,
}
MEASURED_FLIGHT_N = 1.0
ESTIMATED_CONTACT_N = 20.0
TIME_TOLERANCE_S = 2e-5


def read_mot(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text().splitlines()
    try:
        endheader = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as exc:
        raise ValueError(f"Missing endheader in {path}") from exc
    if endheader + 1 >= len(lines):
        raise ValueError(f"Missing column names in {path}")
    columns = lines[endheader + 1].split()
    rows = [line.split() for line in lines[endheader + 2:] if line.strip()]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError(f"Column count mismatch in {path}")
    return columns, rows


def find_one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected exactly one {description}, found {len(paths)}: {paths}")
    return paths[0]


def align_indices(source_times: list[float], target_times: list[float], path: Path) -> list[int]:
    if not source_times or any(b <= a for a, b in zip(source_times, source_times[1:])):
        raise ValueError(f"MOT times must be strictly increasing in {path}")
    result: list[int] = []
    for target in target_times:
        right = bisect.bisect_left(source_times, target)
        candidates = [i for i in (right - 1, right) if 0 <= i < len(source_times)]
        best = min(candidates, key=lambda i: abs(source_times[i] - target))
        error = abs(source_times[best] - target)
        if error > TIME_TOLERANCE_S:
            raise ValueError(f"Could not align target time to {path}; error={error:.8g}s")
        result.append(best)
    return result


def target_files(torque_root: Path) -> list[Path]:
    root = torque_root / TARGET_DIRNAME
    files = sorted(root.glob("s*/**/torque_targets.csv"))
    if not files:
        raise FileNotFoundError(f"No torque_targets.csv files under {root}")
    return files


def evaluate_trial(target_path: Path) -> dict[str, object]:
    subject, trial = target_path.parents[1].name, target_path.parent.name
    with target_path.open(newline="") as stream:
        targets = list(csv.DictReader(stream))
    if not targets or not {"time", "segment", "valid"}.issubset(targets[0]):
        raise ValueError(f"Missing target rows/columns in {target_path}")
    valid_targets = [row for row in targets if int(row["valid"]) == 1]
    numerator = denominator = 0

    segment_ids = sorted({int(row["segment"]) for row in valid_targets})
    for segment in segment_ids:
        selected = [row for row in valid_targets if int(row["segment"]) == segment]
        times = [float(row["time"]) for row in selected]
        segment_dir = target_path.parent / "segments" / f"segment_{segment}"
        measured_path = find_one(list(segment_dir.glob("*_grf_measured.mot")), "measured GRF MOT")
        estimated_path = find_one(list((segment_dir / "estimated_id").glob("*_grf_estimated.mot")), "estimated GRF MOT")
        measured_columns, measured_rows = read_mot(measured_path)
        estimated_columns, estimated_rows = read_mot(estimated_path)
        measured_map = {name: index for index, name in enumerate(measured_columns)}
        estimated_map = {name: index for index, name in enumerate(estimated_columns)}
        measured_idx = align_indices([float(row[measured_map["time"]]) for row in measured_rows], times, measured_path)
        estimated_idx = align_indices([float(row[estimated_map["time"]]) for row in estimated_rows], times, estimated_path)

        for side in ("l", "r"):
            measured_col = f"ground_force_calcn_{side}_vz"
            estimated_cols = (
                f"ground_force_calcn_{side}_vz",
                f"ground_force_toes_{side}_vz",
            )
            if measured_col not in measured_map:
                raise ValueError(f"Missing {measured_col} in {measured_path}")
            if any(name not in estimated_map for name in estimated_cols):
                raise ValueError(f"Missing estimated calcn/toes columns in {estimated_path}")
            for measured_i, estimated_i in zip(measured_idx, estimated_idx):
                measured_fz = float(measured_rows[measured_i][measured_map[measured_col]])
                estimated_fz = sum(float(estimated_rows[estimated_i][estimated_map[name]]) for name in estimated_cols)
                if not math.isfinite(measured_fz) or not math.isfinite(estimated_fz):
                    raise ValueError(f"Non-finite force in {target_path}, segment {segment}, side {side}")
                if measured_fz <= MEASURED_FLIGHT_N:
                    denominator += 1
                    numerator += int(estimated_fz > ESTIMATED_CONTACT_N)

    return {
        "subject": subject,
        "trial": trial,
        "valid_frames": len(valid_targets),
        "false_foot_frames": numerator,
        "measured_flight_foot_frames": denominator,
        "false_contact_fraction_pooled_trial": numerator / denominator if denominator else math.nan,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-root", type=Path, required=True, help="GroundLink_torque root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.002)
    args = parser.parse_args()

    trial_rows = [evaluate_trial(path) for path in target_files(args.torque_root)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_out = args.output_dir / "false_contact_gate_trials.csv"
    write_csv(trial_out, trial_rows)

    summary_rows: list[dict[str, object]] = []
    for subject, expected in sorted(EXPECTED.items()):
        subject_trials = [row for row in trial_rows if row["subject"] == subject]
        false_frames = sum(int(row["false_foot_frames"]) for row in subject_trials)
        flight_frames = sum(int(row["measured_flight_foot_frames"]) for row in subject_trials)
        rate = false_frames / flight_frames if flight_frames else math.nan
        trial_rates = [
            float(row["false_contact_fraction_pooled_trial"])
            for row in subject_trials
            if math.isfinite(float(row["false_contact_fraction_pooled_trial"]))
        ]
        mean_trial = sum(trial_rates) / len(trial_rates) if trial_rates else math.nan
        difference = abs(rate - expected)
        summary_rows.append({
            "subject": subject,
            "trials": len(subject_trials),
            "valid_frames": sum(int(row["valid_frames"]) for row in subject_trials),
            "false_foot_frames": false_frames,
            "measured_flight_foot_frames": flight_frames,
            "false_contact_fraction_pooled": rate,
            "mean_trial_fraction_diagnostic": mean_trial,
            "expected_fraction": expected,
            "absolute_difference": difference,
            "gate_pass": difference <= args.tolerance,
        })
    summary_out = args.output_dir / "false_contact_gate.csv"
    write_csv(summary_out, summary_rows)
    metadata = {
        "definition": "valid==1 target rows; align by segment,time; measured flight is non-filtered measured MOT Fz<=1N; false contact is estimated calcn+toes Fz>20N; pool foot-frames over all trials per subject",
        "measured_flight_threshold_n": MEASURED_FLIGHT_N,
        "estimated_contact_threshold_n": ESTIMATED_CONTACT_N,
        "time_tolerance_s": TIME_TOLERANCE_S,
        "expected_by_subject": EXPECTED,
        "gate_tolerance": args.tolerance,
        "n_trials": len(trial_rows),
        "n_subjects": len({row["subject"] for row in trial_rows}),
    }
    (args.output_dir / "false_contact_gate_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for row in summary_rows:
        print(f"{row['subject']} n={row['trials']} pooled={row['false_contact_fraction_pooled']:.6f} "
              f"mean_trial={row['mean_trial_fraction_diagnostic']:.6f} "
              f"expected={row['expected_fraction']:.3f} pass={row['gate_pass']}")
    passed = len(trial_rows) == 206 and len(summary_rows) == 6 and all(row["gate_pass"] for row in summary_rows)
    print(f"trials={len(trial_rows)} gate_pass={passed} output={summary_out}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

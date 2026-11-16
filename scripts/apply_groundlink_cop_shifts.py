#!/usr/bin/env python3
"""Stage the authorized GroundLink B2 CoP shifts without rerunning inverse dynamics.

The script copies one subject at a time from the immutable ``_nomuscles`` input,
translates valid measured CoP x/y tokens in the copied MOT files, and records the
same trial shift in run_metadata.json. Forces, invalid/flight CoP samples, and all
other copied target artifacts are left untouched. Run ``--verify`` after all six
subject tasks complete; it writes the blocking-trial table for the Phase C gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any


SUBJECTS = ("s001", "s002", "s003", "s004", "s005", "s006")
METHOD = "B2_median_CoP_minus_midfoot_Fz_gt_100N"
MOT_TOKEN = re.compile(r"\S+")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def _shifts_from_diagnosis(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    shifts: dict[tuple[str, str], tuple[float, float]] = {}
    for row in _read_rows(path):
        key = (row["subject"], row["trial"])
        if key in shifts:
            raise ValueError(f"Duplicate diagnostic shift for {key[0]}/{key[1]}")
        shift = (
            float(row["candidate_cop_shift_x_m"]),
            float(row["candidate_cop_shift_y_m"]),
        )
        if not all(math.isfinite(value) for value in shift):
            raise ValueError(f"Non-finite diagnostic shift for {key[0]}/{key[1]}")
        shifts[key] = shift
    if len(shifts) != 206:
        raise ValueError(f"Expected 206 trial shifts, found {len(shifts)} in {path}")
    return shifts


def prepare_shift_csv(output_root: Path) -> Path:
    diagnostic = output_root / "logs" / "root_translation_vs_offset.csv"
    shifts = _shifts_from_diagnosis(diagnostic)
    rows = [
        {"subject": subject, "trial": trial, "dx_m": shift[0], "dy_m": shift[1]}
        for (subject, trial), shift in sorted(shifts.items())
    ]
    path = output_root / "logs" / "cop_shift_by_trial.csv"
    _write_rows(path, ["subject", "trial", "dx_m", "dy_m"], rows)
    return path


def _read_shifts(path: Path) -> dict[tuple[str, str], tuple[float, float]]:
    shifts: dict[tuple[str, str], tuple[float, float]] = {}
    for row in _read_rows(path):
        key = (row["subject"], row["trial"])
        shift = (float(row["dx_m"]), float(row["dy_m"]))
        if key in shifts or not all(math.isfinite(value) for value in shift):
            raise ValueError(f"Duplicate or invalid shift for {key[0]}/{key[1]}")
        shifts[key] = shift
    if len(shifts) != 206:
        raise ValueError(f"Expected 206 per-trial shifts, found {len(shifts)}")
    return shifts


def _manifest_trials(source_root: Path, subject: str) -> list[str]:
    rows = _read_rows(source_root / f"execution_manifest_{subject}.csv")
    trials = [row["trial"] for row in rows if row.get("status") == "ok"]
    if len(trials) != len(set(trials)) or not trials:
        raise ValueError(f"Invalid successful-trial manifest for {subject}")
    return trials


def _mot_layout(lines: list[str], path: Path) -> tuple[int, list[str], list[int]]:
    end_header = next(
        (i for i, line in enumerate(lines) if line.strip().lower() == "endheader"), None
    )
    if end_header is None or end_header + 1 >= len(lines):
        raise ValueError(f"Invalid MOT header: {path}")
    columns = lines[end_header + 1].split()
    data_lines = [i for i in range(end_header + 2, len(lines)) if lines[i].strip()]
    return end_header, columns, data_lines


def _shift_mot(path: Path, dx: float, dy: float) -> int:
    if path.is_symlink():
        raise ValueError(f"Measured force MOT must be a regular copied file: {path}")
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    _, columns, data_lines = _mot_layout(lines, path)
    indices: dict[str, int] = {}
    for side in ("l", "r"):
        for suffix in ("vz", "px", "py"):
            name = f"ground_force_calcn_{side}_{suffix}"
            if name not in columns:
                raise ValueError(f"Missing MOT column {name} in {path}")
            indices[name] = columns.index(name)
    changed_points = 0
    for line_index in data_lines:
        line = lines[line_index]
        matches = list(MOT_TOKEN.finditer(line))
        if len(matches) != len(columns):
            raise ValueError(f"MOT row/column mismatch in {path}:{line_index + 1}")
        replacements: list[tuple[int, int, str]] = []
        for side in ("l", "r"):
            fz = float(matches[indices[f"ground_force_calcn_{side}_vz"]].group())
            px_match = matches[indices[f"ground_force_calcn_{side}_px"]]
            py_match = matches[indices[f"ground_force_calcn_{side}_py"]]
            px, py = float(px_match.group()), float(py_match.group())
            # Keep flight, sub-threshold, and invalid-zero CoP samples unchanged.
            if fz <= 20.0 or (px == 0.0 and py == 0.0):
                continue
            replacements.append((px_match.start(), px_match.end(), f"{px - dx:.12f}"))
            replacements.append((py_match.start(), py_match.end(), f"{py - dy:.12f}"))
            changed_points += 1
        for start, end, replacement in sorted(replacements, reverse=True):
            line = line[:start] + replacement + line[end:]
        lines[line_index] = line
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text("".join(lines), encoding="utf-8", newline="")
    os.replace(temporary, path)
    return changed_points


def _write_trial_metadata(
    source_trial: Path, output_trial: Path, final_trial: Path, dx: float, dy: float
) -> None:
    metadata_path = output_trial / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("subject") != source_trial.parent.name or metadata.get("trial") != output_trial.name:
        raise ValueError(f"Trial metadata key mismatch: {metadata_path}")
    metadata["cop_shift_m"] = [float(dx), float(dy)]
    metadata["cop_shift_method"] = METHOD
    metadata["cop_shift_source"] = "logs/cop_shift_by_trial.csv; Fz > 100 N across both feet and all segments"
    metadata["output_csv"] = str(final_trial / "torque_targets.csv")
    partial = metadata_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, metadata_path)


def apply_subject(source_root: Path, output_root: Path, shifts_path: Path, subject: str) -> dict[str, int | str]:
    shifts = _read_shifts(shifts_path)
    trials = _manifest_trials(source_root, subject)
    source_subject = source_root / subject
    destination_subject = output_root / subject
    staged_subject = output_root / f".{subject}.b2.partial"
    if destination_subject.exists():
        metadata_paths = sorted(destination_subject.glob("*/run_metadata.json"))
        if len(metadata_paths) == len(trials):
            complete = True
            for path in metadata_paths:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                key = (subject, path.parent.name)
                complete &= (
                    metadata.get("cop_shift_method") == METHOD
                    and metadata.get("cop_shift_m") == list(shifts[key])
                )
            if complete:
                return {"subject": subject, "trials": len(trials), "status": "already_staged"}
        raise FileExistsError(f"Refusing to overwrite non-B2 candidate subject directory: {destination_subject}")
    if staged_subject.exists():
        shutil.rmtree(staged_subject)
    shutil.copytree(source_subject, staged_subject, symlinks=True, copy_function=shutil.copy2)
    segments = 0
    changed_points = 0
    try:
        for trial in trials:
            key = (subject, trial)
            dx, dy = shifts[key]
            source_trial = source_subject / trial
            output_trial = staged_subject / trial
            if not output_trial.is_dir():
                raise FileNotFoundError(f"Copied trial is missing: {output_trial}")
            measured_files = sorted(output_trial.glob("segments/segment_*/*_grf_measured.mot"))
            if not measured_files:
                raise FileNotFoundError(f"No measured force MOTs under {output_trial}")
            for mot in measured_files:
                changed_points += _shift_mot(mot, dx, dy)
                segments += 1
            _write_trial_metadata(
                source_trial, output_trial, destination_subject / trial, dx, dy
            )
        os.replace(staged_subject, destination_subject)
    except Exception:
        if staged_subject.exists():
            shutil.rmtree(staged_subject)
        raise
    return {
        "subject": subject,
        "trials": len(trials),
        "segments": segments,
        "valid_cop_points_shifted": changed_points,
        "status": "staged",
    }


def _mot_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    _, columns, data_lines = _mot_layout(lines, path)
    rows = [lines[index].split() for index in data_lines]
    return columns, rows


def _verify_mot(source: Path, output: Path, dx: float, dy: float) -> int:
    source_lines = source.read_text(encoding="utf-8", errors="strict").splitlines()
    output_lines = output.read_text(encoding="utf-8", errors="strict").splitlines()
    _, columns, source_data_lines = _mot_layout(source_lines, source)
    _, output_columns, output_data_lines = _mot_layout(output_lines, output)
    if columns != output_columns or len(source_data_lines) != len(output_data_lines):
        raise ValueError(f"MOT schema or row count changed: {source} -> {output}")
    x_indices = {columns.index(f"ground_force_calcn_{side}_px") for side in ("l", "r")}
    y_indices = {columns.index(f"ground_force_calcn_{side}_py") for side in ("l", "r")}
    z_force_indices = {side: columns.index(f"ground_force_calcn_{side}_vz") for side in ("l", "r")}
    side_indices = {
        side: (columns.index(f"ground_force_calcn_{side}_px"), columns.index(f"ground_force_calcn_{side}_py"))
        for side in ("l", "r")
    }
    changed_points = 0
    for src_idx, dst_idx in zip(source_data_lines, output_data_lines):
        src_tokens, dst_tokens = source_lines[src_idx].split(), output_lines[dst_idx].split()
        if len(src_tokens) != len(columns) or len(dst_tokens) != len(columns):
            raise ValueError(f"MOT row/column mismatch in {source} or {output}")
        if len(src_tokens) != len(dst_tokens):
            raise ValueError(f"MOT row width changed: {source} -> {output}")
        for side in ("l", "r"):
            fz = float(src_tokens[z_force_indices[side]])
            x_i, y_i = side_indices[side]
            old_x, old_y = float(src_tokens[x_i]), float(src_tokens[y_i])
            valid = fz > 20.0 and not (old_x == 0.0 and old_y == 0.0)
            if valid:
                if abs(float(dst_tokens[x_i]) - (old_x - dx)) > 2e-12:
                    raise ValueError(f"CoP x shift mismatch: {output}:{dst_idx + 1}")
                if abs(float(dst_tokens[y_i]) - (old_y - dy)) > 2e-12:
                    raise ValueError(f"CoP y shift mismatch: {output}:{dst_idx + 1}")
                changed_points += 1
            elif dst_tokens[x_i] != src_tokens[x_i] or dst_tokens[y_i] != src_tokens[y_i]:
                raise ValueError(f"Invalid or unloaded CoP was changed: {output}:{dst_idx + 1}")
        for token_index, (source_token, output_token) in enumerate(zip(src_tokens, dst_tokens)):
            if token_index in x_indices or token_index in y_indices:
                continue
            if source_token != output_token:
                raise ValueError(f"Non-CoP MOT value changed: {output}:{dst_idx + 1}")
    return changed_points


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_and_write_after(source_root: Path, output_root: Path, shifts_path: Path) -> dict[str, Any]:
    shifts = _read_shifts(shifts_path)
    verified_trials = 0
    verified_segments = 0
    shifted_points = 0
    for subject in SUBJECTS:
        for trial in _manifest_trials(source_root, subject):
            key = (subject, trial)
            dx, dy = shifts[key]
            source_trial, output_trial = source_root / subject / trial, output_root / subject / trial
            metadata = json.loads((output_trial / "run_metadata.json").read_text(encoding="utf-8"))
            if metadata.get("cop_shift_m") != [dx, dy] or metadata.get("cop_shift_method") != METHOD:
                raise ValueError(f"Incorrect copied CoP shift metadata: {output_trial}")
            source_targets, output_targets = source_trial / "torque_targets.csv", output_trial / "torque_targets.csv"
            if _sha256(source_targets) != _sha256(output_targets):
                raise ValueError(f"Target table changed before Phase C: {output_trial}")
            source_mots = sorted(source_trial.glob("segments/segment_*/*_grf_measured.mot"))
            output_mots = sorted(output_trial.glob("segments/segment_*/*_grf_measured.mot"))
            if len(source_mots) != len(output_mots) or not source_mots:
                raise ValueError(f"Measured MOT set differs: {output_trial}")
            for source_mot, output_mot in zip(source_mots, output_mots):
                if source_mot.relative_to(source_trial) != output_mot.relative_to(output_trial):
                    raise ValueError(f"Measured MOT ordering differs: {output_trial}")
                shifted_points += _verify_mot(source_mot, output_mot, dx, dy)
                verified_segments += 1
            verified_trials += 1

    diagnosis_path = output_root / "logs" / "root_translation_vs_offset.csv"
    diagnostic_rows = _read_rows(diagnosis_path)
    detail_rows = _read_rows(output_root / "logs" / "cop_offset_before.csv")
    detail_by_trial: dict[tuple[str, str], list[dict[str, str]]] = {}
    for detail in detail_rows:
        detail_by_trial.setdefault((detail["subject"], detail["trial"]), []).append(detail)
    failures: list[dict[str, Any]] = []
    for row in diagnostic_rows:
        subject, trial = row["subject"], row["trial"]
        median_distance = float(row["candidate_median_cop_to_own_foot_cm"])
        p90_distance = float(row["candidate_p90_cop_to_own_foot_cm"])
        own_closer = float(row["candidate_own_foot_closer_percent"])
        distance_pass = median_distance < 5.0 and p90_distance < 10.0
        offset_x_cm = (float(row["median_offset_x_m"]) - float(row["candidate_cop_shift_x_m"])) * 100.0
        offset_y_cm = (float(row["median_offset_y_m"]) - float(row["candidate_cop_shift_y_m"])) * 100.0
        offset_pass = abs(offset_x_cm) < 3.0 and abs(offset_y_cm) < 3.0
        side_pass = own_closer > 95.0
        if not (distance_pass and offset_pass and side_pass):
            details = detail_by_trial.get((subject, trial), [])
            distance_rows = [
                row for row in details
                if float(row["candidate_cop_to_own_foot_cm"]) >= 5.0
            ]
            side_rows = [
                row for row in details
                if float(row["candidate_own_foot_closer_percent"]) <= 95.0
            ]
            failures.append({
                "subject": subject,
                "trial": trial,
                "cop_shift_dx_m": row["candidate_cop_shift_x_m"],
                "cop_shift_dy_m": row["candidate_cop_shift_y_m"],
                "median_cop_to_own_foot_cm": f"{median_distance:.6f}",
                "p90_cop_to_own_foot_cm": f"{p90_distance:.6f}",
                "median_residual_offset_x_cm": f"{offset_x_cm:.9f}",
                "median_residual_offset_y_cm": f"{offset_y_cm:.9f}",
                "own_foot_closer_percent": f"{own_closer:.6f}",
                "distance_criterion_pass": int(distance_pass),
                "offset_criterion_pass": int(offset_pass),
                "side_criterion_pass": int(side_pass),
                "segment_sides_median_distance_ge_5cm": ";".join(
                    f"segment_{row['segment']}/{row['side']}"
                    for row in distance_rows
                ),
                "segment_sides_own_closer_le_95_percent": ";".join(
                    f"segment_{row['segment']}/{row['side']}"
                    for row in side_rows
                ),
                "failed_criteria": ";".join(
                    name for name, passed in (
                        ("distance", distance_pass), ("offset", offset_pass), ("side", side_pass)
                    ) if not passed
                ),
            })
    after_path = output_root / "logs" / "cop_offset_after.csv"
    fields = [
        "subject", "trial", "cop_shift_dx_m", "cop_shift_dy_m",
        "median_cop_to_own_foot_cm", "p90_cop_to_own_foot_cm",
        "median_residual_offset_x_cm", "median_residual_offset_y_cm",
        "own_foot_closer_percent", "distance_criterion_pass",
        "offset_criterion_pass", "side_criterion_pass",
        "segment_sides_median_distance_ge_5cm",
        "segment_sides_own_closer_le_95_percent", "failed_criteria",
    ]
    _write_rows(after_path, fields, failures)
    return {
        "trials_verified": verified_trials,
        "segments_verified": verified_segments,
        "valid_cop_points_shifted": shifted_points,
        "criteria_pass_trials": verified_trials - len(failures),
        "criteria_fail_trials": len(failures),
        "failure_report": str(after_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shift-csv", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="Write logs/cop_shift_by_trial.csv from diagnosis.")
    mode.add_argument("--subject", choices=SUBJECTS, help="Stage one subject for the B2 candidate.")
    mode.add_argument("--verify", action="store_true", help="Verify staged MOTs and write cop_offset_after.csv.")
    args = parser.parse_args()
    source_root = args.source_dir.resolve()
    output_root = args.output_dir.resolve()
    if args.prepare:
        print(json.dumps({"shift_csv": str(prepare_shift_csv(output_root))}, indent=2))
        return 0
    shifts_path = args.shift_csv.resolve() if args.shift_csv else output_root / "logs" / "cop_shift_by_trial.csv"
    if args.subject:
        result = apply_subject(source_root, output_root, shifts_path, args.subject)
    else:
        result = verify_and_write_after(source_root, output_root, shifts_path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

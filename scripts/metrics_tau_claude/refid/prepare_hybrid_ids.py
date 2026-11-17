#!/usr/bin/env python3
"""Create C/V/H/P hybrid GRF MOTs and an ID manifest for GroundLink.

Measured forces are used only to build evaluation inputs. The estimator itself
is not changed by this script. Existing source files remain read-only; every
generated MOT/XML and inverse-dynamics result is written below a new study root.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


VARIANTS = ("C", "V", "H", "P", "CV", "CVH", "CVHP", "PH", "PHV", "PHVC", "ALL")
COMPONENTS = {
    "C": ("C",), "V": ("V",), "H": ("H",), "P": ("P",),
    "CV": ("C", "V"), "CVH": ("C", "V", "H"), "CVHP": ("C", "V", "H", "P"),
    "PH": ("P", "H"), "PHV": ("P", "H", "V"), "PHVC": ("P", "H", "V", "C"),
}
TIME_TOLERANCE_S = 2e-5
MEASURED_CONTACT_N = 1.0


def read_mot(path: Path) -> tuple[list[str], list[list[float]], list[str]]:
    lines = path.read_text().splitlines()
    try:
        endheader = next(i for i, line in enumerate(lines) if line.strip().lower() == "endheader")
    except StopIteration as exc:
        raise ValueError(f"Missing endheader in {path}") from exc
    columns = lines[endheader + 1].split()
    raw_rows = [line.split() for line in lines[endheader + 2:] if line.strip()]
    if any(len(row) != len(columns) for row in raw_rows):
        raise ValueError(f"MOT column mismatch in {path}")
    rows = [[float(value) for value in row] for row in raw_rows]
    return columns, rows, lines[:endheader + 1]


def find_one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected exactly one {description}; found {len(paths)}: {paths}")
    return paths[0]


def align_indices(source_times: list[float], target_times: list[float], path: Path) -> list[int]:
    indices: list[int] = []
    for target in target_times:
        right = bisect.bisect_left(source_times, target)
        choices = [i for i in (right - 1, right) if 0 <= i < len(source_times)]
        index = min(choices, key=lambda i: abs(source_times[i] - target))
        if abs(source_times[index] - target) > TIME_TOLERANCE_S:
            raise ValueError(f"Time alignment failed for {path} at {target}")
        indices.append(index)
    return indices


def load_target_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"segment", "time", "valid"}.issubset(rows[0]):
        raise ValueError(f"Missing target columns in {path}")
    return rows


def set_force(row: list[float], col_map: dict[str, int], body: str, vector: list[float]) -> None:
    for axis, value in zip("xyz", vector):
        row[col_map[f"ground_force_{body}_v{axis}"]] = value


def get_force(row: list[float], col_map: dict[str, int], body: str) -> list[float]:
    return [row[col_map[f"ground_force_{body}_v{axis}"]] for axis in "xyz"]


def set_point(row: list[float], col_map: dict[str, int], body: str, point: list[float]) -> None:
    for axis, value in zip("xyz", point):
        row[col_map[f"ground_force_{body}_p{axis}"]] = value


def body_shares(row: list[float], col_map: dict[str, int], side: str) -> tuple[float, float]:
    calcn, toes = f"calcn_{side}", f"toes_{side}"
    vertical = [max(0.0, get_force(row, col_map, body)[2]) for body in (calcn, toes)]
    total = sum(vertical)
    if total > 1e-9:
        return vertical[0] / total, vertical[1] / total
    magnitudes = [math.sqrt(sum(x * x for x in get_force(row, col_map, body))) for body in (calcn, toes)]
    total = sum(magnitudes)
    return (magnitudes[0] / total, magnitudes[1] / total) if total > 1e-9 else (1.0, 0.0)


def make_variant_rows(
    estimated_columns: list[str],
    estimated_rows: list[list[float]],
    measured_columns: list[str],
    measured_rows: list[list[float]],
    measured_indices: list[int],
    variant: str,
) -> list[list[float]]:
    output = [row[:] for row in estimated_rows]
    est = {name: index for index, name in enumerate(estimated_columns)}
    meas = {name: index for index, name in enumerate(measured_columns)}
    feet = {"l": ("calcn_l", "toes_l"), "r": ("calcn_r", "toes_r")}
    for row_index, (row, measured_index) in enumerate(zip(output, measured_indices)):
        measured = measured_rows[measured_index]
        measured_fz = {
            side: measured[meas[f"ground_force_calcn_{side}_vz"]]
            for side in "lr"
        }
        measured_contact = {side: measured_fz[side] > MEASURED_CONTACT_N for side in "lr"}
        if variant == "C":
            total_force = [
                sum(get_force(row, est, body)[axis] for side in "lr" for body in feet[side])
                for axis in range(3)
            ]
            active = [side for side in "lr" if measured_contact[side]]
            if not active:
                for side in "lr":
                    for body in feet[side]:
                        set_force(row, est, body, [0.0, 0.0, 0.0])
            elif len(active) == 1:
                active_side = active[0]
                shares = body_shares(row, est, active_side)
                for side in "lr":
                    bodies = feet[side]
                    if side != active_side:
                        for body in bodies:
                            set_force(row, est, body, [0.0, 0.0, 0.0])
                    else:
                        for body, share in zip(bodies, shares):
                            set_force(row, est, body, [share * value for value in total_force])
        elif variant == "V":
            for side in "lr":
                shares = body_shares(row, est, side)
                for body, share in zip(feet[side], shares):
                    force = get_force(row, est, body)
                    force[2] = measured_fz[side] * share
                    set_force(row, est, body, force)
        elif variant == "H":
            for side in "lr":
                shares = body_shares(row, est, side)
                measured_h = [
                    measured[meas[f"ground_force_calcn_{side}_v{axis}"]]
                    for axis in "xy"
                ]
                for body, share in zip(feet[side], shares):
                    force = get_force(row, est, body)
                    force[0], force[1] = measured_h[0] * share, measured_h[1] * share
                    set_force(row, est, body, force)
        elif variant == "P":
            for side in "lr":
                if measured_contact[side]:
                    measured_cop = [
                        measured[meas[f"ground_force_calcn_{side}_p{axis}"]]
                        for axis in "xyz"
                    ]
                    for body in feet[side]:
                        set_point(row, est, body, measured_cop)
    return output


def write_mot(path: Path, columns: list[str], rows: list[list[float]], name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [row[0] for row in rows]
    with path.open("w") as handle:
        handle.write(f"name {name}\n")
        handle.write(f"datacolumns {len(columns)}\n")
        handle.write(f"datarows {len(rows)}\n")
        handle.write(f"range {times[0]:.8f} {times[-1]:.8f}\nendheader\n")
        handle.write(" ".join(columns) + "\n")
        for row in rows:
            handle.write(" ".join(f"{value:.10f}" for value in row) + "\n")


def make_external_xml(source: Path, destination: Path, mot_name: str, variant: str) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    external = root.find("ExternalLoads")
    if external is None:
        raise ValueError(f"No ExternalLoads in {source}")
    external.set("name", f"{external.get('name', 'GroundLink')}_{variant}")
    datafile = external.find("datafile")
    if datafile is None:
        raise ValueError(f"No datafile entry in {source}")
    datafile.text = mot_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def prepare_segment(target_path: Path, study_root: Path, manifest: list[dict[str, str]]) -> int:
    trial_dir = target_path.parent
    subject, trial = trial_dir.parent.name, trial_dir.name
    target_rows = load_target_rows(target_path)
    segment_ids = sorted({int(row["segment"]) for row in target_rows})
    for segment in segment_ids:
        segment_dir = trial_dir / "segments" / f"segment_{segment}"
        estimated_dir = segment_dir / "estimated_id"
        estimated_mot = find_one(list(estimated_dir.glob("*_grf_estimated.mot")), "estimated GRF MOT")
        measured_mot = find_one(list(segment_dir.glob("*_grf_measured.mot")), "measured GRF MOT")
        estimated_xml = find_one(list(estimated_dir.glob("*_external_forces_estimated.xml")), "estimated ExternalLoads XML")
        estimated_setup = find_one(list(estimated_dir.glob("Setup_ID_estimatedGRF_*.xml")), "estimated ID setup")
        measured_xml = find_one(list(segment_dir.glob("*_external_forces_measured.xml")), "measured ExternalLoads XML")
        measured_setup = find_one(list(segment_dir.glob("Setup_ID_measuredGRF_*.xml")), "measured ID setup")
        est_columns, est_rows, _ = read_mot(estimated_mot)
        meas_columns, meas_rows, _ = read_mot(measured_mot)
        est_map = {name: i for i, name in enumerate(est_columns)}
        meas_map = {name: i for i, name in enumerate(meas_columns)}
        for side in "lr":
            for body in (f"calcn_{side}", f"toes_{side}"):
                for kind in ("v", "p", "m"):
                    for axis in "xyz":
                        if f"ground_force_{body}_{kind}{axis}" not in est_map:
                            raise ValueError(f"Missing ground_force_{body}_{kind}{axis} in {estimated_mot}")
            for kind in ("v", "p"):
                for axis in "xyz":
                    if f"ground_force_calcn_{side}_{kind}{axis}" not in meas_map:
                        raise ValueError(f"Missing measured calcn column in {measured_mot}")
        est_times = [row[est_map["time"]] for row in est_rows]
        meas_times = [row[meas_map["time"]] for row in meas_rows]
        if any(b <= a for a, b in zip(est_times, est_times[1:])):
            raise ValueError(f"Estimated MOT times are not increasing: {estimated_mot}")
        measured_indices = align_indices(meas_times, est_times, measured_mot)

        for variant in VARIANTS:
            output_dir = study_root / "phase_a" / "id_outputs" / variant / subject / trial / f"segment_{segment}"
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"ID output is not empty: {output_dir}")
            if variant == "ALL":
                setup_path, external_path = measured_setup, measured_xml
            else:
                transformed = [row[:] for row in est_rows]
                for component in COMPONENTS[variant]:
                    transformed = make_variant_rows(
                        est_columns, transformed, meas_columns, meas_rows, measured_indices, component
                    )
                mot_name = f"{subject}_{trial}_segment_{segment}_grf_{variant.lower()}.mot"
                hybrid_mot = study_root / "phase_a" / "hybrid_loads" / subject / trial / f"segment_{segment}" / mot_name
                write_mot(hybrid_mot, est_columns, transformed, mot_name.removesuffix(".mot"))
                external_path = hybrid_mot.with_name(mot_name.replace(".mot", "_external.xml"))
                make_external_xml(estimated_xml, external_path, mot_name, variant)
                setup_path = estimated_setup
            id_name = f"id_{variant.lower()}.sto"
            target_sto = next(iter(segment_dir.glob("*_id_measuredGRF.sto")), None)
            estimated_sto = next(iter(estimated_dir.glob("*_id_estimatedGRF.sto")), None)
            if target_sto is None:
                raise FileNotFoundError(f"Missing measured ID target under {segment_dir}")
            if estimated_sto is None:
                raise FileNotFoundError(f"Missing estimated ID baseline under {estimated_dir}")
            manifest.append({
                "variant": variant,
                "subject": subject,
                "trial": trial,
                "segment": str(segment),
                "setup": str(setup_path.resolve()),
                "external_loads_file": str(external_path.resolve()),
                "output_dir": str(output_dir.resolve()),
                "output_name": id_name,
                "forces_to_exclude": "Actuators",
                "target_sto": str(target_sto.resolve()),
                "estimated_sto": str(estimated_sto.resolve()),
                "target_csv": str(target_path.resolve()),
                "measured_mot": str(measured_mot.resolve()),
                "estimated_mot": str(estimated_mot.resolve()),
            })
    return len(segment_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torque-root", type=Path, required=True, help="GroundLink_torque root")
    parser.add_argument("--study-root", type=Path, required=True, help="New pipeline_tuning study output")
    parser.add_argument("--trial-limit", type=int, default=0, help="Optional preflight limit; zero processes all trials")
    args = parser.parse_args()
    study_root = args.study_root.resolve()
    manifests_dir = study_root / "phase_a"
    manifest_path = manifests_dir / "id_manifest.csv"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace existing manifest: {manifest_path}")
    target_paths = sorted((args.torque_root / "torque_targets_12hz_nomuscles_copfix_tcode").glob("s*/**/torque_targets.csv"))
    if args.trial_limit > 0:
        target_paths = target_paths[:args.trial_limit]
    if not target_paths:
        raise FileNotFoundError("No torque target trials found")
    rows: list[dict[str, str]] = []
    segment_count = 0
    for target_path in target_paths:
        segment_count += prepare_segment(target_path, study_root, rows)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "trials": len(target_paths),
        "segments": segment_count,
        "variants": list(VARIANTS),
        "id_rows": len(rows),
        "measured_contact_threshold_n": MEASURED_CONTACT_N,
        "time_alignment_tolerance_s": TIME_TOLERANCE_S,
        "geometry_copied": False,
        "inference_uses_measured_forces": False,
        "component_notes": {
            "C": "Use non-filtered measured MOT contact state (Fz>1N), zero the airborne foot; transfer the total estimated vector to the remaining foot, or zero both in bilateral flight.",
            "V": "Replace each foot's total vertical force from the measured MOT; retain estimated horizontal forces and points.",
            "H": "Replace each foot's horizontal force from the measured MOT; retain estimated vertical forces and points.",
            "P": "Replace both application points for a measured-contact foot (Fz>1N) with the T_code corrected measured CoP; retain estimated forces.",
            "CV/CVH/CVHP": "Cumulative C→V→H→P order; each prefix is a separate ID run.",
            "PH/PHV/PHVC": "Cumulative P→H→V→C order; each prefix is a separate ID run.",
            "ALL": "Use the existing measured-GRF setup and ExternalLoads files from the target cache.",
        },
    }
    (manifests_dir / "id_manifest_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"trials={len(target_paths)} segments={segment_count} rows={len(rows)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

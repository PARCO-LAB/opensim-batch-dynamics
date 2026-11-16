#!/usr/bin/env python3
"""Audit a full GroundLink torque-target cohort against its source labels and ID setups."""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from aggregate_groundlink_torque_targets import EXPECTED_COUNTS, aggregate

COORDINATES = (
    "hip_flexion_l", "knee_angle_l", "ankle_angle_l",
    "hip_flexion_r", "knee_angle_r", "ankle_angle_r",
)
PATH_TAGS = {
    "model_file", "coordinates_file", "external_loads_file",
    "results_directory", "output_gen_force_file",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _tool_signature(element: ET.Element, protocol: str):
    mutable_tags = PATH_TAGS | (
        {"lowpass_cutoff_frequency_for_coordinates"}
        if protocol == "explicit_order4" else set()
    )
    text = "<path>" if element.tag in mutable_tags else (element.text or "").strip()
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        text,
        tuple(_tool_signature(child, protocol) for child in element),
    )


def _resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def validate(output_root: Path, protocol: str = "original_id_setup") -> dict:
    output_root = output_root.expanduser().resolve()
    manifest_path = aggregate(output_root, protocol)
    _, records = _rows(manifest_path)
    assert len(records) == 206, len(records)
    actual_counts = {subject: sum(row["subject"] == subject for row in records) for subject in EXPECTED_COUNTS}
    assert actual_counts == EXPECTED_COUNTS, actual_counts

    expected_columns = {
        "time", "segment", "valid",
        *[f"{name}_tau_target" for name in COORDINATES],
        *[f"{name}_tau_hat" for name in COORDINATES],
    }
    total_frames = 0
    total_valid = 0
    setup_count = 0
    source_file_count = 0
    max_label_time_error = 0.0
    max_tau_hat_error = 0.0
    max_q_copy_error = 0.0
    missing_source_meshes: set[str] = set()
    for record in records:
        assert record["protocol"] == protocol and record["status"] == "ok", record
        trial_root = output_root / record["subject"] / record["trial"]
        output_csv = trial_root / "torque_targets.csv"
        metadata_path = trial_root / "run_metadata.json"
        assert _sha256(output_csv) == record["output_sha256"], output_csv
        assert _sha256(metadata_path) == record["metadata_sha256"], metadata_path
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["protocol"] == protocol
        assert metadata["force_threshold_n"] == 20.0
        assert metadata["frames"] == int(record["frames"])
        fields, output_rows = _rows(output_csv)
        assert expected_columns.issubset(fields), (output_csv, expected_columns - set(fields))
        assert len(output_rows) == int(record["frames"]), (output_csv, len(output_rows))

        source_details = metadata["source_details"]
        for key in ("h1_index", "h1_labels", "measured_force", "pipeline_csv", "model", "ik"):
            source = source_details[key]
            source_path = Path(source["path"])
            assert source_path.is_file(), (key, source_path)
            assert _sha256(source_path) == source["sha256"], (key, source_path)
            source_file_count += 1
        for source in source_details["original_id_setups"]:
            source_path = Path(source["path"])
            assert source_path.is_file() and _sha256(source_path) == source["sha256"], source_path
            source_file_count += 1

        label_path = Path(metadata["source_details"]["h1_labels"]["path"])
        label_fields, label_rows = _rows(label_path)
        assert {"time", "segment", "valid"}.issubset(label_fields), label_path
        assert len(label_rows) == len(output_rows), (label_path, output_csv)
        for output_row, label_row in zip(output_rows, label_rows):
            time_error = abs(float(output_row["time"]) - float(label_row["time"]))
            max_label_time_error = max(max_label_time_error, time_error)
            # H1 label CSVs round time to five decimals; preserve the full
            # encoder/pipeline timestamp in the output and verify row alignment.
            assert time_error <= 1e-5, (output_csv, output_row["time"], label_row["time"])
            assert int(output_row["segment"]) == int(label_row["segment"]), output_csv
            assert int(output_row["valid"]) == int(label_row["valid"]), output_csv
            total_valid += int(output_row["valid"])
            for column in expected_columns - {"segment", "valid"}:
                assert math.isfinite(float(output_row[column])), (output_csv, column)

        if protocol == "original_id_setup":
            pipeline_path = Path(metadata["source_details"]["pipeline_csv"]["path"])
            pipeline_fields, pipeline_rows = _rows(pipeline_path)
            assert len(pipeline_rows) == len(output_rows), (pipeline_path, output_csv)
            assert "time" in pipeline_fields, pipeline_path
            for name in COORDINATES:
                source_column = f"{name}_tau"
                assert source_column in pipeline_fields, (pipeline_path, source_column)
                for output_row, pipeline_row in zip(output_rows, pipeline_rows):
                    assert abs(float(output_row["time"]) - float(pipeline_row["time"])) <= 1e-9
                    error = abs(float(output_row[f"{name}_tau_hat"]) - float(pipeline_row[source_column]))
                    max_tau_hat_error = max(max_tau_hat_error, error)
                    assert error <= 1e-8, (output_csv, name, error)

        setup_paths = sorted(trial_root.glob("segments/segment_*/Setup_ID*.xml"))
        source_setups = metadata["source_details"]["original_id_setups"]
        assert len(setup_paths) == len(source_setups), (trial_root, len(setup_paths), len(source_setups))
        segment_cutoffs: list[float] = []
        for setup_path, source_setup in zip(setup_paths, source_setups):
            source_setup_path = Path(source_setup["path"])
            tree = ET.parse(setup_path)
            tool = tree.getroot().find("InverseDynamicsTool")
            source_tool = ET.parse(source_setup_path).getroot().find("InverseDynamicsTool")
            assert tool is not None and source_tool is not None, setup_path
            assert _tool_signature(tool, protocol) == _tool_signature(source_tool, protocol), setup_path
            cutoff = float(tool.findtext("lowpass_cutoff_frequency_for_coordinates"))
            source_cutoff = float(source_tool.findtext("lowpass_cutoff_frequency_for_coordinates"))
            if protocol == "original_id_setup":
                assert cutoff == source_cutoff, (setup_path, cutoff, source_cutoff)
            else:
                assert cutoff == -1.0, (setup_path, cutoff)
                assert source_cutoff == float(metadata["coordinate_cutoff_hz"])
            assert cutoff == float(metadata["effective_coordinate_cutoff_hz"])
            segment_cutoffs.append(cutoff)

            model_path = _resolve(tool.findtext("model_file") or "", setup_path.parent)
            coordinates_path = _resolve(tool.findtext("coordinates_file") or "", setup_path.parent)
            external_path = _resolve(tool.findtext("external_loads_file") or "", setup_path.parent)
            assert model_path.is_file() and coordinates_path.is_file() and external_path.is_file(), setup_path
            output_q_hash = _sha256(coordinates_path)
            if protocol == "original_id_setup":
                source_ik = Path(metadata["source_details"]["ik"]["path"])
                segment_index = int(setup_path.parent.name.removeprefix("segment_"))
                ik_candidates = [source_ik]
                if source_ik.name.endswith("_ik.mot"):
                    segmented_name = source_ik.name.removesuffix("_ik.mot") + f"_segment_{segment_index}_ik.mot"
                    ik_candidates.append(source_ik.with_name(segmented_name))
                matching_candidates = [candidate for candidate in ik_candidates if candidate.is_file()]
            else:
                matching_candidates = [setup_path.parent / "coordinates_explicit_order4.mot"]
            q_error = 0.0 if any(candidate.is_file() and _sha256(candidate) == output_q_hash for candidate in matching_candidates) else 1.0
            max_q_copy_error = max(max_q_copy_error, q_error)
            assert q_error == 0.0, (coordinates_path, matching_candidates)

            geometry = setup_path.parent / "Geometry"
            assert geometry.is_dir() and not geometry.is_symlink(), geometry
            source_geometry = Path(metadata["geometry_source"])
            model_root = ET.parse(model_path).getroot()
            for mesh in model_root.iter("mesh_file"):
                mesh_name = (mesh.text or "").strip()
                if mesh_name:
                    mesh_path = Path(mesh_name)
                    if not mesh_path.is_absolute():
                        mesh_path = geometry / mesh_path
                    if not mesh_path.is_file():
                        source_mesh = source_geometry / Path(mesh_name)
                        if source_mesh.is_file():
                            raise FileNotFoundError((model_path, mesh_path, source_mesh))
                        missing_source_meshes.add(mesh_name)

            external_root = ET.parse(external_path).getroot()
            force_bodies = sorted(
                node.text.strip()
                for node in external_root.iter("applied_to_body")
                if node.text and node.text.strip()
            )
            assert force_bodies == ["calcn_l", "calcn_r"], (external_path, force_bodies)
            datafile = external_root.find(".//datafile")
            assert datafile is not None and datafile.text, external_path
            measured_mot = _resolve(datafile.text.strip(), external_path.parent)
            assert measured_mot.is_file(), measured_mot

        assert len(set(segment_cutoffs)) == 1, (trial_root, segment_cutoffs)
        assert math.isclose(
            segment_cutoffs[0],
            float(metadata["effective_coordinate_cutoff_hz"]),
            abs_tol=1e-12,
        )
        total_frames += len(output_rows)
        setup_count += len(setup_paths)

    report = {
        "protocol": protocol,
        "trials": len(records),
        "per_subject": actual_counts,
        "frames": total_frames,
        "valid_frames": total_valid,
        "setups": setup_count,
        "source_files_verified": source_file_count,
        "max_label_time_error_s": max_label_time_error,
        "max_tau_hat_vs_pipeline_error_nm": max_tau_hat_error if protocol == "original_id_setup" else None,
        "q_artifacts_match_saved_setup_inputs": max_q_copy_error == 0.0,
        "unresolved_mesh_references_already_missing_in_source_models": sorted(missing_source_meshes),
        "output_root": str(output_root),
    }
    validation_path = output_root / "logs" / "cohort_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = validation_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(validation_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=("original_id_setup", "explicit_order4"), default="original_id_setup")
    args = parser.parse_args()
    print(json.dumps(validate(args.output_dir, args.protocol), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

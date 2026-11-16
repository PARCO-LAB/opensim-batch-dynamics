#!/usr/bin/env python3
"""Make copied GroundLink torque-target outputs self-contained and portable."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from aggregate_groundlink_torque_targets import aggregate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite(value, mappings: list[tuple[str, str]]):
    if isinstance(value, str):
        for old, new in mappings:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_rewrite(item, mappings) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite(item, mappings) for key, item in value.items()}
    return value


def _link_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _artifact_files(trial_root: Path) -> list[str]:
    """List trial files while pruning mesh directories from the walk."""
    files: list[str] = []
    for directory, subdirectories, filenames in os.walk(trial_root):
        current = Path(directory)
        subdirectories[:] = [name for name in subdirectories if name != "Geometry"]
        for filename in filenames:
            path = current / filename
            if not path.is_symlink():
                files.append(str(path.relative_to(trial_root)))
    return sorted(files)


def _geometry_directories(trial_root: Path) -> list[str]:
    paths: list[str] = []
    for directory, subdirectories, _ in os.walk(trial_root):
        current = Path(directory)
        for name in list(subdirectories):
            if name == "Geometry":
                path = current / name
                if not path.is_symlink():
                    paths.append(str(path.relative_to(trial_root)))
                subdirectories.remove(name)
    return sorted(paths)


def prepare(
    output_root: Path,
    path_mappings: list[tuple[str, str]],
    geometry_source_root: Path | None = None,
    protocol: str = "original_id_setup",
) -> dict[str, int | str]:
    output_root = output_root.expanduser().resolve()
    manifest_paths = sorted(output_root.glob("execution_manifest_s*.csv"))
    if len(manifest_paths) != 6:
        raise FileNotFoundError(f"Expected six subject manifests under {output_root}, found {len(manifest_paths)}")

    processed = 0
    geometry_dirs = 0
    coordinate_files = 0
    report_trials: list[dict[str, str | int]] = []
    for manifest_path in manifest_paths:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
            fieldnames = list(records[0]) if records else []
        for record in records:
            subject, trial = record["subject"], record["trial"]
            trial_root = output_root / subject / trial
            metadata_path = trial_root / "run_metadata.json"
            output_csv = trial_root / "torque_targets.csv"
            if not metadata_path.is_file() or not output_csv.is_file():
                raise FileNotFoundError(f"Missing trial output or metadata: {trial_root}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            setups = sorted(trial_root.glob("segments/segment_*/Setup_ID*.xml"))
            if not setups:
                raise FileNotFoundError(f"No saved ID setups under {trial_root}")
            geometry_paths: list[str] = []
            for setup_path in setups:
                segment_dir = setup_path.parent
                geometry_dir = segment_dir / "Geometry"
                if geometry_dir.is_symlink():
                    geometry_dir.unlink()
                if not geometry_dir.is_dir():
                    if geometry_source_root is not None:
                        source_geometry = (
                            geometry_source_root / subject / trial / "segments"
                            / segment_dir.name / "Geometry"
                        )
                    else:
                        source_value = _rewrite(metadata.get("geometry_source", ""), path_mappings)
                        source_geometry = Path(source_value) if source_value else Path()
                    if not source_geometry.is_dir():
                        raise FileNotFoundError(
                            f"Cannot locate source Geometry for {subject}/{trial}/{segment_dir.name}: "
                            f"{source_geometry}"
                        )
                    shutil.copytree(
                        source_geometry,
                        geometry_dir,
                        copy_function=_link_or_copy,
                    )
                    geometry_dirs += 1
                geometry_paths.append(str(geometry_dir.relative_to(trial_root)))

                tree = ET.parse(setup_path)
                tool = tree.getroot().find("InverseDynamicsTool")
                if tool is None:
                    raise ValueError(f"No InverseDynamicsTool in {setup_path}")
                coordinates_node = tool.find("coordinates_file")
                if coordinates_node is None or not coordinates_node.text:
                    raise ValueError(f"Setup has no coordinates_file: {setup_path}")
                coordinates_value = coordinates_node.text.strip()
                coordinates_candidate = Path(coordinates_value)
                if not coordinates_candidate.is_absolute():
                    coordinates_candidate = segment_dir / coordinates_candidate
                if not coordinates_candidate.is_file():
                    rewritten = Path(_rewrite(str(coordinates_candidate), path_mappings))
                    if rewritten.is_file():
                        coordinates_candidate = rewritten
                    else:
                        raise FileNotFoundError(
                            f"Cannot locate setup coordinates for {subject}/{trial}: {coordinates_candidate}"
                        )
                local_coordinates = segment_dir / "coordinates_input.mot"
                if coordinates_candidate.resolve() != local_coordinates.resolve():
                    if not local_coordinates.exists():
                        _link_or_copy(str(coordinates_candidate), str(local_coordinates))
                    coordinate_files += 1
                coordinates_node.text = os.path.relpath(local_coordinates, segment_dir)
                setup_temporary = setup_path.with_suffix(setup_path.suffix + ".partial")
                tree.write(setup_temporary, encoding="utf-8", xml_declaration=True)
                setup_temporary.replace(setup_path)

                model_node = tool.find("model_file")
                external_node = tool.find("external_loads_file")
                for node, name in ((model_node, "model_file"), (external_node, "external_loads_file")):
                    if node is None or not node.text:
                        raise ValueError(f"Setup has no {name}: {setup_path}")
                    resolved = Path(node.text)
                    if not resolved.is_absolute():
                        resolved = segment_dir / resolved
                    if not resolved.is_file():
                        raise FileNotFoundError(f"Unresolved {name} in {setup_path}: {resolved}")

            metadata = _rewrite(metadata, path_mappings)
            metadata["output_csv"] = str(output_csv)
            metadata["portable_setup_files"] = [
                str(setup.relative_to(trial_root)) for setup in setups
            ]
            metadata["geometry_directories"] = _geometry_directories(trial_root)
            metadata["artifact_files"] = _artifact_files(trial_root)
            _atomic_json(metadata_path, metadata)

            record["output_csv"] = str(output_csv)
            record["metadata_json"] = str(metadata_path)
            record["output_sha256"] = _sha256(output_csv)
            record["metadata_sha256"] = _sha256(metadata_path)
            for key in ("source_hashes_json", "source_details_json"):
                if record.get(key):
                    parsed = json.loads(record[key])
                    record[key] = json.dumps(_rewrite(parsed, path_mappings), sort_keys=True)
            report_trials.append({"subject": subject, "trial": trial, "segments": len(setups)})
            processed += 1

        temporary_manifest = manifest_path.with_suffix(".csv.partial")
        with temporary_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        temporary_manifest.replace(manifest_path)

    combined = aggregate(output_root, protocol)
    report = {
        "trials": processed,
        "geometry_directories_created": geometry_dirs,
        "coordinate_files_created": coordinate_files,
        "manifest": str(combined),
        "output_root": str(output_root),
        "validated_setups": sum(int(row["segments"]) for row in report_trials),
    }
    validation_path = output_root / "logs" / "portable_setup_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(validation_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--path-map", action="append", nargs=2, default=[], metavar=("OLD_PREFIX", "NEW_PREFIX"),
        help="Rewrite a path prefix in metadata and source paths; may be specified repeatedly.",
    )
    parser.add_argument(
        "--geometry-source-root", type=Path,
        help="Optional parallel output tree whose segment Geometry directories can be reused.",
    )
    parser.add_argument(
        "--protocol", choices=("original_id_setup", "explicit_order4"),
        default="original_id_setup",
    )
    args = parser.parse_args()
    result = prepare(
        args.output_dir,
        [tuple(item) for item in args.path_map],
        args.geometry_source_root,
        args.protocol,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

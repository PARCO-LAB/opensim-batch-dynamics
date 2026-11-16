#!/usr/bin/env python3
"""Validate and combine per-subject GroundLink torque-target manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

EXPECTED_COUNTS = {"s001": 33, "s002": 33, "s003": 36, "s004": 27, "s005": 35, "s006": 42}
FIELDS = [
    "subject", "trial", "protocol", "status", "frames", "cutoff_hz",
    "moment_source", "output_csv", "metadata_json", "output_sha256",
    "metadata_sha256", "source_hashes_json", "source_details_json", "error",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(output_root: Path, protocol: str = "original_id_setup") -> Path:
    records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for subject, expected in EXPECTED_COUNTS.items():
        manifest = output_root / f"execution_manifest_{subject}.csv"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing per-subject manifest: {manifest}")
        with manifest.open(newline="", encoding="utf-8") as handle:
            subject_records = list(csv.DictReader(handle))
        if len(subject_records) != expected:
            raise ValueError(f"{subject} manifest has {len(subject_records)} trials; expected {expected}")
        for record in subject_records:
            key = (record.get("subject", ""), record.get("trial", ""))
            if key[0] != subject:
                raise ValueError(f"Unexpected subject {key[0]!r} in {manifest}")
            if key in seen:
                raise ValueError(f"Duplicate subject/trial in worker manifests: {key}")
            seen.add(key)
            if record.get("protocol") != protocol:
                raise ValueError(f"Unexpected protocol in {manifest}: {record.get('protocol')!r}")
            if record.get("status") != "ok":
                raise RuntimeError(f"Failed trial {key}: {record.get('error', '')}")
            target = output_root / subject / key[1] / "torque_targets.csv"
            metadata = target.with_name("run_metadata.json")
            if not target.is_file() or not metadata.is_file():
                raise FileNotFoundError(f"Successful manifest row has missing output: {target}")
            if record.get("output_sha256") != _sha256(target):
                raise ValueError(f"Output hash mismatch for {target}")
            if record.get("metadata_sha256") != _sha256(metadata):
                raise ValueError(f"Metadata hash mismatch for {metadata}")
            try:
                json.loads(record.get("source_hashes_json", ""))
                source_details = json.loads(record.get("source_details_json", ""))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid source hashes in manifest row {key}") from exc
            if not source_details or "measured_force" not in source_details or "ik" not in source_details:
                raise ValueError(f"Source paths are missing from manifest row {key}")
            records.append(record)
    if len(records) != 206:
        raise ValueError(f"Combined manifest has {len(records)} records; expected exactly 206")

    output = output_root / "execution_manifest.csv"
    temporary = output.with_suffix(".csv.partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--output-root", dest="output_root", type=Path, required=True)
    parser.add_argument("--protocol", default="original_id_setup")
    args = parser.parse_args()
    manifest = aggregate(args.output_root.expanduser().resolve(), args.protocol)
    print(f"Aggregated 206 successful trial records into {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

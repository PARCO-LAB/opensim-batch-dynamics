#!/usr/bin/env python3
"""Run the rows in an alternative-load ID manifest, or one SLURM array row."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


FIELDS = ("setup", "external_loads_file", "output_dir", "output_name", "forces_to_exclude")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV with one ID run per row")
    parser.add_argument("--row-index", type=int, default=None, help="Zero-based row; defaults to SLURM_ARRAY_TASK_ID")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = sorted(set(FIELDS[:3]) - set(rows[0] if rows else ()))
    if missing:
        parser.error(f"Manifest lacks columns: {', '.join(missing)}")
    row_index = args.row_index
    if row_index is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        row_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    selected = [(row_index, rows[row_index])] if row_index is not None and 0 <= row_index < len(rows) else (
        list(enumerate(rows)) if row_index is None else []
    )
    if not selected:
        parser.error(f"No manifest row selected (rows={len(rows)}, row-index={row_index})")
    runner = Path(__file__).with_name("ref_id.py")
    failures = []
    for index, row in selected:
        command = [
            sys.executable, str(runner), "--setup", row["setup"],
            "--external-loads", row["external_loads_file"], "--output-dir", row["output_dir"],
        ]
        if row.get("output_name"):
            command += ["--output-name", row["output_name"]]
        if row.get("forces_to_exclude"):
            command += ["--forces-to-exclude", row["forces_to_exclude"]]
        result = subprocess.run(command, check=False)
        print(f"row={index} returncode={result.returncode}", flush=True)
        if result.returncode:
            failures.append(index)
            if not args.keep_going:
                break
    print(f"rows={len(selected)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run one (variant, subject) slice of a hybrid ID manifest."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ref_id import run_id  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--group-index", type=int, default=None)
    parser.add_argument("--list-groups", action="store_true")
    args = parser.parse_args()
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = list(dict.fromkeys((row["variant"], row["subject"]) for row in rows))
    if args.list_groups:
        for index, (variant, subject) in enumerate(groups):
            print(f"{index},{variant},{subject},{sum((r['variant'], r['subject']) == (variant, subject) for r in rows)}")
        return 0
    if args.group_index is None or not 0 <= args.group_index < len(groups):
        parser.error(f"--group-index must be in [0, {len(groups) - 1}]")
    variant, subject = groups[args.group_index]
    selected = [row for row in rows if (row["variant"], row["subject"]) == (variant, subject)]
    failures: list[str] = []
    skipped = 0
    for index, row in enumerate(selected):
        output_dir = Path(row["output_dir"])
        output_file = output_dir / row["output_name"]
        if output_file.is_file() and output_file.stat().st_size > 0:
            skipped += 1
            continue
        if output_dir.exists():
            shutil.rmtree(output_dir)
        try:
            result = run_id(
                setup_path=Path(row["setup"]),
                loads_path=Path(row["external_loads_file"]),
                output_dir=output_dir,
                output_name=row["output_name"],
                forces_to_exclude=row.get("forces_to_exclude") or "Actuators",
            )
            print(f"{variant} {subject} {row['trial']} segment={row['segment']} ok {result}", flush=True)
        except Exception as exc:  # Keep the remaining independent segments running.
            failures.append(f"{row['trial']}/segment_{row['segment']}: {type(exc).__name__}: {exc}")
            print(f"{variant} {subject} {row['trial']} segment={row['segment']} FAILED {failures[-1]}", flush=True)
    status_path = args.manifest.parent / "status" / f"{variant}_{subject}.csv"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("variant", "subject", "rows", "skipped_existing", "failures"))
        writer.writerow((variant, subject, len(selected), skipped, len(failures)))
        for failure in failures:
            writer.writerow((variant, subject, "", "", failure))
    print(f"group={variant}/{subject} rows={len(selected)} skipped={skipped} failures={len(failures)}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

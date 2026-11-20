#!/usr/bin/env python3
"""Compute the E5 per-sequence quality diagnostic from pipeline wrench traces.

This score is intentionally model-internal: it verifies whether the contact-wrench
solver matched its own force target.  It is not a substitute for force-plate
validation and does not include quantities the pipeline does not export yet
(marker fit, foot skate, ground-plane fit, or mass provenance).
"""
from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable


GRAVITY = 9.81
CONTACT_THRESHOLD_BW = 0.05
# A 95th-percentile residual of 0.5 BW marks a clearly unreliable dynamic trial.
# This is a screening threshold, to be calibrated against reference-force error.
RESIDUAL_P95_LIMIT_BW = 0.50
UNSUPPORTED_FRACTION_LIMIT = 0.10


def _number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower, upper = int(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mass_from_final_csv(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream), None)
    if row is None:
        raise ValueError("empty final CSV")
    mass = _number(row, "subject_mass_kg")
    if mass <= 0:
        raise ValueError(f"invalid subject mass: {mass}")
    return mass


def score_wrench_csv(wrench_csv: Path, mass_kg: float) -> dict[str, float | int]:
    """Return transparent score components for one contact-wrench trace."""
    bw = mass_kg * GRAVITY
    residuals_bw: list[float] = []
    jerk_bw_s: list[float] = []
    target_count = unsupported_count = 0
    previous_time: float | None = None
    previous_target: tuple[float, float, float] | None = None
    with wrench_csv.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            target = tuple(_number(row, f"target_force_{axis}") for axis in "xyz")
            achieved_z = _number(row, "achieved_force_z")
            residuals_bw.append(_number(row, "force_balance_residual_norm") / bw)
            if target[2] > CONTACT_THRESHOLD_BW * bw:
                target_count += 1
                if achieved_z <= CONTACT_THRESHOLD_BW * bw:
                    unsupported_count += 1
            now = _number(row, "time")
            if previous_time is not None and previous_target is not None and now > previous_time:
                delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, previous_target)))
                jerk_bw_s.append(delta / (now - previous_time) / bw)
            previous_time, previous_target = now, target
    if not residuals_bw:
        raise ValueError("empty wrench CSV")

    residual_mean = sum(residuals_bw) / len(residuals_bw)
    residual_p95 = _percentile(residuals_bw, 95)
    unsupported_fraction = unsupported_count / target_count if target_count else 0.0
    # ponytail: only score diagnostics exported for every completed sequence.
    quality_score = max(
        0.0,
        1.0 - max(residual_p95 / RESIDUAL_P95_LIMIT_BW, unsupported_fraction / UNSUPPORTED_FRACTION_LIMIT),
    )
    return {
        "frames": len(residuals_bw),
        "force_balance_residual_mean_bw": residual_mean,
        "force_balance_residual_p95_bw": residual_p95,
        "target_contact_frames": target_count,
        "unsupported_target_fraction": unsupported_fraction,
        "target_force_jerk_p95_bw_s": _percentile(jerk_bw_s, 95),
        "quality_score": quality_score,
    }


def _wrench_csv(pipeline_dir: Path, final_csv: Path) -> Path | None:
    matches = sorted(pipeline_dir.glob(
        f"{final_csv.stem}_*/results/ID_estimatedGRF/*_contact_wrenches_estimated.csv"
    ))
    return matches[0] if matches else None


def _score_final_csv(pipeline_dir: Path, final_csv: Path) -> dict[str, str | float | int]:
    row: dict[str, str | float | int] = {"trial": final_csv.stem, "final_csv": str(final_csv)}
    wrench = _wrench_csv(pipeline_dir, final_csv)
    if wrench is None:
        row.update({"status": "missing_wrench", "wrench_csv": ""})
    else:
        try:
            row.update(score_wrench_csv(wrench, _mass_from_final_csv(final_csv)))
            row.update({"status": "ok", "wrench_csv": str(wrench)})
        except (KeyError, ValueError, OSError) as exc:
            row.update({"status": f"invalid: {exc}", "wrench_csv": str(wrench)})
    return row


def score_pipeline(pipeline_dir: Path, workers: int = 8) -> list[dict[str, str | float |int]]:
    final_csvs = sorted(pipeline_dir.glob("*_stageii.csv"))
    with ThreadPoolExecutor(max_workers=min(workers, len(final_csvs) or 1)) as pool:
        return list(pool.map(lambda path: _score_final_csv(pipeline_dir, path), final_csvs))


def write_report(rows: Iterable[dict[str, str | float | int]], output: Path) -> None:
    fields = [
        "trial", "status", "quality_score", "force_balance_residual_mean_bw",
        "force_balance_residual_p95_bw", "unsupported_target_fraction",
        "target_contact_frames", "target_force_jerk_p95_bw_s", "frames", "final_csv", "wrench_csv",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True, help="One subject directory, e.g. pipeline/s007")
    parser.add_argument("--output", type=Path, help="CSV report path (default: <pipeline-dir>/quality_report.csv)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent NAS reads (default: 8)")
    args = parser.parse_args()
    output = args.output or args.pipeline_dir / "quality_report.csv"
    rows = score_pipeline(args.pipeline_dir, args.workers)
    write_report(rows, output)
    ok = sum(row["status"] == "ok" for row in rows)
    print(f"Wrote {output}: {ok}/{len(rows)} trials scored")
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

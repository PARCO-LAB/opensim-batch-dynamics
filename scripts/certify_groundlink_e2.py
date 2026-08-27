#!/usr/bin/env python3
"""Write the seven-subject E2 certification table from per-subject metrics."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


SUBJECTS = tuple(f"s{i:03d}" for i in range(1, 8))
METHODS = ("BW_equal_split", "COM_equal_split", "pipeline")
METRICS = (
    ("fz_rmse_contact", "RMSE Fz [BW]"),
    ("fz_corr_contact", "r_contact"),
    ("fz_peak_error_abs_median", "|peak| [BW]"),
    ("fz_impulse_error_abs", "|impulse| [s]"),
    ("contact_f1", "Contact F1"),
    ("pfa_planar_error", "PFA [m]"),
    ("physics_residual", "Physics residual [BW]"),
)
T95_DF6 = 2.447  # 95% two-sided Student-t critical value for seven subjects.


def _mean_ci(values: list[float]) -> tuple[float, float]:
    if len(values) != 7:
        raise ValueError(f"expected seven subject means, got {len(values)}")
    return statistics.mean(values), T95_DF6 * statistics.stdev(values) / math.sqrt(7)


def summarize(root: Path) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    result: dict[str, dict[str, dict[str, tuple[float, float]]]] = {method: {} for method in METHODS}
    for method in METHODS:
        for metric, _ in METRICS:
            if metric == "pfa_planar_error" and method != "pipeline":
                continue
            subject_means: dict[str, float] = {}
            for subject in SUBJECTS:
                path = root / subject / "metrics_per_trial.csv"
                with path.open(newline="", encoding="utf-8") as handle:
                    values = [float(row["value"]) for row in csv.DictReader(handle)
                              if row["method"] == method and row["metric"] == metric
                              and row["value"] not in ("", "nan") and math.isfinite(float(row["value"]))]
                if not values:
                    raise ValueError(f"missing {method}/{metric} for {subject}")
                subject_means[subject] = statistics.mean(values)
            mean, ci = _mean_ci(list(subject_means.values()))
            result[method][metric] = {subject: (value, 0.0) for subject, value in subject_means.items()}
            result[method][metric]["all"] = (mean, ci)
    return result


def _format(value: tuple[float, float] | None) -> str:
    return "—" if value is None else f"{value[0]:.3f} ± {value[1]:.3f}"


def render(summary: dict[str, dict[str, dict[str, tuple[float, float]]]]) -> str:
    header = "| Method | " + " | ".join(label for _, label in METRICS) + " |"
    divider = "|---|" + "---:|" * len(METRICS)
    rows = ["# E2 — Certification on seven subjects", "", "**PASS:** all required metrics are present for s001–s007. Values are mean ± 95% t-CI across subject means.", "", header, divider]
    for method in METHODS:
        rows.append("| " + method + " | " + " | ".join(
            _format(summary[method].get(metric, {}).get("all")) for metric, _ in METRICS
        ) + " |")
    rows.extend(["", "## Pipeline per subject", "", header, divider])
    for subject in SUBJECTS:
        rows.append("| " + subject + " | " + " | ".join(
            _format(summary["pipeline"].get(metric, {}).get(subject)) for metric, _ in METRICS
        ) + " |")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.metrics_root / "e2_7_subjects.md"
    output.write_text(render(summarize(args.metrics_root)), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

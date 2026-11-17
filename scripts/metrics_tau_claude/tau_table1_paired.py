#!/usr/bin/env python3
"""Paired subject-level comparisons for the Table I methods."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t, ttest_rel


METHODS = ("BW_equal_split", "CoM_equal_split")


def paired_table(metrics_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(metrics_path)
    frame = frame[(frame["level"] == "trial") & frame["method"].isin(("pipeline", *METHODS))]
    subject = (
        frame.groupby(["subject", "method", "coordinate", "metric"], dropna=False)["value"]
        .mean()
        .reset_index()
    )
    rows = []
    for (coordinate, metric), group in subject.groupby(["coordinate", "metric"], dropna=False):
        wide = group.pivot(index="subject", columns="method", values="value")
        if "pipeline" not in wide:
            continue
        for reference in METHODS:
            if reference not in wide:
                continue
            pair = wide[["pipeline", reference]].dropna()
            if pair.empty:
                continue
            delta = pair["pipeline"] - pair[reference]
            mean = float(delta.mean())
            half = (
                float(student_t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / math.sqrt(len(delta)))
                if len(delta) > 1 else float("nan")
            )
            pvalue = float(ttest_rel(pair["pipeline"], pair[reference]).pvalue) if len(delta) > 1 else float("nan")
            rows.append({
                "coordinate": "" if pd.isna(coordinate) else coordinate,
                "metric": metric,
                "comparison": f"pipeline - {reference}",
                "n_subjects": len(delta),
                "mean_delta": mean,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "paired_t_pvalue": pvalue,
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="baseline_metrics.csv from tau_table1_all.py")
    parser.add_argument("--output", type=Path, help="Optional CSV path; defaults beside the input file")
    args = parser.parse_args()
    result = paired_table(args.metrics)
    output = args.output or args.metrics.with_name("paired_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"paired_rows={len(result)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Add paired subject-level summaries for the Phase C direct-GRF ablation."""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


METRICS = (
    "fz_rmse_contact_bw", "contact_f1", "cop_error_cm", "horizontal_rmse_bw",
    "false_contact_pooled", "J_vs_baseline",
)


def signflip_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if not len(values):
        return math.nan
    observed = abs(float(np.mean(values)))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        total += 1
        extreme += abs(float(np.mean(values * np.asarray(signs)))) >= observed - 1e-12
    return extreme / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = pd.read_csv(args.ablation)
    rows = []
    baseline = data[data.variant == "BASELINE"].set_index(["fold", "phase"])
    for variant in sorted(set(data.variant) - {"BASELINE"}):
        selected = data[data.variant == variant]
        for phase, group in selected.groupby("phase", sort=True):
            for metric in METRICS:
                diffs, bases, candidates = [], [], []
                for _, row in group.iterrows():
                    key = (row.fold, phase)
                    if key not in baseline.index:
                        continue
                    base_value = float(baseline.loc[key, metric])
                    candidate_value = float(row[metric])
                    if math.isfinite(base_value) and math.isfinite(candidate_value):
                        bases.append(base_value)
                        candidates.append(candidate_value)
                        diffs.append(candidate_value - base_value)
                delta = np.asarray(diffs, dtype=float)
                n = len(delta)
                mean_delta = float(np.mean(delta)) if n else math.nan
                half = float(t.ppf(0.975, n - 1) * np.std(delta, ddof=1) / math.sqrt(n)) if n > 1 else math.nan
                rows.append({
                    "variant": variant, "phase": phase, "metric": metric, "n_subjects": n,
                    "baseline_mean": float(np.mean(bases)) if n else math.nan,
                    "variant_mean": float(np.mean(candidates)) if n else math.nan,
                    "mean_paired_delta": mean_delta,
                    "ci95_low": mean_delta - half if math.isfinite(half) else math.nan,
                    "ci95_high": mean_delta + half if math.isfinite(half) else math.nan,
                    "signflip_p": signflip_p(delta),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"variants={data.variant.nunique() - 1} summary_rows={len(rows)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

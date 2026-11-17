#!/usr/bin/env python3
"""Summarize CoP, horizontal-force, and false-contact metrics by subject."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXTRA = (
    "cop_error_cm",
    "horizontal_rmse_bw",
    "false_contact_fraction",
    "false_contact_fraction_all_valid",
    "false_contact_fraction_body_flight",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True, help="baseline_metrics.csv from tau_table1_all.py")
    parser.add_argument("--output", type=Path, help="Optional CSV path; defaults beside the input file")
    args = parser.parse_args()
    frame = pd.read_csv(args.metrics)
    frame = frame[
        (frame["method"] == "pipeline")
        & frame["metric"].isin(EXTRA)
        & frame["level"].isin(("subject", "overall"))
    ]
    output = args.output or args.metrics.with_name("grf_extra.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"\nrows={len(frame)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

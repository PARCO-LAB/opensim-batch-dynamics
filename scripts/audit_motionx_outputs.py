#!/usr/bin/env python3
"""Audit final Motion-X CSVs; keeps translational forces separate from torques."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def audit(path: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
    t = np.asarray([float(r["time"]) for r in rows])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    rates = {}
    for h in headers:
        if h.endswith("_tau") and h in {"pelvis_tx_tau", "pelvis_ty_tau", "pelvis_tz_tau"}:
            continue
        if h.endswith("_tau"):
            rates[h] = float(np.max(np.abs(np.diff(np.asarray([float(r[h]) for r in rows])))) / dt)
    root = [float(np.max(np.abs(np.diff(np.asarray([float(r[h]) for r in rows])))) / dt) for h in headers if h in {"pelvis_tx_tau", "pelvis_ty_tau", "pelvis_tz_tau"}]
    return {"csv": str(path), "frames": len(rows), "joint_torque_rate_max": max(rates.values(), default=0.0), "root_force_rate_max": max(root, default=0.0)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    report = [audit(path) for path in sorted(args.root.rglob("*.csv"))]
    payload = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

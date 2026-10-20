from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _pelvis_angular_speed(data: dict[str, np.ndarray]) -> np.ndarray:
    """Return frame-to-frame pelvis angular speed from the full Z-X-Y rotation.

    OpenSim's ground-pelvis CustomJoint uses intrinsic rotations about Z, X,
    then Y.  Near the X-axis gimbal singularity, tilt and rotation coordinate
    rates can both be very large while the physical orientation changes
    slowly.  Comparing each Euler-coordinate rate independently therefore
    rejects valid motion.
    """
    required = ("time", "pelvis_tilt", "pelvis_list", "pelvis_rotation")
    if any(key not in data for key in required):
        return np.empty(0, dtype=float)

    time = np.asarray(data["time"], dtype=float)
    if len(time) < 2:
        return np.empty(0, dtype=float)
    dt = np.diff(time)
    if np.any(dt <= 0):
        return np.full(len(dt), np.inf, dtype=float)

    tilt = np.asarray(data["pelvis_tilt"], dtype=float)
    listing = np.asarray(data["pelvis_list"], dtype=float)
    rotation = np.asarray(data["pelvis_rotation"], dtype=float)

    ct, st = np.cos(tilt), np.sin(tilt)
    cl, sl = np.cos(listing), np.sin(listing)
    cr, sr = np.cos(rotation), np.sin(rotation)

    rz = np.zeros((len(time), 3, 3), dtype=float)
    rz[:, 0, 0], rz[:, 0, 1] = ct, -st
    rz[:, 1, 0], rz[:, 1, 1] = st, ct
    rz[:, 2, 2] = 1.0

    rx = np.zeros_like(rz)
    rx[:, 0, 0] = 1.0
    rx[:, 1, 1], rx[:, 1, 2] = cl, -sl
    rx[:, 2, 1], rx[:, 2, 2] = sl, cl

    ry = np.zeros_like(rz)
    ry[:, 0, 0], ry[:, 0, 2] = cr, sr
    ry[:, 1, 1] = 1.0
    ry[:, 2, 0], ry[:, 2, 2] = -sr, cr

    orientation = rz @ rx @ ry
    relative = np.swapaxes(orientation[:-1], 1, 2) @ orientation[1:]
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cosine) / dt


def assess_csv(path: str | Path, *, mass_kg: float | None = None) -> dict[str, object]:
    """Small, unit-aware sanity gate for exported inverse-dynamics CSVs."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames or []
    if not rows:
        return {"status": "rejected", "reason": "empty_csv"}
    data = {h: np.array([float(r[h]) for r in rows], dtype=float) for h in headers if h}
    if any(not np.all(np.isfinite(v)) for v in data.values()):
        return {"status": "rejected", "reason": "nonfinite"}
    t = data.get("time")
    dt = float(np.median(np.diff(t))) if t is not None and len(t) > 1 else 1 / 30
    reasons: list[str] = []
    if t is not None and len(t) > 1 and float(t[-1] - t[0]) < 1.0:
        reasons.append("duration_lt_1s")
    pelvis = [data[k] for k in ("pelvis_tx", "pelvis_ty", "pelvis_tz") if k in data]
    if pelvis and max(float(np.ptp(v)) for v in pelvis) > 10:
        reasons.append("pelvis_span_gt_10m")
    for key, values in data.items():
        if key.endswith(("_tx", "_ty", "_tz")):
            continue
        if key.endswith("_tau") and np.max(np.abs(np.diff(values))) / max(dt, 1e-9) > 1e7:
            reasons.append("torque_rate_gt_1e7")
    pelvis_speed = _pelvis_angular_speed(data)
    if pelvis_speed.size and float(np.max(pelvis_speed)) > 20:
        reasons.append("pelvis_rotation_speed_gt_20rad_s")
    return {"status": "rejected" if reasons else "ok", "reasons": reasons, "frames": len(rows), "dt_s": dt}


def leading_pelvis_transient_frames(path: str | Path, *, max_frames: int = 10) -> int:
    """Return the shortest unstable leading prefix, or zero if it is not one."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) < 3:
        return 0
    time = np.array([float(row["time"]) for row in rows], dtype=float)
    keys = ("pelvis_tilt", "pelvis_list", "pelvis_rotation")
    if any(key not in rows[0] for key in keys):
        return 0
    pelvis = {
        "time": time,
        **{
            key: np.array([float(row[key]) for row in rows], dtype=float)
            for key in keys
        },
    }
    for trim in range(1, min(max_frames, len(rows) - 2) + 1):
        speed = _pelvis_angular_speed(
            {key: values[trim:] for key, values in pelvis.items()}
        )
        if speed.size and float(np.max(speed)) <= 20:
            return trim
    return 0


def write_quality(path: str | Path, report: dict[str, object]) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

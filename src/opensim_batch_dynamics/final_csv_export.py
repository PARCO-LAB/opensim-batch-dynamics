from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FinalCsvSummary:
    output_csv_path: Path
    frames: int
    dof_names: tuple[str, ...]
    contact_body_names: tuple[str, ...]


def _to_float(raw: str) -> float:
    text = raw.strip()
    if text == "":
        return math.nan
    return float(text)


def _load_numeric_csv(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        if not rows:
            raise ValueError(f"CSV has no data rows: {path}")

    headers = [str(name) for name in reader.fieldnames]
    data: dict[str, np.ndarray] = {}
    for header in headers:
        data[header] = np.asarray([_to_float(row.get(header, "")) for row in rows], dtype=np.float64)
    return headers, data


def _align_series(source_t: np.ndarray, source_v: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    if (
        source_t.shape == target_t.shape
        and np.all(np.isfinite(source_t))
        and np.all(np.isfinite(target_t))
        and np.allclose(source_t, target_t, atol=1e-8, rtol=1e-6)
    ):
        return source_v.copy()

    valid = np.isfinite(source_t) & np.isfinite(source_v)
    if int(np.sum(valid)) < 2:
        return np.full(target_t.shape, math.nan, dtype=np.float64)
    return np.interp(
        target_t,
        source_t[valid],
        source_v[valid],
        left=float(source_v[valid][0]),
        right=float(source_v[valid][-1]),
    )


def _find_contact_bodies(headers: list[str]) -> list[str]:
    suffix = "_force_x"
    names: list[str] = []
    for header in headers:
        if not header.endswith(suffix):
            continue
        candidate = header[: -len(suffix)]
        # Keep only real contact bodies that also expose COP columns.
        if f"{candidate}_cop_x" not in headers:
            continue
        names.append(candidate)
    return names


def export_final_csv(
    dof_csv_path: str | Path,
    torque_csv_path: str | Path,
    output_csv_path: str | Path,
    contact_wrench_csv_path: str | Path | None,
    excluded_dofs: tuple[str, ...] = (),
    fallback_contact_bodies: tuple[str, ...] = ("calcn_l", "calcn_r"),
    contact_force_threshold_n: float = 1e-6,
) -> FinalCsvSummary:
    """
    Export one unified CSV with:
    - positions
    - velocities
    - accelerations
    - torques
    - GRFs (per contact body + total)
    """
    dof_csv = Path(dof_csv_path).resolve()
    torque_csv = Path(torque_csv_path).resolve()
    output_csv = Path(output_csv_path).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    dof_headers, dof_data = _load_numeric_csv(dof_csv)
    torque_headers, torque_data = _load_numeric_csv(torque_csv)

    if "time" not in dof_data:
        raise ValueError(f"Missing 'time' column in DOF CSV: {dof_csv}")
    if "time" not in torque_data:
        raise ValueError(f"Missing 'time' column in torque CSV: {torque_csv}")

    time_values = dof_data["time"]
    frames = int(time_values.shape[0])
    excluded = set(excluded_dofs)

    dof_names: list[str] = []
    for header in dof_headers:
        if header in ("frame", "time"):
            continue
        if header.endswith("_vel") or header.endswith("_acc"):
            continue
        if header in excluded:
            continue
        if np.isnan(dof_data[header]).all():
            continue
        dof_names.append(header)

    aligned_torque: dict[str, np.ndarray] = {}
    for dof_name in dof_names:
        torque_col = f"{dof_name}_tau"
        if torque_col not in torque_data:
            aligned_torque[torque_col] = np.full((frames,), math.nan, dtype=np.float64)
            continue
        aligned_torque[torque_col] = _align_series(
            source_t=torque_data["time"],
            source_v=torque_data[torque_col],
            target_t=time_values,
        )

    contact_body_names: list[str] = []
    aligned_grf: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    if contact_wrench_csv_path is not None:
        contact_csv = Path(contact_wrench_csv_path).resolve()
        if contact_csv.exists():
            contact_headers, contact_data = _load_numeric_csv(contact_csv)
            contact_body_names = _find_contact_bodies(contact_headers)
            for body_name in contact_body_names:
                fx = _align_series(contact_data["time"], contact_data[f"{body_name}_force_x"], time_values)
                fy = _align_series(contact_data["time"], contact_data[f"{body_name}_force_y"], time_values)
                fz = _align_series(contact_data["time"], contact_data[f"{body_name}_force_z"], time_values)
                aligned_grf[body_name] = (fx, fy, fz)

    if not contact_body_names:
        contact_body_names = [name for name in fallback_contact_bodies]
        for body_name in contact_body_names:
            zeros = np.zeros((frames,), dtype=np.float64)
            aligned_grf[body_name] = (zeros.copy(), zeros.copy(), zeros.copy())

    total_fx = np.zeros((frames,), dtype=np.float64)
    total_fy = np.zeros((frames,), dtype=np.float64)
    total_fz = np.zeros((frames,), dtype=np.float64)
    for body_name in contact_body_names:
        fx, fy, fz = aligned_grf[body_name]
        total_fx += fx
        total_fy += fy
        total_fz += fz

    contact_codes: dict[str, np.ndarray] = {}
    for body_name in contact_body_names:
        fx, fy, fz = aligned_grf[body_name]
        norm = np.sqrt(fx * fx + fy * fy + fz * fz)
        contact_codes[body_name] = (norm > float(contact_force_threshold_n)).astype(np.int32)

    with output_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        header = ["frame", "time"]
        for dof_name in dof_names:
            header.extend([dof_name, f"{dof_name}_vel", f"{dof_name}_acc", f"{dof_name}_tau"])
        for body_name in contact_body_names:
            header.extend(
                [
                    f"{body_name}_grf_x",
                    f"{body_name}_grf_y",
                    f"{body_name}_grf_z",
                    f"{body_name}_contact",
                ]
            )
        header.extend(["grf_total_x", "grf_total_y", "grf_total_z"])
        writer.writerow(header)

        for frame_idx in range(frames):
            row: list[float | int] = [frame_idx, float(time_values[frame_idx])]
            for dof_name in dof_names:
                row.append(float(dof_data[dof_name][frame_idx]))
                row.append(float(dof_data[f"{dof_name}_vel"][frame_idx]))
                row.append(float(dof_data[f"{dof_name}_acc"][frame_idx]))
                row.append(float(aligned_torque[f"{dof_name}_tau"][frame_idx]))
            for body_name in contact_body_names:
                fx, fy, fz = aligned_grf[body_name]
                row.append(float(fx[frame_idx]))
                row.append(float(fy[frame_idx]))
                row.append(float(fz[frame_idx]))
                row.append(int(contact_codes[body_name][frame_idx]))
            row.append(float(total_fx[frame_idx]))
            row.append(float(total_fy[frame_idx]))
            row.append(float(total_fz[frame_idx]))
            writer.writerow(row)

    return FinalCsvSummary(
        output_csv_path=output_csv,
        frames=frames,
        dof_names=tuple(dof_names),
        contact_body_names=tuple(contact_body_names),
    )

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class FinalCsvSummary:
    output_csv_path: Path
    frames: int
    dof_names: tuple[str, ...]
    contact_body_names: tuple[str, ...]
    body_scale_names: tuple[str, ...]
    mass_kg: float
    height_m: float


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


def _load_subject_mass_height(subject_json_path: Path | None) -> tuple[float, float]:
    if subject_json_path is None or not subject_json_path.exists():
        return math.nan, math.nan
    try:
        payload = json.loads(subject_json_path.read_text(encoding="utf-8"))
    except Exception:
        return math.nan, math.nan
    mass = payload.get("massKg", math.nan)
    height = payload.get("heightM", math.nan)
    try:
        mass_value = float(mass)
    except Exception:
        mass_value = math.nan
    try:
        height_value = float(height)
    except Exception:
        height_value = math.nan
    return mass_value, height_value


def _parse_vec3(text: str | None) -> tuple[float, float, float] | None:
    if text is None:
        return None
    parts = text.strip().split()
    if len(parts) < 3:
        return None
    try:
        values = (float(parts[0]), float(parts[1]), float(parts[2]))
    except Exception:
        return None
    return values


def _extract_body_scales_and_mass(model_path: Path | None) -> tuple[list[tuple[str, tuple[float, float, float]]], float]:
    """
    Extract one representative scale triplet per Body and model total mass from .osim XML.

    For each body, OpenSim XML commonly contains:
    - frame-geometry default scale (often 0.2, 0.2, 0.2)
    - actual mesh/body scale factors
    We ignore default frame-geometry scales when better candidates exist.
    """
    if model_path is None or not model_path.exists():
        return [], math.nan
    try:
        root = ET.parse(model_path).getroot()
    except Exception:
        return [], math.nan

    body_entries: list[tuple[str, tuple[float, float, float]]] = []
    total_mass = 0.0
    has_mass = False

    for body_elem in root.findall(".//BodySet/objects/Body"):
        body_name = body_elem.attrib.get("name", "").strip()
        if not body_name:
            continue

        # Aggregate mass for fallback subject mass.
        mass_elem = body_elem.find("./mass")
        if mass_elem is not None and mass_elem.text:
            try:
                total_mass += float(mass_elem.text.strip())
                has_mass = True
            except Exception:
                pass

        candidates: list[tuple[float, float, float]] = []
        for scale_elem in body_elem.findall(".//scale_factors"):
            parsed = _parse_vec3(scale_elem.text)
            if parsed is not None:
                candidates.append(parsed)
        if not candidates:
            continue

        default_scale = np.array([0.2, 0.2, 0.2], dtype=np.float64)
        non_default = [
            scale
            for scale in candidates
            if not np.allclose(np.asarray(scale, dtype=np.float64), default_scale, atol=1e-10)
        ]
        usable = non_default if non_default else candidates

        # Pick the candidate closest to the component-wise median for robustness.
        usable_array = np.asarray(usable, dtype=np.float64)
        median = np.median(usable_array, axis=0)
        dist2 = np.sum((usable_array - median.reshape(1, 3)) ** 2, axis=1)
        idx = int(np.argmin(dist2))
        selected = usable_array[idx, :]
        body_entries.append(
            (
                body_name,
                (float(selected[0]), float(selected[1]), float(selected[2])),
            )
        )

    return body_entries, (float(total_mass) if has_mass else math.nan)


def export_final_csv(
    dof_csv_path: str | Path,
    torque_csv_path: str | Path,
    output_csv_path: str | Path,
    contact_wrench_csv_path: str | Path | None,
    subject_json_path: str | Path | None = None,
    model_path: str | Path | None = None,
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
    subject_json = Path(subject_json_path).resolve() if subject_json_path is not None else None
    resolved_model_path = Path(model_path).resolve() if model_path is not None else None

    dof_headers, dof_data = _load_numeric_csv(dof_csv)
    torque_headers, torque_data = _load_numeric_csv(torque_csv)

    if "time" not in dof_data:
        raise ValueError(f"Missing 'time' column in DOF CSV: {dof_csv}")
    if "time" not in torque_data:
        raise ValueError(f"Missing 'time' column in torque CSV: {torque_csv}")

    mass_kg, height_m = _load_subject_mass_height(subject_json)
    body_scales, model_mass_kg = _extract_body_scales_and_mass(resolved_model_path)
    if not np.isfinite(mass_kg) and np.isfinite(model_mass_kg):
        mass_kg = model_mass_kg

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
        header.extend(["subject_mass_kg", "subject_height_m"])
        for body_name, _ in body_scales:
            header.extend(
                [
                    f"{body_name}_scale_x",
                    f"{body_name}_scale_y",
                    f"{body_name}_scale_z",
                ]
            )
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
            row.append(float(mass_kg))
            row.append(float(height_m))
            for _, scales in body_scales:
                row.append(float(scales[0]))
                row.append(float(scales[1]))
                row.append(float(scales[2]))
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
        body_scale_names=tuple(name for name, _ in body_scales),
        mass_kg=float(mass_kg),
        height_m=float(height_m),
    )

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class MotionCsvData:
    path: Path
    columns: list[str]
    values: dict[str, np.ndarray]
    n_frames: int
    time: np.ndarray
    time_source: str
    sample_rate_hz: float | None
    dof_names: list[str]
    root_dofs: list[str]
    grf_bodies: list[str]
    has_total_grf: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a multi-page PDF report from a pipeline CSV "
            "(positions, velocities, accelerations, GRF, contacts, torques)."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Path to input CSV in pipeline format")
    parser.add_argument(
        "--output-pdf",
        default=None,
        help="Output PDF path (default: <input_stem>_report.pdf next to CSV)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title shown in the first report page",
    )
    parser.add_argument(
        "--max-dofs",
        type=int,
        default=None,
        help="Optional cap on number of DOFs plotted (default: all)",
    )
    parser.add_argument(
        "--root-force-warning-n",
        type=float,
        default=75.0,
        help="Warning threshold for root translational residual magnitude in N (default: 75.0)",
    )
    parser.add_argument(
        "--root-moment-warning-nm",
        type=float,
        default=25.0,
        help="Warning threshold for root rotational residual magnitude in Nm (default: 25.0)",
    )
    return parser.parse_args()


def _parse_float(value: str) -> float:
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _safe_nan_percentile(data: np.ndarray, q: float) -> float:
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0.0
    return float(np.nanpercentile(data[finite], q))


def _safe_range(data: np.ndarray) -> float:
    finite = np.isfinite(data)
    if not np.any(finite):
        return float("nan")
    return float(np.nanmax(data[finite]) - np.nanmin(data[finite]))


def _safe_peak_abs(data: np.ndarray) -> float:
    finite = np.isfinite(data)
    if not np.any(finite):
        return float("nan")
    return float(np.nanmax(np.abs(data[finite])))


def _safe_rms(data: np.ndarray) -> float:
    finite = np.isfinite(data)
    if not np.any(finite):
        return float("nan")
    y = data[finite]
    return float(np.sqrt(np.mean(y * y)))


def _estimate_sample_rate(time_values: np.ndarray) -> float | None:
    if time_values.size < 2:
        return None
    dt = np.diff(time_values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    dt_median = float(np.median(dt))
    if dt_median <= 0:
        return None
    return 1.0 / dt_median


def _infer_time(values: dict[str, np.ndarray], n_frames: int) -> tuple[np.ndarray, str, float | None]:
    if "time" in values:
        t = values["time"].astype(np.float64, copy=True)
        finite = np.isfinite(t)
        if np.any(finite):
            first_valid = np.argmax(finite)
            if first_valid > 0:
                t[:first_valid] = t[first_valid]
            for idx in range(1, t.size):
                if not np.isfinite(t[idx]):
                    t[idx] = t[idx - 1]
            if np.any(np.diff(t) <= 0):
                eps = 1e-6
                for idx in range(1, t.size):
                    if t[idx] <= t[idx - 1]:
                        t[idx] = t[idx - 1] + eps
            return t, "time_column", _estimate_sample_rate(t)
    if "frame" in values:
        frame = values["frame"].astype(np.float64, copy=False)
        return frame, "frame_column", _estimate_sample_rate(frame)
    t = np.arange(n_frames, dtype=np.float64)
    return t, "frame_index", None


def _detect_dofs(columns: list[str], values: dict[str, np.ndarray]) -> list[str]:
    dof_names: list[str] = []
    for col in columns:
        if col in {"frame", "time"}:
            continue
        if col.startswith("grf_total_"):
            continue
        if "_grf_" in col or col.endswith("_contact"):
            continue
        if col.endswith("_vel") or col.endswith("_acc") or col.endswith("_tau"):
            continue
        if col not in values:
            continue
        # Keep base columns only when at least one derivative/torque partner exists.
        if any(f"{col}{suffix}" in values for suffix in ("_vel", "_acc", "_tau")):
            dof_names.append(col)
    return dof_names


def _detect_grf_bodies(columns: list[str]) -> list[str]:
    axis_pat = re.compile(r"^(?P<body>.+)_grf_(?P<axis>[xyz])$")
    body_order: list[str] = []
    seen: set[str] = set()
    for col in columns:
        match = axis_pat.match(col)
        if not match:
            continue
        body = match.group("body")
        if body not in seen:
            seen.add(body)
            body_order.append(body)
    return body_order


def load_motion_csv(path: str | Path) -> MotionCsvData:
    csv_path = Path(path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        columns = list(reader.fieldnames)
        storage: dict[str, list[float]] = {col: [] for col in columns}
        row_count = 0
        for row in reader:
            row_count += 1
            for col in columns:
                storage[col].append(_parse_float(row.get(col, "")))

    if row_count == 0:
        raise ValueError(f"CSV has no data rows: {csv_path}")

    values = {col: np.asarray(storage[col], dtype=np.float64) for col in columns}
    time, time_source, sample_rate_hz = _infer_time(values, row_count)
    dof_names = _detect_dofs(columns, values)
    root_dofs = dof_names[:6]
    grf_bodies = _detect_grf_bodies(columns)
    has_total_grf = all(name in values for name in ("grf_total_x", "grf_total_y", "grf_total_z"))

    return MotionCsvData(
        path=csv_path,
        columns=columns,
        values=values,
        n_frames=row_count,
        time=time,
        time_source=time_source,
        sample_rate_hz=sample_rate_hz,
        dof_names=dof_names,
        root_dofs=root_dofs,
        grf_bodies=grf_bodies,
        has_total_grf=has_total_grf,
    )


def _format_float(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _is_translational_dof(name: str) -> bool:
    return name.endswith("_tx") or name.endswith("_ty") or name.endswith("_tz")


def _root_residual_summary(
    data: MotionCsvData,
    force_warning_n: float,
    moment_warning_nm: float,
) -> dict[str, object]:
    root_dofs = data.root_dofs[:6]
    root_moment_names = [name for name in root_dofs if not _is_translational_dof(name)]
    root_force_names = [name for name in root_dofs if _is_translational_dof(name)]
    root_moment_values = [data.values[name] for name in root_moment_names if name in data.values]
    root_force_values = [data.values[name] for name in root_force_names if name in data.values]

    root_moment_mag = (
        np.sqrt(np.sum(np.square(np.vstack(root_moment_values)), axis=0))
        if root_moment_values
        else np.array([], dtype=np.float64)
    )
    root_force_mag = (
        np.sqrt(np.sum(np.square(np.vstack(root_force_values)), axis=0))
        if root_force_values
        else np.array([], dtype=np.float64)
    )

    moment_peak = _safe_peak_abs(root_moment_mag)
    force_peak = _safe_peak_abs(root_force_mag)
    warning_lines: list[str] = []
    if np.isfinite(force_peak) and force_peak > force_warning_n:
        warning_lines.append(
            f"Root translational residuals are large: peak |F| = {_format_float(force_peak, 2)} N "
            f"> threshold {_format_float(force_warning_n, 2)} N"
        )
    if np.isfinite(moment_peak) and moment_peak > moment_warning_nm:
        warning_lines.append(
            f"Root rotational residuals are large: peak |M| = {_format_float(moment_peak, 2)} Nm "
            f"> threshold {_format_float(moment_warning_nm, 2)} Nm"
        )

    per_dof_peaks = []
    for name in root_dofs:
        values = data.values.get(name)
        if values is None:
            continue
        threshold = force_warning_n if _is_translational_dof(name) else moment_warning_nm
        peak = _safe_peak_abs(values)
        per_dof_peaks.append(
            {
                "name": name,
                "peak": peak,
                "threshold": threshold,
                "is_warning": bool(np.isfinite(peak) and peak > threshold),
            }
        )
        if np.isfinite(peak) and peak > threshold:
            kind = "force" if _is_translational_dof(name) else "moment"
            unit = "N" if kind == "force" else "Nm"
            warning_lines.append(
                f"{name} {kind} peak {_format_float(peak, 2)} {unit} exceeds "
                f"{_format_float(threshold, 2)} {unit}"
            )

    return {
        "root_dofs": root_dofs,
        "root_force_mag": root_force_mag,
        "root_moment_mag": root_moment_mag,
        "force_peak": force_peak,
        "moment_peak": moment_peak,
        "warning_lines": warning_lines,
        "per_dof_peaks": per_dof_peaks,
    }


def _contact_intervals(time_values: np.ndarray, contact: np.ndarray) -> list[tuple[float, float]]:
    if contact.size == 0:
        return []
    mask = np.asarray(contact > 0.5, dtype=bool)
    if not np.any(mask):
        return []

    starts: list[int] = []
    ends: list[int] = []
    for i, is_on in enumerate(mask):
        prev_on = bool(mask[i - 1]) if i > 0 else False
        if is_on and not prev_on:
            starts.append(i)
        if not is_on and prev_on:
            ends.append(i - 1)
    if mask[-1]:
        ends.append(mask.size - 1)

    # Estimate half-step extension for cleaner visual spans.
    dt = np.diff(time_values)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    half_step = 0.5 * float(np.median(dt)) if dt.size > 0 else 0.0

    intervals: list[tuple[float, float]] = []
    for start_idx, end_idx in zip(starts, ends):
        start_t = float(time_values[start_idx] - half_step)
        end_t = float(time_values[end_idx] + half_step)
        intervals.append((start_t, end_t))
    return intervals


def _shade_contact(ax: plt.Axes, intervals: list[tuple[float, float]]) -> None:
    for start_t, end_t in intervals:
        ax.axvspan(start_t, end_t, color="#DDF1FF", alpha=0.35, linewidth=0)


def _plot_signal(
    ax: plt.Axes,
    time_values: np.ndarray,
    signal: np.ndarray,
    label: str,
    color: str,
    contact_intervals: list[tuple[float, float]] | None = None,
) -> None:
    if contact_intervals:
        _shade_contact(ax, contact_intervals)
    ax.plot(time_values, signal, color=color, linewidth=1.0)
    ax.set_ylabel(label)
    ax.grid(True, linestyle="--", alpha=0.25)


def _summary_lines(data: MotionCsvData) -> list[str]:
    duration_s = float(data.time[-1] - data.time[0]) if data.n_frames > 1 else 0.0
    n_vel = sum(1 for name in data.dof_names if f"{name}_vel" in data.values)
    n_acc = sum(1 for name in data.dof_names if f"{name}_acc" in data.values)
    n_tau = sum(1 for name in data.dof_names if f"{name}_tau" in data.values)

    lines = [
        f"Input CSV: {data.path}",
        f"Frames: {data.n_frames}",
        f"Duration: {_format_float(duration_s)} s",
        f"Time source: {data.time_source}",
        f"Estimated sample rate: {_format_float(data.sample_rate_hz)} Hz",
        f"DOF count (base columns): {len(data.dof_names)}",
        f"DOF with velocity: {n_vel}",
        f"DOF with acceleration: {n_acc}",
        f"DOF with torque: {n_tau}",
        f"GRF bodies: {', '.join(data.grf_bodies) if data.grf_bodies else 'none'}",
        f"Total GRF columns: {'yes' if data.has_total_grf else 'no'}",
    ]

    contact_lines: list[str] = []
    for body in data.grf_bodies:
        contact_col = f"{body}_contact"
        if contact_col not in data.values:
            continue
        contact = data.values[contact_col]
        valid = np.isfinite(contact)
        if not np.any(valid):
            ratio = float("nan")
        else:
            ratio = float(np.mean(contact[valid] > 0.5) * 100.0)
        contact_lines.append(f"{body}: {_format_float(ratio, 1)}%")
    if contact_lines:
        lines.append("Contact ratio (% of frames): " + " | ".join(contact_lines))

    if data.has_total_grf:
        gx = data.values["grf_total_x"]
        gy = data.values["grf_total_y"]
        gz = data.values["grf_total_z"]
        gmag = np.sqrt(gx * gx + gy * gy + gz * gz)
        lines.append(f"Peak |GRF total|: {_format_float(_safe_peak_abs(gmag), 2)}")
        lines.append(f"Peak GRF_y total: {_format_float(_safe_peak_abs(gy), 2)}")

    return lines


def _rom_table_lines(data: MotionCsvData, limit: int = 15) -> list[str]:
    rows: list[tuple[str, float, float, float]] = []
    for dof in data.dof_names:
        q = data.values.get(dof)
        if q is None:
            continue
        rom = _safe_range(q)
        vmax = _safe_peak_abs(data.values.get(f"{dof}_vel", np.array([], dtype=np.float64)))
        amax = _safe_peak_abs(data.values.get(f"{dof}_acc", np.array([], dtype=np.float64)))
        rows.append((dof, rom, vmax, amax))

    rows.sort(key=lambda x: (np.nan_to_num(x[1], nan=-1e9)), reverse=True)
    top = rows[:limit]

    lines = ["Top DOFs by ROM (range):", "DOF                               ROM        Peak|vel|   Peak|acc|"]
    for dof, rom, vmax, amax in top:
        lines.append(
            f"{dof[:30]:<30} "
            f"{_format_float(rom, 3):>10} "
            f"{_format_float(vmax, 3):>12} "
            f"{_format_float(amax, 3):>12}"
        )
    return lines


def add_title_page(pdf: PdfPages, data: MotionCsvData, title: str | None) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(title or "Motion CSV Explorer Report", fontsize=18, fontweight="bold", y=0.98)
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = _summary_lines(data)
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_root_residual_page(
    pdf: PdfPages,
    data: MotionCsvData,
    force_warning_n: float,
    moment_warning_nm: float,
) -> list[str]:
    summary = _root_residual_summary(data, force_warning_n, moment_warning_nm)
    root_dofs = summary["root_dofs"]
    warning_lines = list(summary["warning_lines"])

    fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27), sharex=True)
    fig.suptitle("Root Residuals", fontsize=16, fontweight="bold")

    if warning_lines:
        fig.text(
            0.5,
            0.955,
            "Warnings: " + " | ".join(warning_lines),
            ha="center",
            va="top",
            fontsize=9,
            color="#B91C1C",
            bbox=dict(facecolor="#FEE2E2", edgecolor="#FCA5A5", boxstyle="round,pad=0.4"),
        )
    else:
        fig.text(
            0.5,
            0.955,
            "No root residual warnings at the current thresholds.",
            ha="center",
            va="top",
            fontsize=9,
            color="#166534",
            bbox=dict(facecolor="#DCFCE7", edgecolor="#86EFAC", boxstyle="round,pad=0.4"),
        )

    contact_intervals: list[tuple[float, float]] = []
    for body in data.grf_bodies:
        ccol = f"{body}_contact"
        if ccol in data.values:
            contact_intervals.extend(_contact_intervals(data.time, data.values[ccol]))
    contact_intervals.sort(key=lambda x: x[0])

    root_force_names = [name for name in root_dofs if _is_translational_dof(name)]
    root_moment_names = [name for name in root_dofs if not _is_translational_dof(name)]
    plot_order = root_moment_names + root_force_names
    axes_flat = [ax for row in axes for ax in row]

    for idx, (ax, dof_name) in enumerate(zip(axes_flat, plot_order)):
        color = "#B45309" if _is_translational_dof(dof_name) else "#1D4ED8"
        threshold = force_warning_n if _is_translational_dof(dof_name) else moment_warning_nm
        unit = "N" if _is_translational_dof(dof_name) else "Nm"
        _plot_signal(
            ax,
            data.time,
            data.values[dof_name],
            f"{dof_name} [{unit}]",
            color,
            contact_intervals=contact_intervals,
        )
        ax.axhline(threshold, color="#B91C1C", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.axhline(-threshold, color="#B91C1C", linestyle="--", linewidth=0.9, alpha=0.8)
        peak = _safe_peak_abs(data.values[dof_name])
        ax.set_title(f"peak |{dof_name}| = {_format_float(peak, 2)} {unit}", fontsize=9)

    for ax in axes_flat[len(plot_order) :]:
        ax.axis("off")

    if warning_lines:
        fig.text(
            0.02,
            0.02,
            "Root residual warnings:\n" + "\n".join(f"- {line}" for line in warning_lines),
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
            color="#7F1D1D",
        )
    else:
        fig.text(
            0.02,
            0.02,
            "Root residuals are below the configured warning thresholds.",
            ha="left",
            va="bottom",
            fontsize=9,
            family="monospace",
            color="#14532D",
        )

    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return warning_lines


def add_overview_page(pdf: PdfPages, data: MotionCsvData) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("Global Motion Overview", fontsize=16, fontweight="bold")

    rom_values = []
    vel_peaks = []
    acc_peaks = []
    for dof in data.dof_names:
        rom_values.append(_safe_range(data.values[dof]))
        vel_peaks.append(_safe_peak_abs(data.values.get(f"{dof}_vel", np.array([], dtype=np.float64))))
        acc_peaks.append(_safe_peak_abs(data.values.get(f"{dof}_acc", np.array([], dtype=np.float64))))

    rom_values = np.asarray(rom_values, dtype=np.float64)
    vel_peaks = np.asarray(vel_peaks, dtype=np.float64)
    acc_peaks = np.asarray(acc_peaks, dtype=np.float64)

    axes[0, 0].hist(rom_values[np.isfinite(rom_values)], bins=20, color="#3B82F6", alpha=0.8)
    axes[0, 0].set_title("ROM distribution across DOFs")
    axes[0, 0].set_xlabel("Range")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, linestyle="--", alpha=0.25)

    axes[0, 1].hist(vel_peaks[np.isfinite(vel_peaks)], bins=20, color="#10B981", alpha=0.8)
    axes[0, 1].set_title("Peak |velocity| distribution")
    axes[0, 1].set_xlabel("Peak |vel|")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True, linestyle="--", alpha=0.25)

    axes[1, 0].hist(acc_peaks[np.isfinite(acc_peaks)], bins=20, color="#F59E0B", alpha=0.8)
    axes[1, 0].set_title("Peak |acceleration| distribution")
    axes[1, 0].set_xlabel("Peak |acc|")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, linestyle="--", alpha=0.25)

    ax_text = axes[1, 1]
    ax_text.axis("off")
    lines = _rom_table_lines(data, limit=14)
    ax_text.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_dof_pages(pdf: PdfPages, data: MotionCsvData, max_dofs: int | None = None) -> int:
    selected = data.dof_names if max_dofs is None else data.dof_names[: max(0, max_dofs)]
    plotted = 0
    for dof in selected:
        has_vel = f"{dof}_vel" in data.values
        has_acc = f"{dof}_acc" in data.values
        has_tau = f"{dof}_tau" in data.values

        nrows = 1 + int(has_vel) + int(has_acc) + int(has_tau)
        fig, axes = plt.subplots(nrows, 1, figsize=(11.69, 8.27), sharex=True)
        if nrows == 1:
            axes = [axes]
        else:
            axes = list(axes)

        fig.suptitle(f"DOF: {dof}", fontsize=15, fontweight="bold")

        contact_intervals: list[tuple[float, float]] = []
        # If foot contacts are present, add them as light background on all DOF plots.
        for body in data.grf_bodies:
            ccol = f"{body}_contact"
            if ccol in data.values:
                contact_intervals.extend(_contact_intervals(data.time, data.values[ccol]))
        contact_intervals.sort(key=lambda x: x[0])

        idx = 0
        pos_unit = "m" if _is_translational_dof(dof) else "deg"
        _plot_signal(
            axes[idx],
            data.time,
            data.values[dof],
            f"Position [{pos_unit}]",
            "#1D4ED8",
            contact_intervals=contact_intervals,
        )
        idx += 1

        if has_vel:
            vel_unit = "m/s" if _is_translational_dof(dof) else "deg/s"
            _plot_signal(
                axes[idx],
                data.time,
                data.values[f"{dof}_vel"],
                f"Velocity [{vel_unit}]",
                "#059669",
                contact_intervals=contact_intervals,
            )
            idx += 1

        if has_acc:
            acc_unit = "m/s^2" if _is_translational_dof(dof) else "deg/s^2"
            _plot_signal(
                axes[idx],
                data.time,
                data.values[f"{dof}_acc"],
                f"Acceleration [{acc_unit}]",
                "#D97706",
                contact_intervals=contact_intervals,
            )
            idx += 1

        if has_tau:
            # In OpenSim generalized forces are typically Nm for rotational coords and N for translational.
            tau_unit = "N" if _is_translational_dof(dof) else "Nm"
            _plot_signal(
                axes[idx],
                data.time,
                data.values[f"{dof}_tau"],
                f"Torque [{tau_unit}]",
                "#B91C1C",
                contact_intervals=contact_intervals,
            )

        axes[-1].set_xlabel("Time")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        plotted += 1
    return plotted


def add_grf_pages(pdf: PdfPages, data: MotionCsvData) -> int:
    page_count = 0

    for body in data.grf_bodies:
        x_col = f"{body}_grf_x"
        y_col = f"{body}_grf_y"
        z_col = f"{body}_grf_z"
        if not all(col in data.values for col in (x_col, y_col, z_col)):
            continue

        fx = data.values[x_col]
        fy = data.values[y_col]
        fz = data.values[z_col]
        fmag = np.sqrt(fx * fx + fy * fy + fz * fz)

        contact_col = f"{body}_contact"
        has_contact = contact_col in data.values
        nrows = 5 if has_contact else 4

        fig, axes = plt.subplots(nrows, 1, figsize=(11.69, 8.27), sharex=True)
        if nrows == 1:
            axes = [axes]
        else:
            axes = list(axes)

        contact_intervals = (
            _contact_intervals(data.time, data.values[contact_col]) if has_contact else []
        )

        fig.suptitle(f"GRF Report: {body}", fontsize=15, fontweight="bold")
        _plot_signal(
            axes[0],
            data.time,
            fx,
            "Fx [N]",
            "#2563EB",
            contact_intervals=contact_intervals,
        )
        _plot_signal(
            axes[1],
            data.time,
            fy,
            "Fy [N]",
            "#059669",
            contact_intervals=contact_intervals,
        )
        _plot_signal(
            axes[2],
            data.time,
            fz,
            "Fz [N]",
            "#DC2626",
            contact_intervals=contact_intervals,
        )
        _plot_signal(
            axes[3],
            data.time,
            fmag,
            "|F| [N]",
            "#7C3AED",
            contact_intervals=contact_intervals,
        )

        if has_contact:
            contact = data.values[contact_col]
            axes[4].step(data.time, contact, where="mid", color="#111827", linewidth=1.2)
            axes[4].set_ylabel("Contact")
            axes[4].set_yticks([0, 1])
            axes[4].grid(True, linestyle="--", alpha=0.25)

        axes[-1].set_xlabel("Time")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_count += 1

    if data.has_total_grf:
        gx = data.values["grf_total_x"]
        gy = data.values["grf_total_y"]
        gz = data.values["grf_total_z"]
        gmag = np.sqrt(gx * gx + gy * gy + gz * gz)

        fig, axes = plt.subplots(4, 1, figsize=(11.69, 8.27), sharex=True)
        fig.suptitle("GRF Report: Total", fontsize=15, fontweight="bold")
        _plot_signal(axes[0], data.time, gx, "GRF total X [N]", "#2563EB")
        _plot_signal(axes[1], data.time, gy, "GRF total Y [N]", "#059669")
        _plot_signal(axes[2], data.time, gz, "GRF total Z [N]", "#DC2626")
        _plot_signal(axes[3], data.time, gmag, "|GRF total| [N]", "#7C3AED")
        axes[3].set_xlabel("Time")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_count += 1

    return page_count


def default_output_pdf(input_csv: Path) -> Path:
    return input_csv.with_name(f"{input_csv.stem}_report.pdf")


def build_pdf_report(
    data: MotionCsvData,
    output_pdf: Path,
    title: str | None,
    max_dofs: int | None,
    root_force_warning_n: float,
    root_moment_warning_nm: float,
) -> dict[str, object]:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        add_title_page(pdf, data, title=title)
        add_overview_page(pdf, data)
        root_warning_lines = add_root_residual_page(
            pdf,
            data,
            force_warning_n=root_force_warning_n,
            moment_warning_nm=root_moment_warning_nm,
        )
        dof_pages = add_dof_pages(pdf, data, max_dofs=max_dofs)
        grf_pages = add_grf_pages(pdf, data)

    return {
        "input_csv": str(data.path),
        "output_pdf": str(output_pdf),
        "frames": data.n_frames,
        "duration_s": float(data.time[-1] - data.time[0]) if data.n_frames > 1 else 0.0,
        "dof_count": len(data.dof_names),
        "root_dof_count": len(data.root_dofs),
        "dof_pages": dof_pages,
        "grf_pages": grf_pages,
        "sample_rate_hz": data.sample_rate_hz,
        "root_warning_lines": root_warning_lines,
    }


def main() -> int:
    args = parse_args()
    data = load_motion_csv(args.input_csv)
    output_pdf = (
        Path(args.output_pdf).resolve()
        if args.output_pdf is not None
        else default_output_pdf(data.path).resolve()
    )

    summary = build_pdf_report(
        data=data,
        output_pdf=output_pdf,
        title=args.title,
        max_dofs=args.max_dofs,
        root_force_warning_n=args.root_force_warning_n,
        root_moment_warning_nm=args.root_moment_warning_nm,
    )

    print("CSV Explorer report generated successfully.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

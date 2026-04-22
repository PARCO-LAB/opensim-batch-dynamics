#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from csv_explorer import load_motion_csv


METRIC_EXCLUDE_PREFIXES = (
    "ankle_angle_",
    "subtalar_angle_",
    "head_",
    "wrist_",
    "pro_sup_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a multi-page PDF report that compares an offline pipeline CSV "
            "against a realtime reconstructed CSV."
        )
    )
    parser.add_argument("--offline-csv", required=True, type=Path, help="Path to offline pipeline CSV")
    parser.add_argument("--realtime-csv", required=True, type=Path, help="Path to realtime reconstructed CSV")
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=None,
        help="Output PDF path (default: <realtime_stem>_vs_<offline_stem>.pdf)",
    )
    parser.add_argument("--title", default=None, help="Optional title for the report")
    parser.add_argument("--max-dofs", type=int, default=None, help="Optional cap on number of DOFs plotted")
    return parser.parse_args()


def default_output_pdf(offline_csv: Path, realtime_csv: Path) -> Path:
    return realtime_csv.with_name(f"{realtime_csv.stem}_vs_{offline_csv.stem}.pdf")


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def smart_min_span(signal_kind: str, dof_name: str | None = None) -> float:
    translational = False if dof_name is None else is_translational_dof(dof_name)

    if signal_kind == "q":
        return 0.06 if translational else 0.35
    if signal_kind == "dq":
        return 0.20 if translational else 1.0
    if signal_kind == "ddq":
        return 1.0 if translational else 6.0
    if signal_kind == "tau":
        return 60.0 if translational else 15.0
    if signal_kind == "force":
        return 120.0
    if signal_kind == "error_small":
        return 0.01
    if signal_kind == "error_large":
        return 10.0
    if signal_kind == "time_ms":
        return 2.0
    return 1.0


def compute_smart_ylim(
    *arrays: np.ndarray,
    min_span: float,
    pad_ratio: float = 0.12,
    center_on_zero: bool = False,
) -> tuple[float, float] | None:
    finite_arrays = []
    for arr in arrays:
        values = np.asarray(arr, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            finite_arrays.append(values)

    if not finite_arrays:
        return None

    values = np.concatenate(finite_arrays)
    low = float(np.min(values))
    high = float(np.max(values))

    if center_on_zero:
        bound = max(abs(low), abs(high), 0.5 * float(min_span))
        bound *= 1.0 + pad_ratio
        return (-bound, bound)

    span = high - low
    if span < float(min_span):
        mid = 0.5 * (low + high)
        half = 0.5 * float(min_span)
        low = mid - half
        high = mid + half
    else:
        pad = max(span * pad_ratio, 0.02 * float(min_span))
        low -= pad
        high += pad

    return (float(low), float(high))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def binary_classification_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_bool = np.asarray(pred, dtype=bool)
    target_bool = np.asarray(target, dtype=bool)
    tp = float(np.sum(pred_bool & target_bool))
    tn = float(np.sum((~pred_bool) & (~target_bool)))
    fp = float(np.sum(pred_bool & (~target_bool)))
    fn = float(np.sum((~pred_bool) & target_bool))
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0.0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def is_translational_dof(name: str) -> bool:
    return name.endswith("_tx") or name.endswith("_ty") or name.endswith("_tz")


def include_in_precision_metrics(dof_name: str) -> bool:
    return not any(dof_name.startswith(prefix) for prefix in METRIC_EXCLUDE_PREFIXES)


def contact_intervals(time_values: np.ndarray, contact_values: np.ndarray) -> list[tuple[float, float]]:
    t = np.asarray(time_values, dtype=float)
    c = np.asarray(contact_values, dtype=bool)
    if t.size == 0 or c.size == 0:
        return []
    intervals: list[tuple[float, float]] = []
    start = None
    for idx, active in enumerate(c):
        if active and start is None:
            start = float(t[idx])
        if not active and start is not None:
            intervals.append((start, float(t[idx])))
            start = None
    if start is not None:
        intervals.append((start, float(t[-1])))
    return intervals


def plot_compare_signal(
    ax: plt.Axes,
    time_values: np.ndarray,
    offline_values: np.ndarray,
    realtime_values: np.ndarray,
    ylabel: str,
    contact_background: list[tuple[float, float]] | None = None,
    min_span: float | None = None,
    center_on_zero: bool = False,
) -> None:
    if contact_background:
        for start, end in contact_background:
            ax.axvspan(start, end, color="#ECFCCB", alpha=0.18, linewidth=0.0)
    ax.plot(time_values, offline_values, color="#111827", linewidth=1.3, label="offline")
    ax.plot(time_values, realtime_values, color="#2563EB", linewidth=1.1, alpha=0.85, label="realtime")
    ylim = compute_smart_ylim(
        offline_values,
        realtime_values,
        min_span=1.0 if min_span is None else float(min_span),
        center_on_zero=center_on_zero,
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def plot_contact_compare(
    ax: plt.Axes,
    time_values: np.ndarray,
    offline_values: np.ndarray,
    realtime_values: np.ndarray,
    title: str,
) -> None:
    ax.step(time_values, offline_values.astype(float), where="mid", color="#111827", linewidth=1.2, label="offline")
    ax.step(time_values, realtime_values.astype(float), where="mid", color="#2563EB", linewidth=1.0, label="realtime")
    ax.set_title(title, fontsize=10)
    ax.set_yticks([0.0, 1.0])
    ax.set_ylim(-0.15, 1.15)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)


def align_indices(offline_data, realtime_data) -> tuple[np.ndarray, np.ndarray, str]:
    if "frame" in offline_data.values and "frame" in realtime_data.values:
        offline_keys = np.round(offline_data.values["frame"].astype(float), 9)
        realtime_keys = np.round(realtime_data.values["frame"].astype(float), 9)
        key_name = "frame"
    elif "time" in offline_data.values and "time" in realtime_data.values:
        offline_keys = np.round(offline_data.values["time"].astype(float), 9)
        realtime_keys = np.round(realtime_data.values["time"].astype(float), 9)
        key_name = "time"
    else:
        n = min(offline_data.n_frames, realtime_data.n_frames)
        return np.arange(n, dtype=int), np.arange(n, dtype=int), "index"

    realtime_lookup = {}
    for idx, key in enumerate(realtime_keys):
        realtime_lookup.setdefault(float(key), idx)

    offline_indices = []
    realtime_indices = []
    for offline_idx, key in enumerate(offline_keys):
        rt_idx = realtime_lookup.get(float(key))
        if rt_idx is not None:
            offline_indices.append(offline_idx)
            realtime_indices.append(rt_idx)

    if not offline_indices:
        raise ValueError(f"No common {key_name} values between offline and realtime CSVs")

    return np.array(offline_indices, dtype=int), np.array(realtime_indices, dtype=int), key_name


def extract_side_force(data, side: str, indices: np.ndarray) -> np.ndarray:
    prefix = f"{side}_grf_"
    if all(prefix + axis in data.values for axis in ("x", "y", "z")):
        return np.column_stack([data.values[prefix + axis][indices] for axis in ("x", "y", "z")]).astype(float)

    side_suffix = "_l" if side == "left" else "_r"
    force = np.zeros((len(indices), 3), dtype=float)
    found = False
    for body_name in data.grf_bodies:
        if not body_name.endswith(side_suffix):
            continue
        cols = [f"{body_name}_grf_x", f"{body_name}_grf_y", f"{body_name}_grf_z"]
        if all(col in data.values for col in cols):
            force += np.column_stack([data.values[col][indices] for col in cols]).astype(float)
            found = True
    if found:
        return force
    return np.full((len(indices), 3), np.nan, dtype=float)


def extract_side_contact(data, side: str, indices: np.ndarray) -> np.ndarray:
    direct_name = f"{side}_contact"
    if direct_name in data.values:
        return data.values[direct_name][indices].astype(float) > 0.5

    side_suffix = "_l" if side == "left" else "_r"
    contact = np.zeros(len(indices), dtype=bool)
    found = False
    for body_name in data.grf_bodies:
        if not body_name.endswith(side_suffix):
            continue
        col = f"{body_name}_contact"
        if col in data.values:
            contact |= data.values[col][indices].astype(float) > 0.5
            found = True
    if found:
        return contact
    return np.zeros(len(indices), dtype=bool)


def extract_optional_column(data, column_name: str, indices: np.ndarray) -> np.ndarray | None:
    if column_name not in data.values:
        return None
    return np.asarray(data.values[column_name][indices], dtype=float)


def build_report(args: argparse.Namespace) -> dict[str, object]:
    offline_data = load_motion_csv(args.offline_csv)
    realtime_data = load_motion_csv(args.realtime_csv)
    offline_idx, realtime_idx, align_key = align_indices(offline_data, realtime_data)

    common_dofs = [name for name in offline_data.dof_names if name in realtime_data.dof_names]
    if not common_dofs:
        raise ValueError("No common DOF set found between offline and realtime CSVs")

    time_values = offline_data.time[offline_idx]
    q_offline = np.column_stack([offline_data.values[name][offline_idx] for name in common_dofs]).astype(float)
    q_realtime = np.column_stack([realtime_data.values[name][realtime_idx] for name in common_dofs]).astype(float)
    dq_offline = np.column_stack([offline_data.values[name + "_vel"][offline_idx] for name in common_dofs]).astype(float)
    dq_realtime = np.column_stack([realtime_data.values[name + "_vel"][realtime_idx] for name in common_dofs]).astype(float)
    ddq_offline = np.column_stack([offline_data.values[name + "_acc"][offline_idx] for name in common_dofs]).astype(float)
    ddq_realtime = np.column_stack([realtime_data.values[name + "_acc"][realtime_idx] for name in common_dofs]).astype(float)
    tau_offline = np.column_stack([offline_data.values[name + "_tau"][offline_idx] for name in common_dofs]).astype(float)
    tau_realtime = np.column_stack([realtime_data.values[name + "_tau"][realtime_idx] for name in common_dofs]).astype(float)
    metric_mask = np.array([include_in_precision_metrics(name) for name in common_dofs], dtype=bool)
    metric_dof_names = [name for name, keep in zip(common_dofs, metric_mask) if keep]
    metric_q_offline = q_offline[:, metric_mask]
    metric_q_realtime = q_realtime[:, metric_mask]
    metric_dq_offline = dq_offline[:, metric_mask]
    metric_dq_realtime = dq_realtime[:, metric_mask]
    metric_ddq_offline = ddq_offline[:, metric_mask]
    metric_ddq_realtime = ddq_realtime[:, metric_mask]
    metric_tau_offline = tau_offline[:, metric_mask]
    metric_tau_realtime = tau_realtime[:, metric_mask]
    metric_act_mask = metric_mask[6:] if len(metric_mask) > 6 else np.zeros(0, dtype=bool)
    metric_tau_act_offline = tau_offline[:, 6:][:, metric_act_mask] if len(metric_mask) > 6 else metric_tau_offline
    metric_tau_act_realtime = tau_realtime[:, 6:][:, metric_act_mask] if len(metric_mask) > 6 else metric_tau_realtime

    left_force_offline = extract_side_force(offline_data, "left", offline_idx)
    right_force_offline = extract_side_force(offline_data, "right", offline_idx)
    left_force_realtime = extract_side_force(realtime_data, "left", realtime_idx)
    right_force_realtime = extract_side_force(realtime_data, "right", realtime_idx)
    left_contact_offline = extract_side_contact(offline_data, "left", offline_idx)
    right_contact_offline = extract_side_contact(offline_data, "right", offline_idx)
    left_contact_realtime = extract_side_contact(realtime_data, "left", realtime_idx)
    right_contact_realtime = extract_side_contact(realtime_data, "right", realtime_idx)

    q_frame_rmse = np.sqrt(np.mean(np.square(metric_q_realtime - metric_q_offline), axis=1))
    tau_frame_rmse = np.sqrt(np.mean(np.square(metric_tau_realtime - metric_tau_offline), axis=1))
    total_force_offline = np.linalg.norm(left_force_offline + right_force_offline, axis=1)
    total_force_realtime = np.linalg.norm(left_force_realtime + right_force_realtime, axis=1)

    metrics = {
        "q_rmse": rmse(metric_q_realtime, metric_q_offline),
        "q_mae": mae(metric_q_realtime, metric_q_offline),
        "dq_rmse": rmse(metric_dq_realtime, metric_dq_offline),
        "dq_mae": mae(metric_dq_realtime, metric_dq_offline),
        "ddq_rmse": rmse(metric_ddq_realtime, metric_ddq_offline),
        "ddq_mae": mae(metric_ddq_realtime, metric_ddq_offline),
        "tau_full_rmse": rmse(metric_tau_realtime, metric_tau_offline),
        "tau_full_mae": mae(metric_tau_realtime, metric_tau_offline),
        "tau_actuated_rmse": rmse(metric_tau_act_realtime, metric_tau_act_offline),
        "tau_actuated_mae": mae(metric_tau_act_realtime, metric_tau_act_offline),
        "left_grf_rmse": rmse(left_force_realtime, left_force_offline),
        "right_grf_rmse": rmse(right_force_realtime, right_force_offline),
        "left_contact": binary_classification_metrics(left_contact_realtime, left_contact_offline),
        "right_contact": binary_classification_metrics(right_contact_realtime, right_contact_offline),
    }

    return {
        "title": args.title or "Realtime vs Offline CSV Report",
        "offline_csv": args.offline_csv.resolve(),
        "realtime_csv": args.realtime_csv.resolve(),
        "align_key": align_key,
        "frames": len(offline_idx),
        "time": time_values,
        "dof_names": common_dofs,
        "metric_dof_names": metric_dof_names,
        "q_offline": q_offline,
        "q_realtime": q_realtime,
        "dq_offline": dq_offline,
        "dq_realtime": dq_realtime,
        "ddq_offline": ddq_offline,
        "ddq_realtime": ddq_realtime,
        "tau_offline": tau_offline,
        "tau_realtime": tau_realtime,
        "left_force_offline": left_force_offline,
        "right_force_offline": right_force_offline,
        "left_force_realtime": left_force_realtime,
        "right_force_realtime": right_force_realtime,
        "left_contact_offline": left_contact_offline,
        "right_contact_offline": right_contact_offline,
        "left_contact_realtime": left_contact_realtime,
        "right_contact_realtime": right_contact_realtime,
        "q_frame_rmse": q_frame_rmse,
        "tau_frame_rmse": tau_frame_rmse,
        "total_force_offline": total_force_offline,
        "total_force_realtime": total_force_realtime,
        "mpjpe": extract_optional_column(realtime_data, "mpjpe_m", realtime_idx),
        "dyn_residual": extract_optional_column(realtime_data, "dynamics_residual_norm", realtime_idx),
        "solve_time_ms": extract_optional_column(realtime_data, "solve_time_ms", realtime_idx),
        "metrics": metrics,
    }


def add_title_page(pdf: PdfPages, report: dict[str, object]) -> None:
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(str(report["title"]), fontsize=18, fontweight="bold", y=0.98)
    ax = fig.add_subplot(111)
    ax.axis("off")

    lines = [
        f"Offline CSV:  {report['offline_csv']}",
        f"Realtime CSV: {report['realtime_csv']}",
        "",
        f"Matched frames:       {report['frames']}",
        f"Alignment key:        {report['align_key']}",
        f"Common DOFs:          {len(report['dof_names'])}",
        f"Metric DOFs used:     {len(report['metric_dof_names'])} (excluding ankle/head/wrist-related angles)",
        "",
        f"q RMSE:               {format_float(float(metrics['q_rmse']), 6)}",
        f"q MAE:                {format_float(float(metrics['q_mae']), 6)}",
        f"dq RMSE:              {format_float(float(metrics['dq_rmse']), 6)}",
        f"ddq RMSE:             {format_float(float(metrics['ddq_rmse']), 6)}",
        f"tau full RMSE:        {format_float(float(metrics['tau_full_rmse']), 6)}",
        f"tau actuated RMSE:    {format_float(float(metrics['tau_actuated_rmse']), 6)}",
        f"left GRF RMSE:        {format_float(float(metrics['left_grf_rmse']), 6)}",
        f"right GRF RMSE:       {format_float(float(metrics['right_grf_rmse']), 6)}",
        "",
        f"left contact F1:      {format_float(float(metrics['left_contact']['f1']), 4)}",
        f"right contact F1:     {format_float(float(metrics['right_contact']['f1']), 4)}",
    ]

    if report["mpjpe"] is not None:
        lines += [
            "",
            f"realtime MPJPE mean:  {format_float(float(np.mean(report['mpjpe'])), 6)} m",
            f"realtime MPJPE max:   {format_float(float(np.max(report['mpjpe'])), 6)} m",
        ]
    if report["dyn_residual"] is not None:
        lines.append(f"dyn residual mean:    {format_float(float(np.mean(report['dyn_residual'])), 6)}")
    if report["solve_time_ms"] is not None:
        lines.append(f"solve time mean [ms]: {format_float(float(np.mean(report['solve_time_ms'])), 3)}")
        lines.append(f"solve time p95  [ms]: {format_float(float(np.percentile(report['solve_time_ms'], 95.0)), 3)}")

    ax.text(0.02, 0.96, "\n".join(lines), va="top", ha="left", fontsize=11, family="monospace")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_overview_page(pdf: PdfPages, report: dict[str, object]) -> None:
    time_values = np.asarray(report["time"], dtype=float)
    q_frame_rmse = np.asarray(report["q_frame_rmse"], dtype=float)
    tau_frame_rmse = np.asarray(report["tau_frame_rmse"], dtype=float)
    total_force_offline = np.asarray(report["total_force_offline"], dtype=float)
    total_force_realtime = np.asarray(report["total_force_realtime"], dtype=float)
    left_contact_offline = np.asarray(report["left_contact_offline"], dtype=bool)
    right_contact_offline = np.asarray(report["right_contact_offline"], dtype=bool)
    left_contact_realtime = np.asarray(report["left_contact_realtime"], dtype=bool)
    right_contact_realtime = np.asarray(report["right_contact_realtime"], dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27), sharex="col")
    fig.suptitle("Overview", fontsize=16, fontweight="bold")

    axes[0, 0].plot(time_values, q_frame_rmse, color="#2563EB", linewidth=1.2)
    axes[0, 0].set_title("Per-frame q RMSE")
    axes[0, 0].set_ylabel("RMSE")
    axes[0, 0].set_ylim(*compute_smart_ylim(q_frame_rmse, min_span=smart_min_span("error_small")))
    axes[0, 0].grid(True, linestyle="--", alpha=0.25)

    axes[0, 1].plot(time_values, tau_frame_rmse, color="#B91C1C", linewidth=1.2)
    axes[0, 1].set_title("Per-frame tau RMSE")
    axes[0, 1].set_ylabel("RMSE")
    axes[0, 1].set_ylim(*compute_smart_ylim(tau_frame_rmse, min_span=smart_min_span("error_large")))
    axes[0, 1].grid(True, linestyle="--", alpha=0.25)

    plot_compare_signal(
        axes[1, 0],
        time_values,
        total_force_offline,
        total_force_realtime,
        "|GRF total| [N]",
        min_span=smart_min_span("force"),
    )
    axes[1, 0].set_title("Total GRF magnitude")

    axes[1, 1].step(time_values, left_contact_offline.astype(float), where="mid", color="#111827", linewidth=1.1, label="left offline")
    axes[1, 1].step(time_values, left_contact_realtime.astype(float), where="mid", color="#2563EB", linewidth=1.0, label="left realtime")
    axes[1, 1].step(time_values, 2.0 + right_contact_offline.astype(float), where="mid", color="#7C3AED", linewidth=1.1, label="right offline")
    axes[1, 1].step(time_values, 2.0 + right_contact_realtime.astype(float), where="mid", color="#DC2626", linewidth=1.0, label="right realtime")
    axes[1, 1].set_title("Contact states")
    axes[1, 1].set_yticks([0.0, 1.0, 2.0, 3.0])
    axes[1, 1].set_yticklabels(["L off", "L on", "R off", "R on"])
    axes[1, 1].grid(True, linestyle="--", alpha=0.25)
    axes[1, 1].legend(loc="upper right", fontsize=8)

    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_diagnostics_page(pdf: PdfPages, report: dict[str, object]) -> int:
    diagnostics = [
        ("Realtime MPJPE [m]", report["mpjpe"], "#2563EB"),
        ("Dynamics residual norm", report["dyn_residual"], "#B91C1C"),
        ("Solve time [ms]", report["solve_time_ms"], "#059669"),
    ]
    available = [(title, values, color) for title, values, color in diagnostics if values is not None]
    if not available:
        return 0

    time_values = np.asarray(report["time"], dtype=float)
    fig, axes = plt.subplots(len(available), 1, figsize=(11.69, 8.27), sharex=True)
    if len(available) == 1:
        axes = [axes]
    fig.suptitle("Realtime Diagnostics", fontsize=16, fontweight="bold")

    for ax, (title, values, color) in zip(axes, available):
        ax.plot(time_values, np.asarray(values, dtype=float), color=color, linewidth=1.2)
        ax.set_title(title, fontsize=10)
        if title == "Realtime MPJPE [m]":
            ax.set_ylim(*compute_smart_ylim(values, min_span=smart_min_span("error_small")))
        elif title == "Solve time [ms]":
            ax.set_ylim(*compute_smart_ylim(values, min_span=smart_min_span("time_ms")))
        else:
            ax.set_ylim(*compute_smart_ylim(values, min_span=smart_min_span("error_small")))
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[-1].set_xlabel("Time")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return 1


def add_dof_pages(pdf: PdfPages, report: dict[str, object], max_dofs: int | None = None) -> int:
    time_values = np.asarray(report["time"], dtype=float)
    dof_names = list(report["dof_names"])
    q_offline = np.asarray(report["q_offline"], dtype=float)
    q_realtime = np.asarray(report["q_realtime"], dtype=float)
    dq_offline = np.asarray(report["dq_offline"], dtype=float)
    dq_realtime = np.asarray(report["dq_realtime"], dtype=float)
    ddq_offline = np.asarray(report["ddq_offline"], dtype=float)
    ddq_realtime = np.asarray(report["ddq_realtime"], dtype=float)
    tau_offline = np.asarray(report["tau_offline"], dtype=float)
    tau_realtime = np.asarray(report["tau_realtime"], dtype=float)
    selected = dof_names if max_dofs is None else dof_names[: max(0, max_dofs)]

    contact_bg = contact_intervals(
        time_values,
        np.asarray(report["left_contact_offline"], dtype=bool) | np.asarray(report["right_contact_offline"], dtype=bool),
    )

    plotted = 0
    for idx, dof_name in enumerate(selected):
        pos_unit = "m" if is_translational_dof(dof_name) else "rad"
        vel_unit = "m/s" if is_translational_dof(dof_name) else "rad/s"
        acc_unit = "m/s^2" if is_translational_dof(dof_name) else "rad/s^2"
        tau_unit = "N" if is_translational_dof(dof_name) else "Nm"

        fig, axes = plt.subplots(4, 1, figsize=(11.69, 8.27), sharex=True)
        fig.suptitle(
            f"DOF: {dof_name}"
            f" | q={format_float(rmse(q_realtime[:, idx], q_offline[:, idx]), 4)}"
            f" dq={format_float(rmse(dq_realtime[:, idx], dq_offline[:, idx]), 4)}"
            f" ddq={format_float(rmse(ddq_realtime[:, idx], ddq_offline[:, idx]), 4)}"
            f" tau={format_float(rmse(tau_realtime[:, idx], tau_offline[:, idx]), 4)}",
            fontsize=14,
            fontweight="bold",
        )

        plot_compare_signal(
            axes[0],
            time_values,
            q_offline[:, idx],
            q_realtime[:, idx],
            f"q [{pos_unit}]",
            contact_background=contact_bg,
            min_span=smart_min_span("q", dof_name),
        )
        plot_compare_signal(
            axes[1],
            time_values,
            dq_offline[:, idx],
            dq_realtime[:, idx],
            f"dq [{vel_unit}]",
            contact_background=contact_bg,
            min_span=smart_min_span("dq", dof_name),
        )
        plot_compare_signal(
            axes[2],
            time_values,
            ddq_offline[:, idx],
            ddq_realtime[:, idx],
            f"ddq [{acc_unit}]",
            contact_background=contact_bg,
            min_span=smart_min_span("ddq", dof_name),
        )
        plot_compare_signal(
            axes[3],
            time_values,
            tau_offline[:, idx],
            tau_realtime[:, idx],
            f"tau [{tau_unit}]",
            contact_background=contact_bg,
            min_span=smart_min_span("tau", dof_name),
            center_on_zero=not is_translational_dof(dof_name),
        )
        axes[3].set_xlabel("Time")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        plotted += 1

    return plotted


def add_grf_pages(pdf: PdfPages, report: dict[str, object]) -> int:
    time_values = np.asarray(report["time"], dtype=float)
    page_count = 0

    for side, force_offline, force_realtime, contact_offline, contact_realtime in [
        (
            "left",
            np.asarray(report["left_force_offline"], dtype=float),
            np.asarray(report["left_force_realtime"], dtype=float),
            np.asarray(report["left_contact_offline"], dtype=bool),
            np.asarray(report["left_contact_realtime"], dtype=bool),
        ),
        (
            "right",
            np.asarray(report["right_force_offline"], dtype=float),
            np.asarray(report["right_force_realtime"], dtype=float),
            np.asarray(report["right_contact_offline"], dtype=bool),
            np.asarray(report["right_contact_realtime"], dtype=bool),
        ),
    ]:
        fig, axes = plt.subplots(5, 1, figsize=(11.69, 8.27), sharex=True)
        fig.suptitle(f"GRF / Contact: {side}", fontsize=15, fontweight="bold")
        bg = contact_intervals(time_values, contact_offline)
        plot_compare_signal(
            axes[0],
            time_values,
            force_offline[:, 0],
            force_realtime[:, 0],
            "Fx [N]",
            contact_background=bg,
            min_span=smart_min_span("force"),
            center_on_zero=True,
        )
        plot_compare_signal(
            axes[1],
            time_values,
            force_offline[:, 1],
            force_realtime[:, 1],
            "Fy [N]",
            contact_background=bg,
            min_span=smart_min_span("force"),
            center_on_zero=True,
        )
        plot_compare_signal(
            axes[2],
            time_values,
            force_offline[:, 2],
            force_realtime[:, 2],
            "Fz [N]",
            contact_background=bg,
            min_span=smart_min_span("force"),
        )
        plot_compare_signal(
            axes[3],
            time_values,
            np.linalg.norm(force_offline, axis=1),
            np.linalg.norm(force_realtime, axis=1),
            "|F| [N]",
            contact_background=bg,
            min_span=smart_min_span("force"),
        )
        plot_contact_compare(axes[4], time_values, contact_offline, contact_realtime, f"{side} contact")
        axes[4].set_xlabel("Time")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        page_count += 1

    return page_count


def build_pdf_report(report: dict[str, object], output_pdf: Path, max_dofs: int | None) -> dict[str, object]:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        add_title_page(pdf, report)
        add_overview_page(pdf, report)
        diagnostics_pages = add_diagnostics_page(pdf, report)
        dof_pages = add_dof_pages(pdf, report, max_dofs=max_dofs)
        grf_pages = add_grf_pages(pdf, report)

    return {
        "output_pdf": str(output_pdf),
        "frames": int(report["frames"]),
        "diagnostics_pages": diagnostics_pages,
        "dof_pages": dof_pages,
        "grf_pages": grf_pages,
    }


def main() -> int:
    args = parse_args()
    report = build_report(args)
    output_pdf = (
        args.output_pdf.resolve()
        if args.output_pdf is not None
        else default_output_pdf(args.offline_csv.resolve(), args.realtime_csv.resolve()).resolve()
    )
    summary = build_pdf_report(report, output_pdf=output_pdf, max_dofs=args.max_dofs)

    print("Realtime vs offline PDF report generated successfully.")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

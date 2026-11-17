#!/usr/bin/env python3
"""Create compact B1-B5 figures from the study diagnostic CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUBJECT_ORDER = [f"s{i:03d}" for i in range(1, 7)]


def plot_b1(diagnostics: Path, gate_csv: Path | None, output: Path) -> None:
    threshold = pd.read_csv(diagnostics / "B1_threshold_loso.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    if gate_csv is not None and gate_csv.is_file():
        gate = pd.read_csv(gate_csv).set_index("subject").reindex(SUBJECT_ORDER)
        x = np.arange(len(gate))
        axes[0].bar(x - 0.18, gate.false_contact_fraction_pooled, width=0.36, label="Osservato")
        axes[0].bar(x + 0.18, gate.expected_fraction, width=0.36, label="Atteso")
        axes[0].set_xticks(x, gate.index)
        axes[0].set_ylabel("Frazione di falso contatto in volo")
        axes[0].set_title("Gate pooled per soggetto")
        axes[0].legend(frameon=False)
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "Gate CSV non fornito", ha="center", va="center")
    threshold = threshold.set_index("heldout_subject").reindex(SUBJECT_ORDER)
    x = np.arange(len(threshold))
    axes[1].bar(x, threshold.test_contact_f1, color="#4477AA")
    axes[1].set_xticks(x, threshold.index)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("F1 contatto")
    axes[1].set_title("Soglie selezionate in LOSO")
    for index, method in enumerate(threshold.selected_method.fillna("")):
        axes[1].text(index, 0.03, method.replace("height_and_speed", "h+s"), rotation=90,
                     ha="center", va="bottom", fontsize=7, color="white")
    fig.savefig(output / "B1_false_contact_and_thresholds.png", dpi=180)
    plt.close(fig)

    roc = pd.read_csv(diagnostics / "B1_roc.csv")
    summary = roc.groupby(["method", "height_threshold_m", "speed_threshold_mps"], dropna=False)[
        ["false_positive_rate", "true_positive_rate"]
    ].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    for method, marker, color in (
        ("height", "o", "#4477AA"),
        ("speed", "s", "#EE6677"),
        ("height_and_speed", "x", "#228833"),
    ):
        rows = summary[summary.method == method]
        rows = rows.sort_values("height_threshold_m" if method == "height" else "speed_threshold_mps")
        ax.scatter(rows.false_positive_rate, rows.true_positive_rate, marker=marker,
                   color=color, label=method.replace("_", " "), s=36)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.8)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Falso positivo", ylabel="Vero positivo",
           title="ROC LOSO: altezza, velocità e soglia combinata")
    ax.legend(frameon=False)
    fig.savefig(output / "B1_threshold_roc_loso.png", dpi=180)
    plt.close(fig)


def plot_b2(diagnostics: Path, output: Path) -> None:
    frame = pd.read_csv(diagnostics / "B2_loso.csv")
    pivot = frame.pivot(index="heldout_subject", columns="model", values="rmse_left_share").reindex(SUBJECT_ORDER)
    ax = pivot.plot(kind="bar", figsize=(8, 4.6), color=["#228833", "#CCBB44", "#4477AA"])
    ax.set_ylabel("RMSE quota Fz sinistra")
    ax.set_xlabel("")
    ax.set_title("Ripartizione verticale in doppio appoggio")
    ax.legend(title="Modello", frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(output / "B2_vertical_split_loso.png", dpi=180)
    plt.close(ax.figure)


def plot_b3(diagnostics: Path, output: Path) -> None:
    frame = pd.read_csv(diagnostics / "B3_horizontal_h1.csv")
    summary = frame.groupby(["model", "phase"], sort=False).rmse_bw.agg(["mean", "std"]).reset_index()
    phases = ["total", "single_stance", "double_stance", "flight", "transition", "other"]
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    pivot = summary.pivot(index="phase", columns="model", values="mean").reindex(phases)
    pivot.plot(kind="bar", ax=ax, yerr=summary.pivot(index="phase", columns="model", values="std").reindex(phases),
               capsize=3, color=["#4477AA", "#EE6677"])
    ax.set_xticks(np.arange(len(phases)), phases, rotation=20, ha="right")
    ax.set_ylabel("RMSE forza orizzontale (BW)")
    ax.set_xlabel("")
    ax.set_title("Diagnostica H1 CoM e H2 momento angolare")
    ax.legend(title="Modello", frameon=False)
    fig.savefig(output / "B3_horizontal_h1_by_phase.png", dpi=180)
    plt.close(fig)

    opposition = pd.read_csv(diagnostics / "B3_opposing_force_recovery.csv")
    means = opposition.groupby("model")[
        ["opposing_force_recovery_ratio", "opposed_frame_recall"]
    ].mean().reindex(["H1_CoM_direction", "H2_angular_momentum_lstsq"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), constrained_layout=True)
    means.opposing_force_recovery_ratio.plot(kind="bar", ax=axes[0], color="#4477AA")
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Forza opposta predetta / misurata")
    axes[0].set_title("Ampiezza recuperata")
    means.opposed_frame_recall.plot(kind="bar", ax=axes[1], color="#228833")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Richiamo dei frame opposti")
    axes[1].set_title("Frame con forze opposte")
    for axis in axes:
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=20)
    fig.savefig(output / "B3_opposing_force_recovery.png", dpi=180)
    plt.close(fig)


def plot_b4(diagnostics: Path, output: Path) -> None:
    frame = pd.read_csv(diagnostics / "B4_cop_loso.csv")
    pivot = frame.pivot(index="heldout_subject", columns="model", values="mean_abs_cop_error_cm").reindex(SUBJECT_ORDER)
    ax = pivot.plot(kind="bar", figsize=(9, 4.6), color=["#4477AA", "#66CCEE", "#228833", "#CCBB44"])
    ax.set_ylabel("Errore assoluto medio CoP (cm)")
    ax.set_xlabel("")
    ax.set_title("Progressione CoP: confronto LOSO")
    ax.legend(title="Modello", frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(output / "B4_cop_progression_loso.png", dpi=180)
    plt.close(ax.figure)


def plot_b5(diagnostics: Path, output: Path) -> None:
    frame = pd.read_csv(diagnostics / "B5_lag.csv")
    frame = frame[(frame.phase == "total") & np.isfinite(frame.lag_ms_reference_later)]
    comparisons = list(frame.comparison.drop_duplicates())
    data = [frame.loc[frame.comparison == name, "lag_ms_reference_later"].to_numpy() for name in comparisons]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.boxplot(data, tick_labels=[name.replace("_", "\n") for name in comparisons], showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Lag migliore (ms; segno positivo = riferimento più tardivo)")
    ax.set_title("Lag temporale osservato, senza correzione")
    fig.savefig(output / "B5_lag_report_only.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate-csv", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_b1(args.diagnostics_dir, args.gate_csv, args.output_dir)
    plot_b2(args.diagnostics_dir, args.output_dir)
    plot_b3(args.diagnostics_dir, args.output_dir)
    plot_b4(args.diagnostics_dir, args.output_dir)
    plot_b5(args.diagnostics_dir, args.output_dir)
    print(f"figures_written=7 output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

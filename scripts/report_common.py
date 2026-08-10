from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opensim_batch_dynamics.metrics import (  # noqa: E402
    binary_classification_metrics,
    include_in_precision_metrics,
    mae,
    rmse,
)


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


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


def is_translational_dof(name: str) -> bool:
    return name.endswith("_tx") or name.endswith("_ty") or name.endswith("_tz")


def add_text_page(
    pdf,
    title: str,
    lines: list[str],
    *,
    fig_size: tuple[float, float] = (11.69, 8.27),
    title_fontsize: int = 18,
    title_weight: str = "bold",
    title_y: float = 0.98,
    body_x: float = 0.02,
    body_y: float = 0.95,
    body_fontsize: int = 11,
    body_family: str = "monospace",
) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=fig_size)
    fig.suptitle(title, fontsize=title_fontsize, fontweight=title_weight, y=title_y)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        body_x,
        body_y,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=body_fontsize,
        family=body_family,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(output_pdf: Path, page_callbacks: list[Callable[[object], object]]) -> list[object]:
    from matplotlib.backends.backend_pdf import PdfPages

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    results: list[object] = []
    with PdfPages(output_pdf) as pdf:
        for callback in page_callbacks:
            results.append(callback(pdf))
    return results

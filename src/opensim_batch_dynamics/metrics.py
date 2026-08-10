from __future__ import annotations

import numpy as np

METRIC_EXCLUDE_PREFIXES = (
    "ankle_angle_",
    "subtalar_angle_",
    "head_",
    "wrist_",
    "pro_sup_",
)


def include_in_precision_metrics(dof_name: str) -> bool:
    return not any(dof_name.startswith(prefix) for prefix in METRIC_EXCLUDE_PREFIXES)


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


def per_column_rmse(estimate: np.ndarray, reference: np.ndarray, names) -> list[tuple[str, float]]:
    error = np.array(estimate, dtype=float) - np.array(reference, dtype=float)
    values = np.sqrt(np.mean(error * error, axis=0))
    return [(str(name), float(value)) for name, value in zip(names, values)]


def top_k_rmse(estimate: np.ndarray, reference: np.ndarray, names, k: int = 8) -> list[tuple[str, float]]:
    ranked = per_column_rmse(estimate, reference, names)
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]

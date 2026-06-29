from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .bsm_assets import default_bsm_asset_paths


def _parse_marker_pairs(text: str) -> dict[str, int]:
    """Parse simple ``NAME: index`` YAML mappings while preserving order."""
    marker_map: dict[str, int] = {}
    for name, value in re.findall(r"([A-Za-z0-9_]+):\s*([0-9]+)", text):
        marker_map[name] = int(value)
    if not marker_map:
        raise ValueError("Could not parse any marker mappings from the BSM YAML text.")
    return marker_map


def load_bsm_marker_map(yaml_path: str | Path | None = None) -> dict[str, int]:
    """Load the BSM SMPL-X marker map from the repo asset or a YAML path."""
    path = Path(yaml_path) if yaml_path is not None else default_bsm_asset_paths().bsm_marker_yaml
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_marker_pairs(text)

    parsed = yaml.safe_load(text)
    if isinstance(parsed, dict):
        return {str(name): int(index) for name, index in parsed.items()}
    raise ValueError(f"Unsupported YAML structure in {path}")


def build_bsm_marker_positions(
    vertices: np.ndarray,
    marker_map: dict[str, int],
) -> tuple[np.ndarray, list[str]]:
    """Build a ``(T, M, 3)`` marker tensor from SMPL-X vertices."""
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError(f"Expected vertices shape (T, V, 3), got {vertices.shape}")

    marker_names = list(marker_map.keys())
    marker_positions = np.empty((vertices.shape[0], len(marker_names), 3), dtype=np.float32)
    num_vertices = int(vertices.shape[1])

    for marker_idx, marker_name in enumerate(marker_names):
        vertex_idx = int(marker_map[marker_name])
        if vertex_idx < 0 or vertex_idx >= num_vertices:
            raise IndexError(
                f"Marker '{marker_name}' references vertex {vertex_idx}, "
                f"but the mesh only has {num_vertices} vertices."
            )
        marker_positions[:, marker_idx, :] = vertices[:, vertex_idx, :]

    return marker_positions, marker_names

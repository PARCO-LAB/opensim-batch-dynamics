from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_DEFAULT_BSM_MARKERS_SMPLX_TEXT = """
# A dictionary of marker name and the corresponding smpl-X vertex id
BLTI: 3762 BRTI: 6520 C7: 3832 #spine
CHIN: 8943 CLAV: 5619 #clavicula
FBBT: 3752 FLTI: 4003 FRBT: 6510 FRTI: 6750
LAKI: 5753 LANK: 5881 #anklr
LBAC: 4452 LBHD: 1016 LBWT: 5675 LELB: 4293 #elbow
LELS: 5628 LELSO: 4251 LFBB: 3801 LFBT: 3592 LFFB: 3807 LFFT: 3541
LFHD: 797 LFIN: 4828 LFLB: 3652 LFLT: 3479 LFMB: 3658 LFMT: 3565
LFPI: 5779 LFRM: 4324 LFWT: 5727 #waist
LHAP: 3410 LHBA: 4020 LHEB: 8929 LHEE: 8846 #heel
LHFR: 3258 LHME: 5058 LHPI: 4696 LHTH: 4865 LHTO: 4139 LKNE: 3639
LKNI: 3781 LMT5: 5857 LSCA: 5462 LSHN: 3715 LSHO: 3878 LTIA: 3812
LTIB: 3706 LTIC: 5875 LTOE: 5770 LTOP: 5903 LTOS: 5904 LUMB: 5495
LUMC: 5941 LWRA: 4726 LWRB: 4722 NOSE: 8999
RAKI: 8447 RANK: 8575 RBAC: 7188 RBHD: 2374 RBWT: 8369 RELB: 7017
RELS: 8322 RELSO: 6995 RFBB: 6560 RFBT: 6291 RFFB: 6423 RFFT: 6301
RFHD: 2256 RFIN: 7564 RFLB: 6412 RFLT: 6240 RFMB: 6419 RFMT: 6831
RFPI: 8473 RFRM: 7060 RFWT: 8421 RHAP: 6171 RHBA: 6767 RHEB: 8714
RHEE: 8634 RHFR: 6020 RHME: 7794 RHPI: 7432 RHTH: 7601 RHTO: 8133
RIBL: 4118 RIBR: 8140 RITL: 3221 RITR: 5984 RKNE: 6400 RKNI: 6539
RMT5: 8551 RSCA: 6124 RSHN: 6485 RSHO: 6629 # shoulders
RTIA: 6569 RTIB: 6465 RTIC: 8569 RTOE: 5770 RTOP: 8597 RTOS: 8598
RWRA: 7462 #wrist
THD: 8969
"""


def _parse_marker_pairs(text: str) -> dict[str, int]:
    """Parse simple ``NAME: index`` YAML mappings while preserving order."""
    marker_map: dict[str, int] = {}
    for name, value in re.findall(r"([A-Za-z0-9_]+):\s*([0-9]+)", text):
        marker_map[name] = int(value)
    if not marker_map:
        raise ValueError("Could not parse any marker mappings from the BSM YAML text.")
    return marker_map


def load_bsm_marker_map(yaml_path: str | Path | None = None) -> dict[str, int]:
    """Load the BSM SMPL-X marker map from YAML, with a local fallback."""
    if yaml_path is not None:
        path = Path(yaml_path)
        if path.exists():
            try:
                import yaml  # type: ignore
            except ImportError:
                return _parse_marker_pairs(path.read_text(encoding="utf-8", errors="ignore"))

            parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return {str(name): int(index) for name, index in parsed.items()}
            raise ValueError(f"Unsupported YAML structure in {path}")

    return _parse_marker_pairs(_DEFAULT_BSM_MARKERS_SMPLX_TEXT)


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

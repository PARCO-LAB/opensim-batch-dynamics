from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetPaths:
    """Centralized paths used by the AMASS -> OpenSim pipeline."""

    repository_root: Path
    model_path: Path
    smplx_model_dir: Path
    opencap_assets_dir: Path
    opencap_scaling_setup: Path
    opencap_ik_setup: Path
    opencap_marker_set: Path
    opencap_vertex_map: Path

    def ensure_exists(self) -> None:
        """Fail early if one of the required assets is missing."""
        required = {
            "model_path": self.model_path,
            "smplx_model_dir": self.smplx_model_dir,
            "opencap_scaling_setup": self.opencap_scaling_setup,
            "opencap_ik_setup": self.opencap_ik_setup,
            "opencap_marker_set": self.opencap_marker_set,
            "opencap_vertex_map": self.opencap_vertex_map,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            joined = "\n".join(missing)
            raise FileNotFoundError(f"Missing required assets:\n{joined}")


def _resolve_repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return repo_root.resolve()
    return Path(__file__).resolve().parents[2]


def default_asset_paths(repo_root: Path | None = None) -> AssetPaths:
    """Return default repo-relative paths for the torque-only workflow."""
    root = _resolve_repo_root(repo_root)
    opencap_dir = root / "assets" / "opencap"
    return AssetPaths(
        repository_root=root,
        model_path=root / "model" / "LaiUhlrich2022_torque_only.osim",
        smplx_model_dir=root / "model" / "smpl",
        opencap_assets_dir=opencap_dir,
        opencap_scaling_setup=opencap_dir / "Scaling" / "Setup_scaling_LaiUhlrich2022_SMPL.xml",
        opencap_ik_setup=opencap_dir / "IK" / "Setup_IK_SMPL.xml",
        opencap_marker_set=opencap_dir / "Model" / "LaiUhlrich2022_markers_SMPL.xml",
        opencap_vertex_map=opencap_dir / "data" / "vertices_keypoints_corr.csv",
    )

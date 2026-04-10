from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BSMAssetPaths:
    """Repo-relative assets needed by the BSM pipeline."""

    repository_root: Path
    bsm_model_path: Path
    bsm_geometry_dir: Path
    bsm_marker_yaml: Path
    smplx_model_dir: Path

    def ensure_exists(self) -> None:
        """Fail early when the core BSM assets are missing."""
        required = {
            "bsm_model_path": self.bsm_model_path,
            "bsm_geometry_dir": self.bsm_geometry_dir,
            "smplx_model_dir": self.smplx_model_dir,
        }
        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required BSM assets:\n" + "\n".join(missing))


def _resolve_repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return repo_root.resolve()
    return Path(__file__).resolve().parents[2]


def default_bsm_asset_paths(
    repo_root: Path | None = None,
    smplx_model_dir: Path | None = None,
) -> BSMAssetPaths:
    """Return the default repo-local asset layout for the BSM workflow."""
    root = _resolve_repo_root(repo_root)
    smplx_dir = smplx_model_dir.resolve() if smplx_model_dir is not None else root / "model" / "smpl"
    return BSMAssetPaths(
        repository_root=root,
        bsm_model_path=root / "model" / "bsm" / "bsm.osim",
        bsm_geometry_dir=root / "model" / "bsm" / "Geometry",
        bsm_marker_yaml=root / "assets" / "smpl2ab" / "bsm_markers_smplx.yaml",
        smplx_model_dir=smplx_dir,
    )

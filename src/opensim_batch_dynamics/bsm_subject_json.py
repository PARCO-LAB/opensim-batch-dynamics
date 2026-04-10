from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .amass_loader import AMASSSequence


@dataclass(frozen=True)
class SubjectMeasurementEstimate:
    """Simple anthropometric estimate for AddBiomechanics subject JSON."""

    mass_kg: float
    height_m: float


def _normalize_sex(sex: str | None) -> str:
    sex = (sex or "neutral").lower()
    if sex in {"male", "female", "neutral"}:
        return sex
    return "neutral"


def _resolve_smplx_model_path(
    smplx_model_dir: str | Path,
    sex: str,
) -> Path:
    """Mirror the model resolution logic used by the SMPL-X forward pass."""
    model_dir = Path(smplx_model_dir).resolve()
    candidate_file = model_dir / f"SMPLX_{sex.upper()}.npz"
    neutral_candidate_file = model_dir / "SMPLX_NEUTRAL.npz"
    candidate_subdir = model_dir / "smplx"
    if candidate_file.exists():
        return candidate_file
    if neutral_candidate_file.exists():
        return neutral_candidate_file
    if candidate_subdir.exists():
        return candidate_subdir
    raise FileNotFoundError(
        "Could not find SMPL-X model files. Expected either:\n"
        f"- {candidate_file}\n"
        f"- {neutral_candidate_file}\n"
        f"- {candidate_subdir}/SMPLX_{sex.upper()}.npz"
    )


def estimate_subject_measurements(
    sequence: AMASSSequence,
    smplx_model_dir: str | Path,
    sex_override: str | None = None,
    device: str = "cpu",
) -> SubjectMeasurementEstimate:
    """
    Estimate mass and height from a neutral SMPL-X mesh.

    This follows the spirit of SMPL2AddBiomechanics but keeps the implementation
    lightweight and robust. If volume estimation is not available, we fall back
    to a conservative height-based mass heuristic.
    """
    try:
        import smplx
        import torch
    except ImportError as exc:
        raise RuntimeError("smplx and torch are required to estimate subject measurements.") from exc

    sex = _normalize_sex(sex_override or sequence.sex)
    model_path = _resolve_smplx_model_path(smplx_model_dir, sex)
    model = smplx.create(
        model_path=str(model_path),
        model_type="smplx",
        gender=sex,
        ext="npz",
        use_pca=False,
        flat_hand_mean=False,
        num_betas=min(16, int(sequence.betas.shape[0])),
        batch_size=1,
    ).to(device)

    betas = torch.as_tensor(sequence.betas[: min(16, sequence.betas.shape[0])], dtype=torch.float32, device=device).unsqueeze(0)
    zeros3 = torch.zeros((1, 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(
            global_orient=zeros3,
            body_pose=torch.zeros((1, sequence.body_pose.shape[1]), dtype=torch.float32, device=device),
            left_hand_pose=torch.zeros((1, sequence.left_hand_pose.shape[1]), dtype=torch.float32, device=device),
            right_hand_pose=torch.zeros((1, sequence.right_hand_pose.shape[1]), dtype=torch.float32, device=device),
            jaw_pose=zeros3,
            leye_pose=zeros3,
            reye_pose=zeros3,
            transl=zeros3,
            betas=betas,
            return_verts=True,
        )

    vertices = output.vertices.detach().cpu().numpy()[0]
    height_m = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))

    mass_kg = None
    try:
        import trimesh  # type: ignore

        mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces, process=False)
        if mesh.is_watertight and np.isfinite(mesh.volume):
            mass_kg = float(abs(mesh.volume) * 985.0)
    except Exception:
        mass_kg = None

    if mass_kg is None or not np.isfinite(mass_kg):
        mass_kg = float(np.clip(22.0 * height_m * height_m, 40.0, 140.0))

    return SubjectMeasurementEstimate(mass_kg=mass_kg, height_m=height_m)


def build_subject_json(
    sequence: AMASSSequence,
    smplx_model_dir: str | Path,
    sex_override: str | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Create the minimal AddBiomechanics subject JSON for a custom skeleton."""
    measurements = estimate_subject_measurements(
        sequence=sequence,
        smplx_model_dir=smplx_model_dir,
        sex_override=sex_override,
        device=device,
    )
    return {
        "sex": _normalize_sex(sex_override or sequence.sex),
        "massKg": float(measurements.mass_kg),
        "heightM": float(measurements.height_m),
        "subjectTags": ["AMASS", "SMPL-X", "BSM"],
        "skeletonPreset": "custom",
        "disableDynamics": True,
        "runMoco": False,
        "segmentTrials": False,
    }


def write_subject_json(
    output_path: str | Path,
    subject_json: dict[str, object],
) -> Path:
    """Write the AddBiomechanics subject JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subject_json, indent=2) + "\n", encoding="utf-8")
    return path

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .amass_loader import AMASSSequence


@dataclass
class SMPLXForwardResult:
    vertices: np.ndarray
    joints: np.ndarray


class SMPLXDependencyError(RuntimeError):
    """Raised when torch/smplx are not available."""

    pass


def _normalize_sex(sex: str) -> str:
    """Normalize user/AMASS sex labels to SMPL-X accepted values."""
    sex = sex.lower()
    if sex in {"male", "female", "neutral"}:
        return sex
    return "neutral"


def run_smplx_forward(
    sequence: AMASSSequence,
    smplx_model_dir: str | Path,
    sex_override: str | None = None,
    device: str = "cpu",
    num_betas: int = 16,
) -> SMPLXForwardResult:
    """Run SMPL-X forward pass and return vertices + joints for each frame."""
    try:
        import smplx
        import torch
    except ImportError as exc:
        raise SMPLXDependencyError(
            "smplx and torch are required for SMPL-X forward pass."
        ) from exc

    model_dir = Path(smplx_model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"SMPL-X model directory not found: {model_dir}")

    model_sex = _normalize_sex(sex_override or sequence.sex)
    n_frames = sequence.n_frames
    n_betas = min(int(num_betas), int(sequence.betas.shape[0]))

    candidate_file = model_dir / f"SMPLX_{model_sex.upper()}.npz"
    neutral_candidate_file = model_dir / "SMPLX_NEUTRAL.npz"
    candidate_subdir = model_dir / "smplx"
    if candidate_file.exists():
        model_path_for_smplx = candidate_file
    elif neutral_candidate_file.exists():
        # Fallback keeps the pipeline runnable even when only neutral model is shipped.
        model_sex = "neutral"
        model_path_for_smplx = neutral_candidate_file
    elif candidate_subdir.exists():
        model_path_for_smplx = candidate_subdir
    else:
        raise FileNotFoundError(
            "Could not find SMPL-X model files. Expected either:\n"
            f"- {candidate_file}\n"
            f"- {neutral_candidate_file}\n"
            f"- {candidate_subdir}/SMPLX_{model_sex.upper()}.npz"
        )

    model = smplx.create(
        model_path=str(model_path_for_smplx),
        model_type="smplx",
        gender=model_sex,
        ext="npz",
        use_pca=False,
        flat_hand_mean=False,
        num_betas=n_betas,
        batch_size=n_frames,
    ).to(device)

    t = torch.float32
    betas = torch.as_tensor(sequence.betas[:n_betas], dtype=t, device=device).unsqueeze(0)
    betas = betas.repeat(n_frames, 1)

    kwargs = {
        "global_orient": torch.as_tensor(sequence.root_orient, dtype=t, device=device),
        "body_pose": torch.as_tensor(sequence.body_pose, dtype=t, device=device),
        "left_hand_pose": torch.as_tensor(sequence.left_hand_pose, dtype=t, device=device),
        "right_hand_pose": torch.as_tensor(sequence.right_hand_pose, dtype=t, device=device),
        "jaw_pose": torch.as_tensor(sequence.jaw_pose, dtype=t, device=device),
        "leye_pose": torch.as_tensor(sequence.leye_pose, dtype=t, device=device),
        "reye_pose": torch.as_tensor(sequence.reye_pose, dtype=t, device=device),
        "transl": torch.as_tensor(sequence.trans, dtype=t, device=device),
        "betas": betas,
        "return_verts": True,
    }

    with torch.no_grad():
        output = model(**kwargs)

    vertices = output.vertices.detach().cpu().numpy().astype(np.float32)
    joints = output.joints.detach().cpu().numpy().astype(np.float32)
    return SMPLXForwardResult(vertices=vertices, joints=joints)

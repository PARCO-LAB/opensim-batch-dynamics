#!/usr/bin/env python3
"""Run the official GroundLinkNet checkpoint on GroundLink Stage-II files.

The official notebook writes tensors only.  This adapter writes the repository's
standard per-trial CSV, so ``validate_groundlink.py`` can score it unchanged.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def _load_model(repo: Path, checkpoint: Path):
    """Load GroundLink's vendored UnderPressure architecture and checkpoint."""
    sys.path.insert(0, str(repo / "UnderPressure"))
    import torch
    import models

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return torch, models.DeepNetwork(state_dict=state["model"]).eval()


def _positions(npz: Path, pelvis_offset: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]:
    """Reproduce GRF/scripts/preprocess.ipynb without creating .pth intermediates."""
    data = np.load(npz, allow_pickle=True)
    poses = np.asarray(data["poses"], dtype=np.float32).reshape(-1, 55, 3)
    trans = np.asarray(data["trans"], dtype=np.float32) + pelvis_offset
    rate = float(data["mocap_framerate"])
    angles = poses[:, :22]
    root_z = angles[:, 0, 2]
    cos, sin = np.cos(root_z), np.sin(root_z)
    rotation = np.zeros((len(angles), 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = rotation[:, 1, 1] = cos
    rotation[:, 0, 1] = -sin
    rotation[:, 1, 0] = sin
    rotation[:, 2, 2] = 1.0
    origin = trans.copy()
    origin[:, 2] = 0.0
    # inverse(T) * pelvis: only its z offset remains, as in the official notebook.
    pelvis_local = np.einsum("tij,tj->ti", np.swapaxes(rotation, 1, 2), trans - origin)
    return np.concatenate((pelvis_local[:, None], angles), axis=1), (rotation, origin), rate


def _to_global_cop(local_cop: np.ndarray, transform: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    rotation, origin = transform
    return np.einsum("tij,tfj->tfi", rotation, local_cop) + origin[:, None, :]


def _write_csv(path: Path, prediction: np.ndarray, transform: tuple[np.ndarray, np.ndarray], rate: float, mass_kg: float, height_m: float) -> None:
    # GroundLinkNet output is [CoP (m), GRF (kN)] for left/right feet.
    cop = _to_global_cop(prediction[:, :, :3], transform)
    grf = prediction[:, :, 3:] * 1000.0
    fields = ["time", "subject_mass_kg", "subject_height_m"]
    for side in ("l", "r"):
        fields += [f"calcn_{side}_{kind}_{axis}" for kind in ("cop", "grf") for axis in "xyz"]
        fields += [f"calcn_{side}_contact"]
        fields += [f"toes_{side}_{kind}_{axis}" for kind in ("cop", "grf") for axis in "xyz"]
        fields += [f"toes_{side}_contact"]
    fields += [f"grf_total_{axis}" for axis in "xyz"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(len(prediction)):
            row: dict[str, float] = {"time": i / rate, "subject_mass_kg": mass_kg, "subject_height_m": height_m}
            for foot, side in enumerate(("l", "r")):
                for axis, value in zip("xyz", cop[i, foot]): row[f"calcn_{side}_cop_{axis}"] = float(value)
                for axis, value in zip("xyz", grf[i, foot]): row[f"calcn_{side}_grf_{axis}"] = float(value)
                row[f"calcn_{side}_contact"] = float(grf[i, foot, 2] > mass_kg * 9.81 * 0.05)
                for axis in "xyz":
                    row[f"toes_{side}_cop_{axis}"] = float("nan")
                    row[f"toes_{side}_grf_{axis}"] = 0.0
                row[f"toes_{side}_contact"] = 0.0
            for axis, value in zip("xyz", grf[i].sum(axis=0)): row[f"grf_total_{axis}"] = float(value)
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mass-kg", type=float, required=True)
    parser.add_argument("--height-m", type=float, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/tmp/GroundLink-official"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--smplx-model-dir", type=Path, default=Path("model/smpl"))
    args = parser.parse_args()
    checkpoint = args.checkpoint or args.repo / "GRF/checkpoint/pretrained_s7_noshape.tar"
    if not checkpoint.exists():
        parser.error(f"Checkpoint not found: {checkpoint}")
    motion = np.load(args.input, allow_pickle=True)
    sex = str(motion["gender"]).upper()
    pelvis_file = args.smplx_model_dir / f"SMPLX_{sex}.npz"
    if not pelvis_file.exists():
        pelvis_file = args.smplx_model_dir / "SMPLX_NEUTRAL.npz"
    pelvis_offset = np.load(pelvis_file)["J"][0].astype(np.float32)
    positions, transform, rate = _positions(args.input, pelvis_offset)
    torch, model = _load_model(args.repo, checkpoint)
    with torch.no_grad():
        prediction = model.GRFs(torch.from_numpy(positions).unsqueeze(0)).squeeze(0).cpu().numpy()
    _write_csv(args.output, prediction, transform, rate, args.mass_kg, args.height_m)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

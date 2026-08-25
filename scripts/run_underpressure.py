#!/usr/bin/env python3
"""Run the official UnderPressure checkpoint on one GroundLink Stage-II trial.

UnderPressure predicts plantar vertical-force cells, not 3-D GRF or CoP.  The
CSV therefore contains only vertical GRF; its valid comparison metrics are Fz
and contact.  Retargeting is the official AMASS demo procedure.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opensim_batch_dynamics.amass_loader import load_amass_npz
from opensim_batch_dynamics.smplx_forward import run_smplx_forward


AMASS_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine_1", "left_knee", "right_knee", "spine_2",
    "left_ankle", "right_ankle", "neck", "left_foot", "right_foot", "head",
    "left_clavicle", "right_clavicle", "head_top", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_finger_middle_3",
    "left_finger_thumb_3", "right_finger_middle_3", "right_finger_thumb_3",
]
# SMPL-X's first 22 body joints; duplicated tips are deliberate, as in the
# official demo only the joints shared with UnderPressure's topology are used.
SMPLX_IDXS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 10, 11, 15, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20, 20, 21, 21]


def _load_official(repo: Path, checkpoint: Path):
    sys.path.insert(0, str(repo))
    import torch
    import anim, models, util
    from data import FRAMERATE, TOPOLOGY

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return torch, anim, util, FRAMERATE, TOPOLOGY, models.DeepNetwork(state_dict=state["model"]).eval()


def _retarget(torch, anim, util, topology, joints, skeleton, iterations: int):
    """Official demo.py retarget_to_underpressure, kept local to avoid viewer imports."""
    names = [name for name in topology if name in AMASS_NAMES]
    target = joints[:, [AMASS_NAMES.index(name) for name in names]]
    target_ids = [topology.index(name) for name in names]
    frames = target.shape[0]
    angles = torch.nn.Parameter(util.SU2.identity(frames, len(topology)).to(target))
    trajectory = torch.nn.Parameter(joints[:, [0]].clone())
    translate = torch.nn.Parameter(torch.zeros(1, 1, 3, dtype=target.dtype))
    scale = torch.nn.Parameter(torch.ones(1, 1, 1, dtype=target.dtype))
    optimizer = torch.optim.Adam([angles, trajectory, translate, scale], lr=1e-1)
    skeleton = skeleton.to(target)
    p_weight = 1 / (skeleton[..., 2].amax(dim=-1) - skeleton[..., 2].amin(dim=-1)).mean().square()
    for _ in range(iterations):
        positions = anim.FK(util.SU2.normalize(angles), skeleton, None, topology)[:, target_ids]
        positions = scale * positions + trajectory + translate
        loss = p_weight * (target - positions).norm(p=2, dim=-1).square().mean() + 1e-3 * (angles.norm(p=2, dim=-1) - 1).square().mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad()
    return util.SU2.normalize(angles.detach()), ((trajectory + translate) / scale).detach()


def _write_csv(path: Path, fz: np.ndarray, rate: float, mass_kg: float, height_m: float) -> None:
    fields = ["time", "subject_mass_kg", "subject_height_m"]
    for side in ("l", "r"):
        fields += [f"calcn_{side}_{kind}_{axis}" for kind in ("cop", "grf") for axis in "xyz"] + [f"calcn_{side}_contact"]
        fields += [f"toes_{side}_{kind}_{axis}" for kind in ("cop", "grf") for axis in "xyz"] + [f"toes_{side}_contact"]
    fields += [f"grf_total_{axis}" for axis in "xyz"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i, forces in enumerate(fz):
            row: dict[str, float] = {"time": i / rate, "subject_mass_kg": mass_kg, "subject_height_m": height_m}
            for foot, side in enumerate(("l", "r")):
                for axis in "xyz":
                    row[f"calcn_{side}_cop_{axis}"] = float("nan")
                    row[f"calcn_{side}_grf_{axis}"] = float(forces[foot]) if axis == "z" else 0.0
                    row[f"toes_{side}_cop_{axis}"] = float("nan"); row[f"toes_{side}_grf_{axis}"] = 0.0
                row[f"calcn_{side}_contact"] = float(forces[foot] > .05 * mass_kg * 9.81); row[f"toes_{side}_contact"] = 0.0
            row.update({"grf_total_x": 0.0, "grf_total_y": 0.0, "grf_total_z": float(forces.sum())})
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mass-kg", type=float, required=True); parser.add_argument("--height-m", type=float, required=True)
    parser.add_argument("--repo", type=Path, default=Path("/tmp/UnderPressure-official")); parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--smplx-model-dir", type=Path, default=Path("model/smpl")); parser.add_argument("--retarget-iters", type=int, default=150)
    args = parser.parse_args(); checkpoint = args.checkpoint or args.repo / "pretrained.tar"
    torch, anim, util, target_rate, topology, model = _load_official(args.repo, checkpoint)
    sequence = load_amass_npz(args.input)
    # UnderPressure operates at 100 Hz; avoid computing unused 250-Hz vertices.
    indexes = np.unique(np.rint(np.arange(round(sequence.n_frames / sequence.frame_rate_hz * target_rate)) * sequence.frame_rate_hz / target_rate).astype(int))
    indexes = indexes[indexes < sequence.n_frames]
    sequence_100hz = replace(sequence, frame_rate_hz=float(target_rate), trans=sequence.trans[indexes], root_orient=sequence.root_orient[indexes], body_pose=sequence.body_pose[indexes], left_hand_pose=sequence.left_hand_pose[indexes], right_hand_pose=sequence.right_hand_pose[indexes], jaw_pose=sequence.jaw_pose[indexes], leye_pose=sequence.leye_pose[indexes], reye_pose=sequence.reye_pose[indexes])
    smplx = run_smplx_forward(sequence_100hz, args.smplx_model_dir, return_vertices=False)
    joints = torch.from_numpy(smplx.joints[:, SMPLX_IDXS]).float()
    sample = torch.load(args.repo / "footskate_samples/0.pt", map_location="cpu", weights_only=False)
    angles, trajectory = _retarget(torch, anim, util, topology, joints, sample["skeleton"], args.retarget_iters)
    out_frames = len(joints)
    angles = util.resample(angles, out_frames, dim=-3, interpolation_fn=util.SU2.slerp)
    trajectory = util.resample(trajectory, out_frames)
    with torch.no_grad():
        positions = anim.FK(angles, sample["skeleton"], trajectory, topology)
        fz = model.vGRFs(positions.unsqueeze(0)).squeeze(0).sum(dim=-1).cpu().numpy() * args.mass_kg * 9.81
    source_t = np.arange(len(fz)) / target_rate; target_t = np.arange(sequence.n_frames) / sequence.frame_rate_hz
    upsampled = np.column_stack([np.interp(target_t, source_t, fz[:, side]) for side in range(2)])
    _write_csv(args.output, upsampled, sequence.frame_rate_hz, args.mass_kg, args.height_m)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

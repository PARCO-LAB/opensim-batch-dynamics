#!/usr/bin/env python3
"""Export AMASS sequence(s) as frame-wise 3D OpenPose/COCO-18 CSV files."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import sys
from pathlib import Path

import numpy as np


# SMPL-X output order: 55 body/hand joints followed by five face vertices
# (nose, right eye, left eye, right ear, left ear).
COCO18 = (
    ("nose", 55),
    ("neck", 12),
    ("right_shoulder", 17),
    ("right_elbow", 19),
    ("right_wrist", 21),
    ("left_shoulder", 16),
    ("left_elbow", 18),
    ("left_wrist", 20),
    ("right_hip", 2),
    ("right_knee", 5),
    ("right_ankle", 8),
    ("left_hip", 1),
    ("left_knee", 4),
    ("left_ankle", 7),
    ("right_eye", 56),
    ("left_eye", 57),
    ("right_ear", 58),
    ("left_ear", 59),
)


def _add_src_to_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert AMASS .npz file(s) to 3D COCO-18 keypoint CSV files."
    )
    parser.add_argument(
        "--input", required=True, help="AMASS .npz file or directory (searched recursively)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV path for one file, or output directory for a directory input",
    )
    parser.add_argument(
        "--smplx-model-dir",
        default=None,
        help="SMPL-X model directory (default: repository model/smpl)",
    )
    parser.add_argument(
        "--sex",
        choices=("neutral", "male", "female"),
        default=None,
        help="Optional sex override; otherwise use AMASS metadata",
    )
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes for directory input (default: 1)",
    )
    return parser


def _extract_coco18(joints: np.ndarray) -> np.ndarray:
    """Select COCO-18 points from batched SMPL-X joints."""
    joints = np.asarray(joints)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"Expected SMPL-X joints with shape (T, J, 3), got {joints.shape}")
    indices = [index for _, index in COCO18]
    if joints.shape[1] <= max(indices):
        raise ValueError(
            f"SMPL-X output has {joints.shape[1]} joints; at least {max(indices) + 1} are required"
        )
    selected = joints[:, indices, :].astype(np.float32, copy=False)
    if not np.isfinite(selected).all():
        raise ValueError("SMPL-X output contains non-finite COCO-18 coordinates")
    return selected


def write_coco18_csv(
    keypoints: np.ndarray,
    frame_rate_hz: float,
    output_path: str | Path,
) -> Path:
    """Write keypoints shaped (T, 18, 3) as one row per frame."""
    if frame_rate_hz <= 0:
        raise ValueError(f"frame_rate_hz must be positive, got {frame_rate_hz}")
    keypoints = np.asarray(keypoints)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (len(COCO18), 3):
        raise ValueError(f"Expected keypoints shape (T, 18, 3), got {keypoints.shape}")

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["frame", "time"] + [
        f"{name}_{axis}" for name, _ in COCO18 for axis in ("x", "y", "z")
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame, points in enumerate(keypoints):
            writer.writerow([frame, frame / frame_rate_hz, *points.reshape(-1).tolist()])
    return output


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    model_dir: str | Path,
    sex: str | None,
    device: str,
) -> tuple[Path, int]:
    _add_src_to_path()
    from opensim_batch_dynamics.amass_loader import load_amass_npz
    from opensim_batch_dynamics.smplx_forward import run_smplx_forward

    sequence = load_amass_npz(input_path)
    if sex:
        sequence = sequence.copy_with_gender(sex)
    forward = run_smplx_forward(
        sequence=sequence,
        smplx_model_dir=model_dir,
        device=device,
        sex_override=sex,
        return_vertices=False,
    )
    output = write_coco18_csv(
        _extract_coco18(forward.joints), sequence.frame_rate_hz, output_path
    )
    return output, sequence.n_frames


def _amass_motion_file(path: Path) -> tuple[bool, str]:
    """Identify AMASS motion files before submitting batch workers."""
    try:
        with np.load(path, allow_pickle=True) as data:
            fields = set(data.files)
    except Exception as exc:
        return False, f"cannot read npz ({exc})"

    if "trials" in fields:
        return True, ""
    names = {field.rsplit("/", 1)[-1] for field in fields}
    if "trans" not in names:
        return False, "missing 'trans'"
    if not ({"mocap_frame_rate", "mocap_framerate"} & names):
        return False, "missing 'mocap_frame_rate'/'mocap_framerate'"
    return True, ""


def main() -> int:
    _add_src_to_path()
    args = build_arg_parser().parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be at least 1, got {args.workers}")
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = (
        Path(args.smplx_model_dir)
        if args.smplx_model_dir
        else repo_root / "model" / "smpl"
    )

    input_path = Path(args.input).expanduser().resolve()
    if input_path.is_dir():
        input_files = sorted(input_path.rglob("*.npz"))
        if not input_files:
            raise FileNotFoundError(f"No .npz files found under {input_path}")
        motion_files = []
        for input_file in input_files:
            is_motion, reason = _amass_motion_file(input_file)
            if is_motion:
                motion_files.append(input_file)
            else:
                print(f"Skipping {input_file}: {reason}", file=sys.stderr)
        if not motion_files:
            raise FileNotFoundError(f"No AMASS motion .npz files found under {input_path}")
        output_root = Path(args.output or "outputs/coco18")
        if output_root.suffix.lower() == ".csv":
            raise ValueError("For directory input, --output must be a directory, not a .csv file")
        output_files = [
            output_root / input_file.relative_to(input_path).with_suffix(".csv")
            for input_file in motion_files
        ]
        jobs = zip(motion_files, output_files)
        if args.workers == 1:
            results = (
                convert_file(input_file, output_file, model_dir, args.sex, args.device)
                for input_file, output_file in jobs
            )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                results = executor.map(
                    convert_file,
                    motion_files,
                    output_files,
                    repeat(model_dir),
                    repeat(args.sex),
                    repeat(args.device),
                )
        for output, frames in results:
            print(f"Wrote {frames} frames to {output}")
        return 0

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file or directory not found: {input_path}")
    is_motion, reason = _amass_motion_file(input_path)
    if not is_motion:
        raise ValueError(f"Input is not an AMASS motion file: {input_path} ({reason})")
    output = args.output or "outputs/coco18.csv"
    output_path, frames = convert_file(input_path, output, model_dir, args.sex, args.device)
    print(f"Wrote {frames} frames to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

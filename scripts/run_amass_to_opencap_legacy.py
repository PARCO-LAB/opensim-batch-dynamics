#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    # Keep the script runnable without package installation.
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the legacy OpenCap/OpenSim pipeline: AMASS npz -> OpenSim .mot + scaled .osim + DOF CSV."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to AMASS npz file (example: data/A3-_Swing_arms_stageii.npz)",
    )
    parser.add_argument("--trial", required=True, help="Trial name for output folders/files")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output root directory (default: outputs)",
    )
    parser.add_argument("--mass-kg", type=float, default=75.0, help="Subject mass in kg")
    parser.add_argument("--height-m", type=float, default=1.75, help="Subject height in meters")
    parser.add_argument(
        "--sex",
        choices=["neutral", "male", "female"],
        default="neutral",
        help="Subject sex for SMPL-X model selection: neutral|male|female",
    )
    parser.add_argument(
        "--assets-dir",
        default="assets/opencap",
        help="Path to assets/opencap directory used by the legacy OpenCap pipeline",
    )
    parser.add_argument(
        "--model-path",
        default="model/LaiUhlrich2022_torque_only.osim",
        help="Path to OpenSim torque-only model",
    )
    parser.add_argument(
        "--smplx-model-dir",
        default="model/smpl",
        help="Path to directory containing SMPLX_*.npz files",
    )
    parser.add_argument(
        "--rotation-y",
        type=float,
        default=90.0,
        help="Y-axis rotation in degrees from SMPL-X to OpenSim frame",
    )
    parser.add_argument(
        "--no-vertical-offset",
        action="store_true",
        help="Disable automatic vertical offset before TRC export",
    )
    parser.add_argument(
        "--skip-scale",
        action="store_true",
        help="Skip scaling and run IK directly on --model-path",
    )
    parser.add_argument(
        "--skip-ik",
        action="store_true",
        help="Skip IK execution and only generate TRC (and optional scaling).",
    )
    parser.add_argument(
        "--trc-only",
        action="store_true",
        help="Generate trial and neutral TRC only (equivalent to --skip-scale --skip-ik).",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Skip .mot -> CSV conversion.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Output CSV path. Default: alongside .mot with suffix '_dofs.csv'.",
    )
    parser.add_argument(
        "--csv-model-path",
        default=None,
        help=(
            "Model used to order CSV DOF columns. "
            "Default: scaled model (if available) else --model-path."
        ),
    )
    parser.add_argument(
        "--missing-fill",
        default="nan",
        help="Fill value for DOFs missing in .mot (default: nan).",
    )
    parser.add_argument(
        "--filter-mode",
        choices=["auto", "walking", "dynamic", "none"],
        default="auto",
        help=(
            "Low-pass filter mode for DOF signals before derivatives: "
            "walking=12Hz, dynamic=30Hz, auto=heuristic, none=disabled."
        ),
    )
    parser.add_argument(
        "--filter-cutoff-hz",
        type=float,
        default=None,
        help="Override low-pass cutoff frequency in Hz (4th-order Butterworth).",
    )
    parser.add_argument(
        "--no-velocity-columns",
        action="store_true",
        help="Do not add '<dof>_vel' columns to CSV.",
    )
    parser.add_argument(
        "--no-acceleration-columns",
        action="store_true",
        help="Do not add '<dof>_acc' columns to CSV.",
    )
    parser.add_argument(
        "--no-time-column",
        action="store_true",
        help="Do not include 'time' column in CSV.",
    )
    parser.add_argument(
        "--no-frame-column",
        action="store_true",
        help="Do not include 'frame' column in CSV.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for SMPL-X forward (default: cpu)",
    )
    return parser


def main() -> int:
    _add_src_to_path()
    from opensim_batch_dynamics.config import AssetPaths, default_asset_paths
    from opensim_batch_dynamics.mot_to_csv import convert_mot_to_model_csv, parse_missing_fill

    parser = build_arg_parser()
    args = parser.parse_args()
    from opensim_batch_dynamics.opensim_pipeline import run_amass_to_opensim

    repo_root = Path(__file__).resolve().parents[1]
    defaults = default_asset_paths(repo_root=repo_root)
    assets_dir = Path(args.assets_dir).resolve()
    assets = AssetPaths(
        repository_root=repo_root,
        model_path=Path(args.model_path).resolve(),
        smplx_model_dir=Path(args.smplx_model_dir).resolve(),
        opencap_assets_dir=assets_dir,
        opencap_scaling_setup=assets_dir / "Scaling" / "Setup_scaling_LaiUhlrich2022_SMPL.xml",
        opencap_ik_setup=assets_dir / "IK" / "Setup_IK_SMPL.xml",
        opencap_marker_set=assets_dir / "Model" / "LaiUhlrich2022_markers_SMPL.xml",
        opencap_vertex_map=assets_dir / "data" / "vertices_keypoints_corr.csv",
    )
    # If the provided model path does not exist, fallback to repo default torque model.
    if not assets.model_path.exists():
        assets = AssetPaths(
            repository_root=assets.repository_root,
            model_path=defaults.model_path,
            smplx_model_dir=assets.smplx_model_dir,
            opencap_assets_dir=assets.opencap_assets_dir,
            opencap_scaling_setup=assets.opencap_scaling_setup,
            opencap_ik_setup=assets.opencap_ik_setup,
            opencap_marker_set=assets.opencap_marker_set,
            opencap_vertex_map=assets.opencap_vertex_map,
        )

    skip_scale = bool(args.skip_scale or args.trc_only)
    skip_ik = bool(args.skip_ik or args.trc_only)

    outputs = run_amass_to_opensim(
        input_npz_path=Path(args.input),
        trial_name=args.trial,
        output_dir=Path(args.output_dir),
        mass_kg=args.mass_kg,
        height_m=args.height_m,
        sex=args.sex,
        assets=assets,
        model_path=assets.model_path,
        rotations_deg={"x": 0.0, "y": float(args.rotation_y), "z": 0.0},
        apply_vertical_offset=not args.no_vertical_offset,
        skip_scale=skip_scale,
        skip_ik=skip_ik,
        device=args.device,
    )

    # Optional final step: export IK motion (.mot) to frame-by-frame DOF CSV.
    csv_path: Path | None = None
    csv_summary: dict[str, object] | None = None
    if not args.skip_csv and outputs.ik_motion_path is not None:
        if args.csv_path:
            csv_path = Path(args.csv_path).resolve()
        else:
            csv_path = outputs.ik_motion_path.with_name(f"{outputs.ik_motion_path.stem}_dofs.csv")

        csv_model_path = (
            Path(args.csv_model_path).resolve()
            if args.csv_model_path
            else (outputs.scaled_model_path or assets.model_path)
        )
        summary = convert_mot_to_model_csv(
            mot_path=outputs.ik_motion_path,
            model_path=csv_model_path,
            out_csv_path=csv_path,
            missing_fill=parse_missing_fill(args.missing_fill),
            add_velocity=not bool(args.no_velocity_columns),
            add_acceleration=not bool(args.no_acceleration_columns),
            filter_mode=str(args.filter_mode),
            cutoff_hz=args.filter_cutoff_hz,
            include_time=not bool(args.no_time_column),
            include_frame=not bool(args.no_frame_column),
        )
        csv_summary = {
            "input_rows": summary.input_rows,
            "model_dofs": summary.model_dofs,
            "mapped_dofs": summary.mapped_dofs,
            "missing_dofs": list(summary.missing_dofs),
            "filter_cutoff_hz": summary.filter_cutoff_hz,
            "sample_rate_hz": summary.sample_rate_hz,
            "csv_model_path": str(csv_model_path),
        }

    payload = {
        "trial_trc_path": str(outputs.trial_trc_path),
        "neutral_trc_path": str(outputs.neutral_trc_path),
        "scaled_model_path": str(outputs.scaled_model_path) if outputs.scaled_model_path else None,
        "ik_motion_path": str(outputs.ik_motion_path) if outputs.ik_motion_path else None,
        "scale_setup_path": str(outputs.scale_setup_path) if outputs.scale_setup_path else None,
        "ik_setup_path": str(outputs.ik_setup_path) if outputs.ik_setup_path else None,
        "csv_path": str(csv_path) if csv_path else None,
        "csv_summary": csv_summary,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

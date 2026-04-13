#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

# On macOS, torch/nimble/OpenSim stacks can load multiple OpenMP runtimes.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the AMASS -> SMPL-X -> BSM -> AddBiomechanics -> CSV pipeline."
    )
    parser.add_argument("--input", required=True, help="Path to AMASS SMPL-X npz file")
    parser.add_argument(
        "--trial",
        required=True,
        help="Trial name used for outputs, or 'all' to export one CSV per trial found in the input npz",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/bsm",
        help="Output root directory (default: outputs/bsm)",
    )
    parser.add_argument(
        "--smplx-model-dir",
        default="model/smpl",
        help="Path to SMPL-X model directory",
    )
    parser.add_argument(
        "--bsm-model",
        default="model/bsm/bsm.osim",
        help="Path to the unscaled BSM OpenSim model",
    )
    parser.add_argument(
        "--addbio-root",
        default=None,
        help="Path to the local AddBiomechanics checkout or use $ADDBIO_ENGINE_ROOT",
    )
    parser.add_argument(
        "--sex",
        choices=["neutral", "male", "female"],
        default=None,
        help="Optional sex override for SMPL-X and AddBiomechanics metadata",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for SMPL-X forward and body measurement estimation",
    )
    parser.add_argument(
        "--skip-inverse-dynamics",
        action="store_true",
        help="Skip inverse dynamics (no-GRF) torque export",
    )
    parser.add_argument(
        "--id-filter-mode",
        default="auto",
        choices=["auto", "walking", "dynamic", "none"],
        help="Low-pass filter preset for inverse dynamics coordinates (default: auto)",
    )
    parser.add_argument(
        "--id-cutoff-hz",
        type=float,
        default=None,
        help="Optional explicit low-pass cutoff (Hz) for inverse dynamics coordinates",
    )
    parser.add_argument(
        "--id-grf-mode",
        default="estimated",
        choices=["estimated", "none"],
        help="Inverse dynamics external force mode: estimated contact GRF or none (default: estimated)",
    )
    parser.add_argument(
        "--id-contact-bodies",
        default="all",
        help=(
            "Comma-separated contact body names used to estimate GRF. "
            "Use 'all' (default) to consider all body nodes."
        ),
    )
    parser.add_argument(
        "--id-friction-coeff",
        type=float,
        default=0.8,
        help="Friction coefficient for GRF estimation (default: 0.8)",
    )
    parser.add_argument(
        "--id-contact-height-threshold-m",
        type=float,
        default=0.06,
        help="Contact activation height threshold above floor in meters (default: 0.06)",
    )
    parser.add_argument(
        "--id-contact-speed-threshold-mps",
        type=float,
        default=0.6,
        help="Contact activation normal speed threshold in m/s (default: 0.6)",
    )
    parser.add_argument(
        "--final-csv-path",
        default=None,
        help=(
            "Optional output path for the final unified CSV "
            "(default: <output-dir>/<trial>.csv)"
        ),
    )
    parser.add_argument(
        "--cleanup-intermediate",
        action="store_true",
        help=(
            "Remove all intermediate files/folders after successful run, "
            "keeping only the final unified CSV."
        ),
    )
    return parser


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _sanitize_trial_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or fallback


def _resolve_all_mode_final_csv_root(args: argparse.Namespace, output_root: Path) -> Path:
    if args.final_csv_path is None:
        return output_root
    custom_path = Path(args.final_csv_path).resolve()
    if custom_path.suffix.lower() == ".csv":
        raise ValueError(
            "When using --trial all, --final-csv-path must be a directory, not a .csv file."
        )
    custom_path.mkdir(parents=True, exist_ok=True)
    return custom_path


def _parse_contact_body_names(raw_names: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in raw_names.split(",") if name.strip())
    if not names:
        return ("all",)
    return names


def _fallback_contact_bodies_for_csv(contact_body_names: tuple[str, ...]) -> tuple[str, ...]:
    """
    Build safe fallback names for final CSV export.

    We avoid sentinel names like 'all'/'auto' because they are not actual body
    names and would create invalid columns when GRF/contact CSV is unavailable.
    """
    filtered = tuple(
        name
        for name in contact_body_names
        if name.lower() not in {"all", "auto", "*"}
    )
    if filtered:
        return filtered
    return ("calcn_l", "calcn_r")


def _run_single_trial_pipeline(
    args: argparse.Namespace,
    sequence,
    trial_name: str,
    final_csv_path: Path | None = None,
) -> dict[str, object]:
    from opensim_batch_dynamics.addbio_csv_export import export_addbiomechanics_csv
    from opensim_batch_dynamics.addbio_runner import (
        result_to_json,
        run_addbiomechanics_engine,
    )
    from opensim_batch_dynamics.addbio_subject_folder import (
        build_addbiomechanics_subject_folder,
    )
    from opensim_batch_dynamics.bsm_assets import default_bsm_asset_paths
    from opensim_batch_dynamics.bsm_markers import (
        build_bsm_marker_positions,
        load_bsm_marker_map,
    )
    from opensim_batch_dynamics.bsm_subject_json import build_subject_json
    from opensim_batch_dynamics.smplx_forward import run_smplx_forward
    from opensim_batch_dynamics.trc_export import write_trc
    from opensim_batch_dynamics.inverse_dynamics_no_grf import (
        run_inverse_dynamics_and_export_torque_csv,
    )
    from opensim_batch_dynamics.final_csv_export import export_final_csv

    repo_root = Path(__file__).resolve().parents[1]

    def _resolve_repo_path(raw_path: str | Path) -> Path:
        path = Path(raw_path)
        return path if path.is_absolute() else (repo_root / path)

    output_root = Path(args.output_dir).resolve()
    trial_root = output_root / trial_name
    trial_root.mkdir(parents=True, exist_ok=True)
    resolved_final_csv_path = (
        final_csv_path.resolve()
        if final_csv_path is not None
        else (
            Path(args.final_csv_path).resolve()
            if args.final_csv_path
            else (output_root / f"{trial_name}.csv").resolve()
        )
    )
    if args.cleanup_intermediate and _is_relative_to(resolved_final_csv_path, trial_root):
        raise ValueError(
            "--cleanup-intermediate requires --final-csv-path outside the trial folder. "
            f"Current final CSV path is inside trial root: {resolved_final_csv_path}"
        )

    assets = default_bsm_asset_paths(
        repo_root=repo_root,
        smplx_model_dir=_resolve_repo_path(args.smplx_model_dir),
    )
    assets.ensure_exists()

    forward = run_smplx_forward(
        sequence=sequence,
        smplx_model_dir=assets.smplx_model_dir,
        sex_override=args.sex,
        device=args.device,
    )
    marker_map = load_bsm_marker_map(assets.bsm_marker_yaml)
    marker_positions, marker_names = build_bsm_marker_positions(forward.vertices, marker_map)
    trial_trc_path = write_trc(
        marker_positions=marker_positions,
        marker_names=marker_names,
        output_path=trial_root / "trials" / trial_name / "markers.trc",
        frame_rate_hz=sequence.frame_rate_hz,
    )

    subject_json = build_subject_json(
        sequence=sequence,
        smplx_model_dir=assets.smplx_model_dir,
        sex_override=args.sex,
        device=args.device,
    )
    subject_folder = build_addbiomechanics_subject_folder(
        subject_root=trial_root,
        trial_name=trial_name,
        subject_json=subject_json,
        bsm_model_path=_resolve_repo_path(args.bsm_model),
        bsm_geometry_dir=assets.bsm_geometry_dir,
        marker_trc_path=trial_trc_path,
    )

    engine_result = run_addbiomechanics_engine(
        subject_root=subject_folder.subject_root,
        addbio_root=args.addbio_root,
        output_name="results",
    )
    csv_summary = export_addbiomechanics_csv(
        final_model_path=engine_result.final_model_path,
        final_mot_path=engine_result.final_mot_path,
        output_csv_path=subject_folder.subject_root / "CSV" / f"{trial_name}_bsm_dofs.csv",
    )

    torque_summary = None
    final_csv_summary = None
    if not args.skip_inverse_dynamics:
        contact_body_names = _parse_contact_body_names(args.id_contact_bodies)
        fallback_contact_bodies = _fallback_contact_bodies_for_csv(contact_body_names)
        id_output_subdir = "ID_estimatedGRF" if args.id_grf_mode == "estimated" else "ID_noGRF"
        torque_summary = run_inverse_dynamics_and_export_torque_csv(
            model_path=engine_result.final_model_path,
            ik_mot_path=engine_result.final_mot_path,
            output_dir=subject_folder.subject_root / "results" / id_output_subdir,
            trial_name=trial_name,
            torque_csv_path=subject_folder.subject_root / "CSV" / f"{trial_name}_bsm_torques.csv",
            filter_mode=args.id_filter_mode,
            cutoff_hz=args.id_cutoff_hz,
            grf_mode=args.id_grf_mode,
            contact_body_names=contact_body_names,
            friction_coeff=args.id_friction_coeff,
            contact_height_threshold_m=args.id_contact_height_threshold_m,
            contact_speed_threshold_mps=args.id_contact_speed_threshold_mps,
        )
        final_csv_summary = export_final_csv(
            dof_csv_path=csv_summary.output_csv_path,
            torque_csv_path=torque_summary.torque_csv_path,
            output_csv_path=resolved_final_csv_path,
            contact_wrench_csv_path=torque_summary.contact_wrench_csv_path,
            subject_json_path=subject_folder.subject_json_path,
            model_path=engine_result.final_model_path,
            excluded_dofs=("knee_angle_r_beta", "knee_angle_l_beta"),
            fallback_contact_bodies=fallback_contact_bodies,
        )

    summary = {
        "trial": trial_name,
        "subject_root": str(subject_folder.subject_root),
        "trial_trc_path": str(trial_trc_path),
        "subject_json_path": str(subject_folder.subject_json_path),
        "engine_output_root": str(engine_result.output_root),
        "final_model_path": str(engine_result.final_model_path),
        "final_mot_path": str(engine_result.final_mot_path),
        "csv_path": str(csv_summary.output_csv_path),
        "velocity_source": csv_summary.velocity_source,
        "frames": csv_summary.frames,
        "dof_count": len(csv_summary.dof_names),
        "inverse_dynamics_enabled": not args.skip_inverse_dynamics,
        "engine_result": json.loads(result_to_json(engine_result)),
        "final_csv_path": str(resolved_final_csv_path) if final_csv_summary is not None else None,
    }
    if torque_summary is not None:
        summary["torque_csv_path"] = str(torque_summary.torque_csv_path)
        summary["id_sto_path"] = str(torque_summary.id_sto_path)
        summary["id_setup_path"] = str(torque_summary.id_setup_path)
        summary["id_cutoff_hz"] = torque_summary.cutoff_hz
        summary["id_missing_dofs"] = list(torque_summary.missing_dofs)
        summary["id_grf_mode"] = torque_summary.grf_mode
        if torque_summary.grf_mot_path is not None:
            summary["id_grf_mot_path"] = str(torque_summary.grf_mot_path)
        if torque_summary.external_loads_xml_path is not None:
            summary["id_external_loads_xml_path"] = str(torque_summary.external_loads_xml_path)
        if torque_summary.contact_wrench_csv_path is not None:
            summary["id_contact_wrench_csv_path"] = str(torque_summary.contact_wrench_csv_path)
    if final_csv_summary is not None:
        summary["final_dof_count"] = len(final_csv_summary.dof_names)
        summary["final_contact_bodies"] = list(final_csv_summary.contact_body_names)
        summary["final_body_scale_names"] = list(final_csv_summary.body_scale_names)
        summary["final_subject_mass_kg"] = final_csv_summary.mass_kg
        summary["final_subject_height_m"] = final_csv_summary.height_m
        summary["excluded_dofs_in_final_csv"] = ["knee_angle_r_beta", "knee_angle_l_beta"]

    if args.cleanup_intermediate:
        if not resolved_final_csv_path.exists():
            raise FileNotFoundError(f"Final CSV not found after export: {resolved_final_csv_path}")
        if trial_root.exists():
            shutil.rmtree(trial_root)
        summary["cleanup_intermediate"] = True
        summary["summary_json_path"] = None
    else:
        summary_path = subject_folder.subject_root / "summary.json"
        summary["summary_json_path"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    from opensim_batch_dynamics.amass_loader import load_all_amass_npz, load_amass_npz

    # Backward-compatible default: one input sequence + explicit trial name.
    if args.trial.lower() != "all":
        sequence = load_amass_npz(args.input)
        if args.sex:
            sequence = sequence.copy_with_gender(args.sex)
        return _run_single_trial_pipeline(args=args, sequence=sequence, trial_name=args.trial)

    # all-mode: run every trial found inside the npz and export one CSV each.
    sequences = load_all_amass_npz(args.input)
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_mode_final_root = _resolve_all_mode_final_csv_root(args, output_root)

    seen_trial_names: set[str] = set()
    trial_summaries: dict[str, dict[str, object]] = {}
    ordered_final_csv_paths: list[str] = []

    for index, (raw_trial_name, sequence) in enumerate(sequences.items(), start=1):
        fallback_trial = f"trial_{index:03d}"
        normalized_trial_name = _sanitize_trial_name(raw_trial_name, fallback=fallback_trial)
        unique_trial_name = normalized_trial_name
        duplicate_index = 2
        while unique_trial_name in seen_trial_names:
            unique_trial_name = f"{normalized_trial_name}_{duplicate_index}"
            duplicate_index += 1
        seen_trial_names.add(unique_trial_name)

        if args.sex:
            sequence = sequence.copy_with_gender(args.sex)

        final_csv_path = (all_mode_final_root / f"{unique_trial_name}.csv").resolve()
        trial_summary = _run_single_trial_pipeline(
            args=args,
            sequence=sequence,
            trial_name=unique_trial_name,
            final_csv_path=final_csv_path,
        )
        trial_summary["raw_trial_name"] = raw_trial_name
        trial_summaries[unique_trial_name] = trial_summary
        ordered_final_csv_paths.append(str(final_csv_path))

    return {
        "mode": "all",
        "input_npz_path": str(Path(args.input).resolve()),
        "trial_count": len(trial_summaries),
        "trial_names": list(trial_summaries.keys()),
        "final_csv_paths": ordered_final_csv_paths,
        "trials": trial_summaries,
    }


def main() -> int:
    _add_src_to_path()
    parser = build_arg_parser()
    args = parser.parse_args()
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

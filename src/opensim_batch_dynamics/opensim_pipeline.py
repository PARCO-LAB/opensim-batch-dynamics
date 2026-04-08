from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np

from .amass_loader import AMASSSequence, load_amass_npz
from .config import AssetPaths, default_asset_paths
from .opencap_markers import (
    build_opencap_marker_dict,
    collect_required_markers,
    ensure_markers,
    load_vertex_index_map,
    marker_matrix,
)
from .smplx_forward import run_smplx_forward
from .trc_export import infer_trc_time_range, write_trc


DEFAULT_ROTATIONS = {"x": 0.0, "y": 90.0, "z": 0.0}


@dataclass
class PipelineOutputs:
    """Main artifacts produced by the AMASS -> OpenSim pipeline."""

    trial_trc_path: Path
    neutral_trc_path: Path
    scaled_model_path: Path | None
    ik_motion_path: Path | None
    scale_setup_path: Path | None
    ik_setup_path: Path | None


def _run_tool(setup_xml: Path) -> None:
    """Run an OpenSim XML tool setup via opensim-cmd."""
    command = ["opensim-cmd", "-o", "error", "run-tool", str(setup_xml)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "OpenSim tool execution failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def _sync_geometry_assets(source_model_path: Path, target_model_dir: Path) -> Path | None:
    """Copy model Geometry folder next to generated setup/model files."""
    source_geometry_dir = source_model_path.parent / "Geometry"
    if not source_geometry_dir.exists():
        return None
    target_geometry_dir = target_model_dir / "Geometry"
    target_geometry_dir.mkdir(parents=True, exist_ok=True)
    for src in source_geometry_dir.glob("*"):
        if src.is_file():
            shutil.copy2(src, target_geometry_dir / src.name)
    return target_geometry_dir


def _time_range_array(opensim_module, start: float, end: float):
    """Create OpenSim ArrayDouble [start, end]."""
    time_range = opensim_module.ArrayDouble(float(start), 0)
    time_range.insert(1, float(end))
    return time_range


def _prepare_marker_set_model(opensim_module, model_path: Path, marker_set_path: Path, output_path: Path) -> Path:
    """Inject marker set into model and save a temporary model file."""
    model = opensim_module.Model(str(model_path))
    marker_set = opensim_module.MarkerSet(str(marker_set_path))
    model.set_MarkerSet(marker_set)
    model.printToXML(str(output_path))
    return output_path


def _disable_invalid_scale_tasks(opensim_module, scale_tool, model_with_markers_path: Path) -> None:
    """Disable scale tasks that reference markers/coordinates absent in model."""
    model = opensim_module.Model(str(model_with_markers_path))
    coord_set = model.getCoordinateSet()
    coord_names: list[str] = []
    for i in range(coord_set.getSize()):
        coord = coord_set.get(i)
        if not coord.getDefaultLocked():
            coord_names.append(coord.getName())

    marker_set = model.getMarkerSet()
    model_marker_names = [marker_set.get(i).getName() for i in range(marker_set.getSize())]

    marker_placer = scale_tool.getMarkerPlacer()
    task_set = marker_placer.getIKTaskSet()
    for i in range(task_set.getSize()):
        task = task_set.get(i)
        class_name = task.getConcreteClassName()
        if class_name == "IKCoordinateTask" and task.getName() not in coord_names:
            task.setApply(False)
        if class_name == "IKMarkerTask" and task.getName() not in model_marker_names:
            task.setApply(False)

    measurement_set = scale_tool.getModelScaler().getMeasurementSet()
    for m in range(measurement_set.getSize()):
        measurement = measurement_set.get(m)
        marker_pair_set = measurement.getMarkerPairSet()
        pair_index = 0
        while pair_index < measurement.getNumMarkerPairs():
            pair = marker_pair_set.get(pair_index)
            names = [pair.getMarkerName(0), pair.getMarkerName(1)]
            if any(name not in model_marker_names for name in names):
                marker_pair_set.remove(pair_index)
            else:
                pair_index += 1
        if measurement.getNumMarkerPairs() == 0:
            measurement.setApply(False)


def run_scale_tool(
    generic_setup_path: Path,
    generic_model_path: Path,
    marker_set_path: Path,
    neutral_trc_path: Path,
    output_folder: Path,
    subject_mass_kg: float,
    subject_height_m: float,
    time_range: tuple[float, float],
    scaled_model_name: str,
) -> tuple[Path, Path]:
    """Run OpenSim ScaleTool and return (scaled_model_path, scale_setup_xml)."""
    try:
        import opensim
    except ImportError as exc:
        raise RuntimeError("opensim Python package is required for scaling.") from exc

    output_folder.mkdir(parents=True, exist_ok=True)
    _sync_geometry_assets(generic_model_path, output_folder)
    updated_model_path = output_folder / f"{Path(generic_model_path).stem}_generic.osim"
    scaled_model_path = output_folder / f"{scaled_model_name}.osim"
    scaled_motion_path = output_folder / f"{scaled_model_name}.mot"
    setup_output_path = output_folder / f"Setup_Scale_{scaled_model_name}.xml"

    _prepare_marker_set_model(opensim, generic_model_path, marker_set_path, updated_model_path)
    scale_tool = opensim.ScaleTool(str(generic_setup_path))
    scale_tool.setName(scaled_model_name)
    scale_tool.setSubjectMass(float(subject_mass_kg))
    scale_tool.setSubjectHeight(float(subject_height_m))

    generic_model_maker = scale_tool.getGenericModelMaker()
    generic_model_maker.setModelFileName(str(updated_model_path))

    model_scaler = scale_tool.getModelScaler()
    model_scaler.setMarkerFileName(str(neutral_trc_path))
    model_scaler.setOutputModelFileName("")
    model_scaler.setOutputScaleFileName("")
    model_scaler.setTimeRange(_time_range_array(opensim, time_range[0], time_range[1]))

    marker_placer = scale_tool.getMarkerPlacer()
    marker_placer.setMarkerFileName(str(neutral_trc_path))
    marker_placer.setOutputModelFileName(str(scaled_model_path))
    marker_placer.setOutputMotionFileName(str(scaled_motion_path))
    marker_placer.setOutputMarkerFileName("")
    marker_placer.setTimeRange(_time_range_array(opensim, time_range[0], time_range[1]))

    _disable_invalid_scale_tasks(opensim, scale_tool, updated_model_path)
    scale_tool.printToXML(str(setup_output_path))
    _run_tool(setup_output_path)

    if not scaled_model_path.exists():
        raise RuntimeError(f"Scaling finished but scaled model is missing: {scaled_model_path}")
    return scaled_model_path, setup_output_path


def run_ik_tool(
    ik_setup_path: Path,
    scaled_model_path: Path,
    trial_trc_path: Path,
    output_folder: Path,
    ik_name: str,
) -> tuple[Path, Path]:
    """Run OpenSim InverseKinematicsTool and return (mot_path, ik_setup_xml)."""
    try:
        import opensim
    except ImportError as exc:
        raise RuntimeError("opensim Python package is required for inverse kinematics.") from exc

    output_folder.mkdir(parents=True, exist_ok=True)
    _sync_geometry_assets(scaled_model_path, output_folder)
    output_motion_path = output_folder / f"{ik_name}.mot"
    output_setup_path = output_folder / f"Setup_IK_{ik_name}.xml"

    ik_tool = opensim.InverseKinematicsTool(str(ik_setup_path))
    ik_tool.setName(ik_name)
    ik_tool.set_model_file(str(scaled_model_path))
    ik_tool.set_marker_file(str(trial_trc_path))
    start_s, end_s = infer_trc_time_range(trial_trc_path)
    ik_tool.set_time_range(0, float(start_s))
    ik_tool.set_time_range(1, float(end_s))
    ik_tool.setResultsDir(str(output_folder))
    ik_tool.set_report_errors(True)
    ik_tool.set_report_marker_locations(False)
    ik_tool.set_output_motion_file(str(output_motion_path))
    ik_tool.printToXML(str(output_setup_path))
    _run_tool(output_setup_path)

    if not output_motion_path.exists():
        raise RuntimeError(f"IK finished but motion file is missing: {output_motion_path}")
    return output_motion_path, output_setup_path


def _forward_markers(
    sequence: AMASSSequence,
    assets: AssetPaths,
    sex_override: str | None = None,
    device: str = "cpu",
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Run SMPL-X forward and return OpenCap marker dictionary + required order."""
    forward = run_smplx_forward(
        sequence=sequence,
        smplx_model_dir=assets.smplx_model_dir,
        sex_override=sex_override,
        device=device,
        num_betas=16,
    )
    vertex_map = load_vertex_index_map(assets.opencap_vertex_map)
    marker_dict = build_opencap_marker_dict(
        vertices=forward.vertices,
        joints=forward.joints,
        vertex_index_map=vertex_map,
    )
    required = collect_required_markers(
        marker_set_xml=assets.opencap_marker_set,
        ik_setup_xml=assets.opencap_ik_setup,
        scale_setup_xml=assets.opencap_scaling_setup,
    )
    ensure_markers(marker_dict, required)
    return marker_dict, required


def run_amass_to_opensim(
    input_npz_path: str | Path,
    trial_name: str,
    output_dir: str | Path,
    mass_kg: float,
    height_m: float,
    sex: str | None = None,
    assets: AssetPaths | None = None,
    model_path: str | Path | None = None,
    rotations_deg: dict[str, float] | None = None,
    apply_vertical_offset: bool = True,
    skip_scale: bool = False,
    skip_ik: bool = False,
    device: str = "cpu",
) -> PipelineOutputs:
    """
    End-to-end conversion from AMASS npz to TRC + scaled model + IK motion.

    This function is model-agnostic and can operate with the torque-only model
    by passing the corresponding `assets.model_path`.
    """
    assets = assets or default_asset_paths()
    if model_path is not None:
        assets = AssetPaths(
            repository_root=assets.repository_root,
            model_path=Path(model_path).resolve(),
            smplx_model_dir=assets.smplx_model_dir,
            opencap_assets_dir=assets.opencap_assets_dir,
            opencap_scaling_setup=assets.opencap_scaling_setup,
            opencap_ik_setup=assets.opencap_ik_setup,
            opencap_marker_set=assets.opencap_marker_set,
            opencap_vertex_map=assets.opencap_vertex_map,
        )
    assets.ensure_exists()

    output_root = Path(output_dir).resolve()
    marker_dir = output_root / "MarkerData" / trial_name
    model_output_dir = output_root / "OpenSim" / "Model" / trial_name
    ik_output_dir = output_root / "OpenSim" / "IK" / trial_name

    sequence = load_amass_npz(input_npz_path)
    if sex:
        sequence = sequence.copy_with_gender(sex)

    marker_dict, required_markers = _forward_markers(
        sequence=sequence,
        assets=assets,
        sex_override=sex,
        device=device,
    )
    trial_marker_matrix = marker_matrix(marker_dict, required_markers)
    trial_vertical_offset = (
        float(np.min(trial_marker_matrix[:, :, 1])) if apply_vertical_offset else None
    )

    if rotations_deg is None:
        rotations_deg = DEFAULT_ROTATIONS

    trial_trc_path = write_trc(
        marker_positions=trial_marker_matrix,
        marker_names=required_markers,
        output_path=marker_dir / f"{trial_name}.trc",
        frame_rate_hz=sequence.frame_rate_hz,
        rotations_deg=rotations_deg,
        vertical_offset=trial_vertical_offset,
    )

    neutral_sequence = sequence.make_neutral(n_frames=30)
    neutral_markers, _ = _forward_markers(
        sequence=neutral_sequence,
        assets=assets,
        sex_override=sex,
        device=device,
    )
    neutral_marker_matrix = marker_matrix(neutral_markers, required_markers)
    neutral_vertical_offset = (
        float(np.min(neutral_marker_matrix[:, :, 1])) if apply_vertical_offset else None
    )
    neutral_trc_path = write_trc(
        marker_positions=neutral_marker_matrix,
        marker_names=required_markers,
        output_path=model_output_dir / "neutral.trc",
        frame_rate_hz=30.0,
        rotations_deg=rotations_deg,
        vertical_offset=neutral_vertical_offset,
    )

    if skip_scale:
        scaled_model_path = assets.model_path
        scale_setup_path = None
    else:
        scaled_model_path, scale_setup_path = run_scale_tool(
            generic_setup_path=assets.opencap_scaling_setup,
            generic_model_path=assets.model_path,
            marker_set_path=assets.opencap_marker_set,
            neutral_trc_path=neutral_trc_path,
            output_folder=model_output_dir,
            subject_mass_kg=mass_kg,
            subject_height_m=height_m,
            time_range=(0.0, 1.0),
            scaled_model_name=f"{Path(assets.model_path).stem}_scaled",
        )

    if skip_ik:
        ik_motion_path = None
        ik_setup_path = None
    else:
        if scaled_model_path is None:
            raise RuntimeError("IK requires a scaled or provided model path.")
        ik_motion_path, ik_setup_path = run_ik_tool(
            ik_setup_path=assets.opencap_ik_setup,
            scaled_model_path=scaled_model_path,
            trial_trc_path=trial_trc_path,
            output_folder=ik_output_dir,
            ik_name=trial_name,
        )

    return PipelineOutputs(
        trial_trc_path=trial_trc_path,
        neutral_trc_path=neutral_trc_path,
        scaled_model_path=scaled_model_path,
        ik_motion_path=ik_motion_path,
        scale_setup_path=scale_setup_path,
        ik_setup_path=ik_setup_path,
    )

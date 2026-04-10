from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

from .bsm_subject_json import write_subject_json


@dataclass(frozen=True)
class AddBiomechanicsSubjectFolder:
    """Filesystem layout required by the local AddBiomechanics engine."""

    subject_root: Path
    subject_json_path: Path
    unscaled_model_path: Path
    geometry_path: Path
    trial_dir: Path
    trial_marker_path: Path


def _copy_or_link_geometry(source_dir: Path, target_dir: Path) -> None:
    """Prefer a symlink for geometry, but fall back to copying on failure."""
    if target_dir.exists():
        return
    try:
        os.symlink(source_dir, target_dir, target_is_directory=True)
        return
    except OSError:
        pass

    shutil.copytree(source_dir, target_dir)


def _canonicalize_marker_parent_frames(model_path: Path) -> None:
    """
    Normalize marker socket parent frames to explicit body paths.

    Some OpenSim toolchains fail to resolve bare body names (e.g. ``tibia_l``)
    in marker sockets and require ``/bodyset/tibia_l``.
    """
    root = ET.parse(model_path).getroot()
    body_names = {elem.attrib["name"] for elem in root.findall(".//Body") if "name" in elem.attrib}
    updated = False
    for marker in root.findall(".//Marker"):
        socket = marker.find("socket_parent_frame")
        if socket is None or socket.text is None:
            continue
        raw = socket.text.strip()
        if not raw or raw.startswith("/"):
            continue
        if raw in body_names:
            socket.text = f"/bodyset/{raw}"
            updated = True
    if updated:
        ET.ElementTree(root).write(model_path, encoding="utf-8", xml_declaration=True)


def build_addbiomechanics_subject_folder(
    subject_root: str | Path,
    trial_name: str,
    subject_json: dict[str, object],
    bsm_model_path: str | Path,
    bsm_geometry_dir: str | Path,
    marker_trc_path: str | Path,
) -> AddBiomechanicsSubjectFolder:
    """
    Create the AddBiomechanics subject folder expected by ``engine.py``.

    The layout matches the local engine's custom-skeleton path:
    ``_subject.json``, ``unscaled_generic.osim``, and ``trials/<trial>/markers.trc``.
    """
    root = Path(subject_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    subject_json_path = write_subject_json(root / "_subject.json", subject_json)
    unscaled_model_path = root / "unscaled_generic.osim"
    shutil.copy2(Path(bsm_model_path), unscaled_model_path)
    _canonicalize_marker_parent_frames(unscaled_model_path)

    geometry_path = root / "Geometry"
    _copy_or_link_geometry(Path(bsm_geometry_dir), geometry_path)

    trial_dir = root / "trials" / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_marker_path = trial_dir / "markers.trc"
    source_marker_path = Path(marker_trc_path).resolve()
    if source_marker_path != trial_marker_path.resolve():
        shutil.copy2(source_marker_path, trial_marker_path)

    return AddBiomechanicsSubjectFolder(
        subject_root=root,
        subject_json_path=subject_json_path,
        unscaled_model_path=unscaled_model_path,
        geometry_path=geometry_path,
        trial_dir=trial_dir,
        trial_marker_path=trial_marker_path,
    )

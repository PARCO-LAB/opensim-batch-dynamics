from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np


_SMPLX_JOINT_INDEX = {
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "neck": 12,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_eye": 23,
    "right_eye": 24,
}


def load_vertex_index_map(csv_path: str | Path) -> dict[str, int]:
    """Load OpenCap marker -> SMPL-X vertex index mapping."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["Name"]: int(row["Index"]) for row in reader}


def marker_names_from_marker_set_xml(marker_set_xml: str | Path) -> list[str]:
    """Read marker names defined in OpenSim MarkerSet XML."""
    root = ET.parse(marker_set_xml).getroot()
    marker_names: list[str] = []
    for marker in root.findall(".//Marker"):
        name = marker.attrib.get("name")
        if name:
            marker_names.append(name)
    return marker_names


def marker_names_from_ik_setup(ik_setup_xml: str | Path) -> list[str]:
    """Read marker task names from an IK setup XML."""
    root = ET.parse(ik_setup_xml).getroot()
    marker_names: list[str] = []
    for marker_task in root.findall(".//IKMarkerTask"):
        name = marker_task.attrib.get("name")
        if name:
            marker_names.append(name)
    return marker_names


def marker_names_from_scaling_setup(scale_setup_xml: str | Path) -> list[str]:
    """Read marker names used in scaling measurements."""
    text = Path(scale_setup_xml).read_text(encoding="utf-8")
    marker_pairs = re.findall(r"<markers>\s*([^<]+?)\s*</markers>", text)
    marker_names: set[str] = set()
    for pair in marker_pairs:
        for token in pair.strip().split():
            marker_names.add(token)
    return sorted(marker_names)


def collect_required_markers(
    marker_set_xml: str | Path,
    ik_setup_xml: str | Path,
    scale_setup_xml: str | Path,
) -> list[str]:
    """Collect union of markers required by marker set, scaling and IK configs."""
    names: set[str] = set()
    names.update(marker_names_from_marker_set_xml(marker_set_xml))
    names.update(marker_names_from_ik_setup(ik_setup_xml))
    names.update(marker_names_from_scaling_setup(scale_setup_xml))
    return sorted(names)


def _from_vertex(
    vertices: np.ndarray,
    vertex_index_map: dict[str, int],
) -> dict[str, np.ndarray]:
    """Build marker dictionary from direct vertex lookups."""
    marker_data: dict[str, np.ndarray] = {}
    n_vertices = vertices.shape[1]
    for name, index in vertex_index_map.items():
        if 0 <= index < n_vertices:
            marker_data[name] = vertices[:, index, :]
    return marker_data


def _joint(joints: np.ndarray, index_name: str) -> np.ndarray | None:
    idx = _SMPLX_JOINT_INDEX[index_name]
    if joints.shape[1] <= idx:
        return None
    return joints[:, idx, :]


def _average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def build_opencap_marker_dict(
    vertices: np.ndarray,
    joints: np.ndarray,
    vertex_index_map: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Build OpenCap-compatible marker dictionary from SMPL-X vertices/joints.

    This mirrors the marker logic needed by the OpenCap monocular scaling/IK setup.
    """
    if vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError(f"Expected vertices shape (T, V, 3), got {vertices.shape}")
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"Expected joints shape (T, J, 3), got {joints.shape}")

    markers = _from_vertex(vertices, vertex_index_map)

    alias_pairs = {
        "RShoulder": "r_shoulder",
        "LShoulder": "l_shoulder",
        "RElbow": "r_elbow",
        "LElbow": "l_elbow",
        "RKnee": "r_knee",
        "LKnee": "l_knee",
        "RAnkle": "r_ankle",
        "LAnkle": "l_ankle",
        "RBigToe": "r_big_toe",
        "LBigToe": "l_big_toe",
        "RSmallToe": "r_5meta",
        "LSmallToe": "l_5meta",
        "RHeel": "r_calc",
        "LHeel": "l_calc",
    }
    for upper_name, lower_name in alias_pairs.items():
        if lower_name in markers:
            markers[upper_name] = markers[lower_name]

    right_hip = _joint(joints, "right_hip")
    left_hip = _joint(joints, "left_hip")
    right_shoulder = _joint(joints, "right_shoulder")
    left_shoulder = _joint(joints, "left_shoulder")
    right_elbow = _joint(joints, "right_elbow")
    left_elbow = _joint(joints, "left_elbow")
    right_wrist = _joint(joints, "right_wrist")
    left_wrist = _joint(joints, "left_wrist")
    right_knee = _joint(joints, "right_knee")
    left_knee = _joint(joints, "left_knee")
    right_ankle = _joint(joints, "right_ankle")
    left_ankle = _joint(joints, "left_ankle")
    neck = _joint(joints, "neck")

    if right_hip is not None:
        markers["RHip"] = right_hip
    if left_hip is not None:
        markers["LHip"] = left_hip
    if right_knee is not None:
        markers["RKnee"] = right_knee
    if left_knee is not None:
        markers["LKnee"] = left_knee
    if right_ankle is not None:
        markers["RAnkle"] = right_ankle
    if left_ankle is not None:
        markers["LAnkle"] = left_ankle
    if right_shoulder is not None:
        markers["RShoulder"] = right_shoulder
    if left_shoulder is not None:
        markers["LShoulder"] = left_shoulder
    if right_elbow is not None:
        markers["RElbow"] = right_elbow
    if left_elbow is not None:
        markers["LElbow"] = left_elbow
    if right_wrist is not None:
        markers["RWrist"] = right_wrist
    if left_wrist is not None:
        markers["LWrist"] = left_wrist

    if "r_wrist_radius" in markers and "r_wrist_ulna" in markers:
        markers["RWrist"] = _average(markers["r_wrist_radius"], markers["r_wrist_ulna"])
    if "l_wrist_radius" in markers and "l_wrist_ulna" in markers:
        markers["LWrist"] = _average(markers["l_wrist_radius"], markers["l_wrist_ulna"])

    if "RHip" in markers and "LHip" in markers:
        markers["midHip"] = _average(markers["RHip"], markers["LHip"])
    elif "r_ASIS" in markers and "l_ASIS" in markers:
        markers["midHip"] = _average(markers["r_ASIS"], markers["l_ASIS"])

    if neck is not None:
        markers["Neck"] = neck
    elif "RShoulder" in markers and "LShoulder" in markers:
        markers["Neck"] = _average(markers["RShoulder"], markers["LShoulder"])
    elif "C7" in markers:
        markers["Neck"] = markers["C7"]

    return markers


def ensure_markers(marker_dict: dict[str, np.ndarray], required_names: Iterable[str]) -> None:
    """Validate that all markers required by OpenSim XML files are present."""
    missing = sorted(name for name in required_names if name not in marker_dict)
    if missing:
        raise ValueError(f"Missing required markers: {', '.join(missing)}")


def marker_matrix(marker_dict: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """Pack markers in fixed order into a (T, N, 3) array for TRC export."""
    arrays = []
    for name in names:
        if name not in marker_dict:
            raise KeyError(f"Marker not found: {name}")
        arrays.append(marker_dict[name][:, None, :])
    return np.concatenate(arrays, axis=1)

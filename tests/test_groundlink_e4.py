from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_groundlink_e4 import _reference_valid_mask, _run_trial, _trial_metrics  # noqa: E402


class GroundLinkE4Test(unittest.TestCase):
    def test_excluded_trial_is_reported_without_pipeline_artifacts(self) -> None:
        pair = type(
            "Pair",
            (),
            {
                "subject": "s002",
                "trial": "balletsmalljump_1",
                "exclusion": "nonballistic_flight_kinematics",
                "motion_path": Path("balletsmalljump_1_stageii.npz"),
                "force_path": None,
            },
        )()
        rows = _run_trial(pair, Path("/does/not/exist"), Path("/does/not/exist"))
        self.assertEqual(rows[0]["metric"], "excluded_nonballistic_flight_kinematics")

    def test_zero_error_and_normalized_residuals(self) -> None:
        names = ["hip_flexion_l_tau", "knee_angle_l_tau", "ankle_angle_l_tau", "pelvis_tx_tau", "pelvis_ty_tau", "pelvis_tz_tau", "pelvis_tilt_tau", "pelvis_list_tau", "pelvis_rotation_tau"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tau.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["time", *names]); writer.writeheader(); writer.writerow({"time": 0, **{name: 0 for name in names}})
            metrics = _trial_metrics(path, path, 10.0, 2.0)
        self.assertEqual(metrics["hip_rmse_mgh"], 0.0)
        self.assertEqual(metrics["residual_force_pct_bw"], 0.0)

    def test_force_residual_uses_resultant_not_component_rms(self) -> None:
        names = ["hip_flexion_l_tau", "knee_angle_l_tau", "ankle_angle_l_tau", "pelvis_tx_tau", "pelvis_ty_tau", "pelvis_tz_tau", "pelvis_tilt_tau", "pelvis_list_tau", "pelvis_rotation_tau"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tau.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["time", *names])
                writer.writeheader()
                writer.writerow({"time": 0, **{name: (98.1 if name == "pelvis_tx_tau" else 0) for name in names}})
            metrics = _trial_metrics(path, path, 10.0, 2.0)
        self.assertAlmostEqual(metrics["residual_force_pct_bw"], 100.0)
        self.assertAlmostEqual(metrics["residual_force_max_pct_bw"], 100.0)
        self.assertTrue(math.isfinite(metrics["residual_force_x_rms_pct_bw"]))

    def test_valid_mask_excludes_invalid_residual_frame(self) -> None:
        names = ["hip_flexion_l_tau", "knee_angle_l_tau", "ankle_angle_l_tau", "pelvis_tx_tau", "pelvis_ty_tau", "pelvis_tz_tau", "pelvis_tilt_tau", "pelvis_list_tau", "pelvis_rotation_tau"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tau.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["time", *names])
                writer.writeheader()
                for time, force in ((0.0, 98.1), (1.0, 0.0)):
                    writer.writerow({"time": time, **{name: (force if name == "pelvis_tx_tau" else 0) for name in names}})
            metrics = _trial_metrics(
                path,
                path,
                10.0,
                2.0,
                valid_mask=np.array([False, True]),
            )
        self.assertEqual(metrics["residual_force_pct_bw"], 0.0)

    def test_reference_mask_rejects_unmeasured_support_and_keeps_flight(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "target.csv"
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["time", "target_force_x", "target_force_y", "target_force_z"],
                )
                writer.writeheader()
                writer.writerow({"time": 0, "target_force_x": 0, "target_force_y": 0, "target_force_z": 98.1})
                writer.writerow({"time": 1, "target_force_x": 0, "target_force_y": 0, "target_force_z": 0})
            measured = np.zeros((2, 2, 3), dtype=float)
            mask = _reference_valid_mask(
                np.array([0.0, 1.0]), measured, target, mass=10.0, threshold_bw=0.2
            )
        np.testing.assert_array_equal(mask, [False, True])

    def test_reference_mask_rejects_airborne_frames_when_contact_allocator_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "target.csv"
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "time", "target_force_x", "target_force_y", "target_force_z",
                        "contact_activation_sum",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "time": 0, "target_force_x": 0, "target_force_y": 0,
                    "target_force_z": 98.1, "contact_activation_sum": 1,
                })
                writer.writerow({
                    "time": 1, "target_force_x": 0, "target_force_y": 0,
                    "target_force_z": 0, "contact_activation_sum": 0,
                })
            measured = np.zeros((2, 2, 3), dtype=float)
            mask = _reference_valid_mask(
                np.array([0.0, 1.0]), measured, target, mass=10.0, threshold_bw=0.2
            )
        np.testing.assert_array_equal(mask, [False, False])

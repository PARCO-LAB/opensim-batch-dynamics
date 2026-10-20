from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_groundlink import (  # noqa: E402
    EXCLUDED_MOTIONS,
    EXCLUDED_TRIALS,
    _aggregate_side_cop,
    _physics_residual,
    discover_pairs,
    evaluate_signals,
    load_groundlink_force,
    write_manifest,
)


class GroundLinkValidatorTest(unittest.TestCase):
    def test_chair_is_excluded_as_unmodelled_external_contact(self) -> None:
        self.assertIn("chair", EXCLUDED_MOTIONS)

    def test_confirmed_invalid_kinematic_fit_is_excluded(self) -> None:
        self.assertEqual(
            EXCLUDED_TRIALS[("s002", "jumpingjack_3")],
            "invalid_pelvis_kinematics",
        )

    def test_nonballistic_ballet_jump_is_excluded(self) -> None:
        self.assertEqual(
            EXCLUDED_TRIALS[("s002", "balletsmalljump_1")],
            "nonballistic_flight_kinematics",
        )

    def test_manifest_does_not_load_force_for_excluded_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion = Path(tmpdir) / "motion_stageii.npz"
            force = Path(tmpdir) / "force.npy"
            motion.write_bytes(b"placeholder")
            force.write_bytes(b"placeholder")
            pair = SimpleNamespace(
                subject="s002",
                trial="balletsmalljump_1",
                motion_path=motion,
                force_path=force,
                frames=10,
                exclusion="nonballistic_flight_kinematics",
            )
            manifest = write_manifest([pair], Path(tmpdir) / "manifest.csv")
            with manifest.open(newline="", encoding="utf-8") as handle:
                row = next(iter(csv.DictReader(handle)))
        self.assertEqual(row["status"], "excluded")
        self.assertEqual(row["force_frames"], "")

    def test_discover_pairs_derives_force_path_without_directory_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "moshpp" / "s002"
            force_root = Path(tmpdir) / "force" / "s002"
            root.mkdir(parents=True)
            force_root.mkdir(parents=True)
            motion = root / "s002_20220520_balletsmalljump_1_stageii.npz"
            np.savez(motion, trans=np.zeros((4, 2, 3)))
            force = force_root / "s002_20220520_balletsmalljump_1.npy"
            force.write_bytes(b"not loaded")

            pairs = discover_pairs(root, force_root, "s002")

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].force_path, force)
        self.assertEqual(pairs[0].frames, 4)

    def test_fast_discovery_skips_motion_array_shape_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "moshpp" / "s002"
            force_root = Path(tmpdir) / "force" / "s002"
            root.mkdir(parents=True)
            force_root.mkdir(parents=True)
            motion = root / "s002_20220520_balletsmalljump_1_stageii.npz"
            np.savez(motion, trans=np.zeros((4, 2, 3)))
            force = force_root / "s002_20220520_balletsmalljump_1.npy"
            force.write_bytes(b"not loaded")

            pairs = discover_pairs(root, force_root, "s002", read_motion_frames=False)

        self.assertEqual(pairs[0].frames, None)

    def test_metrics_identical_delayed_and_without_contact(self) -> None:
        ref = np.zeros((4, 3), dtype=float)
        ref[:, 2] = (50.0, 100.0, 150.0, 200.0)
        cop = np.zeros_like(ref)
        same = evaluate_signals(ref, ref, cop, cop, 10.0)
        self.assertEqual(next(row["value"] for row in same if row["metric"] == "grf_rmse_z"), 0.0)
        delayed = np.roll(ref, 1, axis=0)
        delayed_metrics = evaluate_signals(delayed, ref, cop, cop, 10.0)
        self.assertGreater(next(row["value"] for row in delayed_metrics if row["metric"] == "grf_rmse_z"), 0.0)
        no_contact = evaluate_signals(np.zeros_like(ref), ref, cop, cop, 10.0)
        self.assertEqual(next(row["value"] for row in no_contact if row["metric"] == "contact_recall"), 0.0)

    def test_contact_metrics_peaks_and_physics_residual(self) -> None:
        ref = np.zeros((6, 3), dtype=float)
        ref[:, 2] = (0, 100, 200, 0, 100, 300)
        pred = ref.copy(); pred[0, 2] = 1_000; pred[2, 2] = 100; pred[5, 2] = 200
        metrics = {row["metric"]: row["value"] for row in evaluate_signals(pred, ref, np.zeros_like(ref), np.zeros_like(ref), 10.0)}
        self.assertAlmostEqual(metrics["fz_rmse_contact"], np.sqrt((100**2 + 100**2) / 4) / 98.1)
        self.assertAlmostEqual(metrics["fz_peak_error_abs_median"], 100 / 98.1)
        self.assertAlmostEqual(metrics["fz_impulse_error_abs"], 800 / 98.1 / 250)
        target_force = np.tile([0.0, 0.0, 98.1], (2, 1))
        self.assertAlmostEqual(_physics_residual(target_force, target_force, 10.0), 0.0)

    def test_loader_converts_kn_to_newtons_and_preserves_sides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "force.npy"
            cop = np.zeros((2, 2, 3))
            grf = np.ones((2, 2, 3))
            np.save(path, {"CoP": cop, "GRF": grf}, allow_pickle=True)
            loaded_cop, loaded_grf = load_groundlink_force(path)
        np.testing.assert_array_equal(loaded_cop, cop)
        np.testing.assert_array_equal(loaded_grf, grf * 1000.0)

    def test_manifest_marks_missing_force(self) -> None:
        from validate_groundlink import Pair

        with tempfile.TemporaryDirectory() as tmpdir:
            output = write_manifest([Pair("s001", "walk_1", Path("motion.npz"), None, 3)], Path(tmpdir) / "manifest.csv")
            self.assertIn("missing_force", output.read_text(encoding="utf-8"))

    def test_side_cop_is_vertical_force_weighted(self) -> None:
        data = {
            "calcn_l_cop_x": np.array([0.0]), "calcn_l_cop_y": np.array([0.0]), "calcn_l_cop_z": np.array([0.0]),
            "calcn_l_grf_z": np.array([100.0]),
            "toes_l_cop_x": np.array([1.0]), "toes_l_cop_y": np.array([0.0]), "toes_l_cop_z": np.array([0.0]),
            "toes_l_grf_z": np.array([300.0]),
        }
        np.testing.assert_allclose(_aggregate_side_cop(data, "l"), [[0.75, 0.0, 0.0]])

    def test_evaluate_trial_accepts_custom_method_name(self) -> None:
        from validate_groundlink import Pair, evaluate_trial

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            force = root / "force.npy"
            np.save(force, {"CoP": np.zeros((2, 2, 3)), "GRF": np.zeros((2, 2, 3))}, allow_pickle=True)
            prediction = root / "prediction.csv"
            prediction.write_text(
                "time,calcn_l_grf_x,calcn_l_grf_y,calcn_l_grf_z,calcn_r_grf_x,calcn_r_grf_y,calcn_r_grf_z\n"
                "0,0,0,0,0,0,0\n0.004,0,0,0,0,0,0\n", encoding="utf-8"
            )
            records = evaluate_trial(Pair("s007", "trial", Path("motion.npz"), force, 2), prediction, 63.96, "GroundLinkNet")
        self.assertTrue(records)
        self.assertEqual({row["method"] for row in records if row["method"] == "GroundLinkNet"}, {"GroundLinkNet"})

    def test_force_target_enables_com_oracle_and_physics_residual(self) -> None:
        from validate_groundlink import Pair, evaluate_trial

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            force = root / "force.npy"
            grf = np.zeros((2, 2, 3)); grf[:, :, 2] = 98.1
            np.save(force, {"CoP": np.zeros_like(grf), "GRF": grf / 1000.0}, allow_pickle=True)
            prediction = root / "prediction.csv"
            prediction.write_text(
                "time,calcn_l_grf_x,calcn_l_grf_y,calcn_l_grf_z,calcn_r_grf_x,calcn_r_grf_y,calcn_r_grf_z\n"
                "0,0,0,98.1,0,0,98.1\n0.004,0,0,98.1,0,0,98.1\n", encoding="utf-8"
            )
            target = root / "target.csv"
            target.write_text(
                "time,target_force_x,target_force_y,target_force_z\n0,0,0,196.2\n0.004,0,0,196.2\n", encoding="utf-8"
            )
            records = evaluate_trial(Pair("s007", "trial", Path("motion.npz"), force, 2), prediction, 10.0,
                                     force_target_csv=target)
        self.assertIn("COM_equal_split", {row["method"] for row in records})
        residuals = [row["value"] for row in records if row["method"] == "COM_equal_split" and row["metric"] == "physics_residual"]
        self.assertEqual(residuals, [0.0])


if __name__ == "__main__":
    unittest.main()

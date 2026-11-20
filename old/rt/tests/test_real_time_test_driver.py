from __future__ import annotations

import tempfile
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RT"))
sys.path.insert(0, str(ROOT / "src"))

import real_time_test as rtt  # noqa: E402


class RealTimeDriverTest(unittest.TestCase):
    def test_load_reference_table_and_extract_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "frame,time,hip,hip_vel,hip_acc,hip_tau",
                        "0,0.0,1.0,2.0,3.0,4.0",
                        "1,0.5,5.0,6.0,7.0,8.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            table = rtt.load_reference_csv(csv_path, max_frames=2)

        self.assertEqual(len(table), 2)
        signals = rtt.extract_reference_signals(pd.DataFrame(
            {
                "frame": [0, 1],
                "time": [0.0, 0.5],
                "hip": [1.0, 5.0],
                "hip_vel": [2.0, 6.0],
                "hip_acc": [3.0, 7.0],
                "hip_tau": [4.0, 8.0],
            }
        ), ["hip"])
        np.testing.assert_allclose(signals["q"], np.array([[1.0], [5.0]]))
        np.testing.assert_allclose(signals["tau"], np.array([[4.0], [8.0]]))

    def test_parse_args_accepts_init_policy(self) -> None:
        args = rtt.parse_args(["--init-policy", "zero_velocity"])
        self.assertEqual(args.init_policy, "zero_velocity")

    def test_compute_metrics_serializes_expected_keys(self) -> None:
        results = {
            "q_rt": np.array([[1.0, 2.0]]),
            "dq_rt": np.array([[1.0, 2.0]]),
            "ddq_rt": np.array([[1.0, 2.0]]),
            "tau_rt": np.array([[1.0, 2.0]]),
            "left_force_rt": np.array([[1.0, 2.0, 3.0]]),
            "right_force_rt": np.array([[1.0, 2.0, 3.0]]),
            "left_contact_rt": np.array([True]),
            "right_contact_rt": np.array([False]),
            "left_wrench_rt": np.zeros((1, 6)),
            "right_wrench_rt": np.zeros((1, 6)),
            "mpjpe_rt": np.array([0.1]),
            "dyn_residual_rt": np.array([0.2]),
            "solve_time_rt": np.array([0.003]),
            "q_ref": np.array([[0.0, 0.0]]),
            "dq_ref": np.array([[0.0, 0.0]]),
            "ddq_ref": np.array([[0.0, 0.0]]),
            "tau_ref": np.array([[0.0, 0.0]]),
            "time_ref": np.array([0.5]),
            "frame_values": np.array([1.0]),
            "left_force_ref": np.array([[0.0, 0.0, 0.0]]),
            "right_force_ref": np.array([[0.0, 0.0, 0.0]]),
            "left_contact_ref": np.array([True]),
            "right_contact_ref": np.array([False]),
            "subject_mass_kg": 70.0,
            "subject_height_m": 1.75,
            "root_residual_rt": np.zeros((1, 6)),
            "metric_mask": np.array([True, True]),
            "metric_dof_names": ["a", "b"],
            "metric_q_rt": np.array([[1.0, 2.0]]),
            "metric_q_ref": np.array([[0.0, 0.0]]),
            "metric_dq_rt": np.array([[1.0, 2.0]]),
            "metric_dq_ref": np.array([[0.0, 0.0]]),
            "metric_ddq_rt": np.array([[1.0, 2.0]]),
            "metric_ddq_ref": np.array([[0.0, 0.0]]),
            "metric_tau_rt": np.array([[1.0, 2.0]]),
            "metric_tau_ref": np.array([[0.0, 0.0]]),
            "metric_tau_act_rt": np.array([[1.0, 2.0]]),
            "metric_tau_act_ref": np.array([[0.0, 0.0]]),
            "tau_jerk_rt": np.zeros((0, 2)),
            "tau_jerk_ref": np.zeros((0, 2)),
        }

        metrics = rtt.compute_metrics(results)

        self.assertIn("q_rmse", metrics)
        self.assertIn("left_contact", metrics)
        self.assertIn("tau_nm_per_kg_mean_abs", metrics)
        self.assertEqual(metrics["frames"], 1)

    def test_write_realtime_csv_includes_normalized_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "out.csv"
            args = Namespace(
                output_csv=csv_path,
                noise_std=0.0,
                drop_joint_prob=0.0,
                mu=0.8,
                stage1_kin_filter=True,
                init_policy="keypoint_bootstrap",
            )
            frame_table = pd.DataFrame(
                {
                    "frame": [0, 1],
                    "time": [0.0, 0.5],
                    "subject_mass_kg": [70.0, 70.0],
                    "subject_height_m": [1.75, 1.75],
                }
            )
            results = {
                "frame_values": np.array([1.0]),
                "time_ref": np.array([0.5]),
                "subject_mass_kg": 70.0,
                "subject_height_m": 1.75,
                "q_rt": np.array([[1.0]]),
                "dq_rt": np.array([[2.0]]),
                "ddq_rt": np.array([[3.0]]),
                "tau_rt": np.array([[4.0]]),
                "left_force_rt": np.array([[5.0, 6.0, 7.0]]),
                "right_force_rt": np.array([[8.0, 9.0, 10.0]]),
                "left_contact_rt": np.array([True]),
                "right_contact_rt": np.array([False]),
                "left_wrench_rt": np.zeros((1, 6)),
                "right_wrench_rt": np.zeros((1, 6)),
                "root_residual_rt": np.array([[11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]),
                "mpjpe_rt": np.array([0.1]),
                "dyn_residual_rt": np.array([0.2]),
                "solve_time_rt": np.array([0.003]),
            }

            output = rtt.write_realtime_csv(args, frame_table, ["hip"], results)
            written = pd.read_csv(output)

        self.assertEqual(output, csv_path.resolve())
        self.assertIn("hip_tau_nm_per_kg", written.columns)
        self.assertIn("left_grf_bw_x", written.columns)
        self.assertIn("root_residual_mx_bwh", written.columns)
        self.assertIn("input_init_policy", written.columns)
        self.assertAlmostEqual(float(written.loc[0, "hip_tau_nm_per_kg"]), 4.0 / 70.0)
        self.assertAlmostEqual(float(written.loc[0, "left_grf_bw_z"]), 7.0 / (70.0 * 9.81))
        self.assertEqual(written.loc[0, "input_init_policy"], "keypoint_bootstrap")


if __name__ == "__main__":
    unittest.main()

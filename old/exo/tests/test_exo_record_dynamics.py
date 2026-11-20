from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_exo_record_dynamics import (  # noqa: E402
    PLATE_SIZE_M,
    merge_mot_parts,
    plate_wrench,
    split_feet,
    unwrap_angles,
    set_coordinate_prior_targets,
    smooth_weak_coordinate_trajectories,
    write_trc_slice,
)


def _moment_about_origin(force: np.ndarray, point: np.ndarray, free: np.ndarray) -> np.ndarray:
    return np.cross(point, force).sum(axis=1) + free.sum(axis=1)


class SplitFeetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.foot_l = np.array([[0.30, 0.60, 0.04], [0.30, 0.60, 0.04], [0.30, 0.60, 0.04]])
        self.foot_r = np.array([[0.50, 0.30, 0.04], [0.50, 0.30, 0.04], [0.50, 0.30, 0.04]])
        self.force = np.array([[5.0, -3.0, 900.0], [1.0, 2.0, 950.0], [0.0, 0.0, 920.0]])
        self.cop = np.array([[0.40, 0.45, 0.0], [0.41, 0.47, 0.0], [0.70, 0.10, 0.0]])
        self.free = np.array([2.0, -1.0, 0.5])

    def test_total_wrench_is_preserved(self) -> None:
        both = np.ones(3, dtype=bool)
        force, point, moment, r = split_feet(
            self.force, self.cop, self.free, self.foot_l, self.foot_r, both, both
        )
        np.testing.assert_allclose(force.sum(axis=1), self.force)
        expected = np.cross(self.cop, self.force) + np.outer(self.free, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(_moment_about_origin(force, point, moment), expected, atol=1e-9)
        self.assertAlmostEqual(r[0], 0.5)
        self.assertEqual(r[2], 1.0)  # CoP beyond the right foot is clipped.

    def test_lifted_foot_gets_no_load(self) -> None:
        on = np.ones(3, dtype=bool)
        force, _, _, r = split_feet(
            self.force, self.cop, self.free, self.foot_l, self.foot_r, on, ~on
        )
        np.testing.assert_allclose(r, 0.0)
        np.testing.assert_allclose(force[:, 1], 0.0)


class PlateWrenchTest(unittest.TestCase):
    def test_cop_and_free_moment_are_recovered_from_plate_moments(self) -> None:
        t_ms = np.arange(2000)
        action = np.array([-6.0, 4.0, -930.0])  # force on the plate
        cop = np.array([0.38, 0.47, 0.0])
        tz = 3.0
        centre = np.array([PLATE_SIZE_M / 2, PLATE_SIZE_M / 2, 0.0])
        moment_nmm = (np.cross(cop - centre, action) + [0.0, 0.0, tz]) * 1000.0
        row = [*action, *moment_nmm, *(cop * 1000.0)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "plate.csv"
            with path.open("w") as handle:
                handle.write("timestamp,Fx,Fy,Fz,Mx,My,Mz,Cx,Cy,Cz\n")
                for t in t_ms:
                    handle.write(",".join(str(v) for v in [t, *row]) + "\n")
            force, got_cop, free = plate_wrench(path, np.array([0.5, 1.0]), cutoff_hz=6.0)
        np.testing.assert_allclose(force, np.tile(-action, (2, 1)), atol=1e-6)
        np.testing.assert_allclose(got_cop, np.tile(cop, (2, 1)), atol=1e-9)
        np.testing.assert_allclose(free, -tz, atol=1e-6)


class UnwrapAnglesTest(unittest.TestCase):
    def test_full_turn_jump_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion = Path(tmpdir) / "motion.mot"
            motion.write_text(
                "Coordinates\nendheader\ntime shoulder_l_x pelvis_tx\n"
                "0.00 57.2 0.5\n0.01 423.7 0.6\n0.02 419.3 0.7\n"
            )
            unwrap_angles(motion)
            values = np.loadtxt(motion.read_text().split("endheader\n", 1)[1].splitlines()[1:])
        np.testing.assert_allclose(values[:, 1], [57.2, 63.7, 59.3])
        np.testing.assert_allclose(values[:, 2], [0.5, 0.6, 0.7])


class MergeIkPartsTest(unittest.TestCase):
    def test_warmup_rows_are_discarded_and_boundaries_are_deduplicated(self) -> None:
        header = "Coordinates\nversion=1\nnRows=3\nnColumns=2\ninDegrees=yes\nendheader\ntime pelvis_tx\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            source_trc = Path(tmpdir) / "source.trc"
            sliced_trc = Path(tmpdir) / "slice.trc"
            source_trc.write_text(
                "PathFileType 4\nDataRate CameraRate NumFrames NumMarkers Units OrigDataRate OrigDataStartFrame OrigNumFrames\n"
                "100 100 4 1 m 100 1 4\nFrame# Time M\nX1 Y1 Z1\n\n"
                "1 0.00 0 0 0\n2 0.01 1 1 1\n3 0.02 2 2 2\n4 0.03 3 3 3\n"
            )
            write_trc_slice(source_trc, sliced_trc, 1, 3, 4)
            trc_lines = sliced_trc.read_text().splitlines()
            self.assertEqual(trc_lines[2].split()[2:], ["2", "1", "m", "100", "2", "4"])
            self.assertEqual(trc_lines[6:], ["2 0.01 1 1 1", "3 0.02 2 2 2"])

            motion = Path(tmpdir) / "motion.mot"
            rows = [f"{i / 100:.2f} {20.0 * (-1) ** i:.1f} 0.5" for i in range(101)]
            motion.write_text(
                "Coordinates\nversion=1\nnRows=101\nnColumns=3\ninDegrees=yes\n"
                "endheader\ntime wrist_flexion_r pelvis_tx\n" + "\n".join(rows) + "\n"
            )
            smooth_weak_coordinate_trajectories(motion, ["wrist_flexion_r"], 100.0, 6.0)
            motion_lines = motion.read_text().split("endheader\n", 1)[1].splitlines()[1:]
            motion_values = np.loadtxt(motion_lines)
            self.assertLess(float(np.std(motion_values[:, 1])), 5.0)
            np.testing.assert_allclose(motion_values[:, 2], 0.5)

            set_coordinate_prior_targets(motion, {"wrist_flexion_r": 12.5})
            targeted_lines = motion.read_text().split("endheader\n", 1)[1].splitlines()[1:]
            targeted_values = np.loadtxt(targeted_lines, ndmin=2)
            np.testing.assert_allclose(targeted_values[:, 1], 12.5)

            first = Path(tmpdir) / "first.mot"
            second = Path(tmpdir) / "second.mot"
            output = Path(tmpdir) / "merged.mot"
            first.write_text(header + "0.0 10\n1.0 11\n")
            second.write_text(header + "0.5 90\n1.0 12\n2.0 13\n")

            merge_mot_parts([first, second], output, 3, keep_from_times=[0.0, 1.0])

            rows = output.read_text().split("endheader\n", 1)[1].splitlines()[1:]
        self.assertEqual(rows, ["0.0 10", "1.0 11", "2.0 13"])


if __name__ == "__main__":
    unittest.main()

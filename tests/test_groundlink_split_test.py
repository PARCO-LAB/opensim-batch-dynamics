import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from groundlink_split_test import (  # noqa: E402
    aggregate_feet_wrench,
    contact_from_kinematics,
    hybrid_split,
    wrench_from_points,
)


class GroundLinkSplitMathTest(unittest.TestCase):
    def test_aggregate_plate_wrench_round_trip(self):
        force_l = np.array([[32.0, -7.0, 280.0], [13.0, 5.0, 110.0]])
        force_r = np.array([[-11.0, 4.0, 260.0], [8.0, -3.0, 140.0]])
        point_l = np.array([[0.1, 0.25, 0.0], [0.0, 0.2, 0.0]])
        point_r = np.array([[0.4, -0.15, 0.0], [0.35, -0.1, 0.0]])
        moment_l = np.array([[0.0, 0.0, 1.2], [0.0, 0.0, -0.3]])
        moment_r = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.5]])
        total_force, cop, free_moment = aggregate_feet_wrench(
            force_l, point_l, moment_l, force_r, point_r, moment_r
        )
        source_moment = wrench_from_points(
            np.stack([force_l, force_r], axis=1),
            np.stack([point_l, point_r], axis=1),
            np.stack([moment_l, moment_r], axis=1),
        )
        plate_moment = np.cross(cop, total_force)
        plate_moment[:, 2] += free_moment
        np.testing.assert_allclose(total_force, force_l + force_r, atol=1e-12)
        np.testing.assert_allclose(plate_moment, source_moment, atol=1e-12)

    def test_hybrid_split_preserves_total_wrench(self):
        total_force = np.array([[18.0, -5.0, 620.0], [-9.0, 8.0, 450.0]])
        total_cop = np.array([[0.23, 0.12, 0.0], [0.17, -0.04, 0.0]])
        total_tz = np.array([2.4, -1.1])
        estimated_force = np.array(
            [[[1.0, 0.0, 180.0], [2.0, -1.0, 420.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        )
        estimated_cop = np.array(
            [[[0.1, 0.2, 0.0], [0.4, -0.1, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        )
        foot_l = np.array([[0.0, 0.2, 0.0], [0.0, 0.2, 0.0]])
        foot_r = np.array([[0.4, -0.1, 0.0], [0.4, -0.1, 0.0]])
        lever_fraction = np.array([0.4, 0.7])
        forces, points, moments, fractions = hybrid_split(
            total_force, total_cop, total_tz, estimated_force,
            estimated_cop, foot_l, foot_r, lever_fraction,
        )
        np.testing.assert_allclose(forces.sum(axis=1), total_force, atol=1e-12)
        np.testing.assert_allclose(
            wrench_from_points(forces, points, moments),
            np.cross(total_cop, total_force) + np.column_stack([np.zeros(2), np.zeros(2), total_tz]),
            atol=1e-12,
        )
        np.testing.assert_allclose(fractions, [0.7, 0.7])

    def test_kinematic_contact_uses_lower_foot_origin_and_segment_minimum(self):
        calcn_l = np.array([0.010, 0.080, 0.035, 0.12])
        toes_l = np.array([0.025, 0.090, 0.055, 0.13])
        calcn_r = np.array([0.075, 0.015, 0.050, 0.12])
        toes_r = np.array([0.090, 0.025, 0.060, 0.14])
        left, right = contact_from_kinematics(calcn_l, toes_l, calcn_r, toes_r)
        np.testing.assert_array_equal(left, [True, False, True, False])
        np.testing.assert_array_equal(right, [False, True, False, False])


if __name__ == "__main__":
    unittest.main()

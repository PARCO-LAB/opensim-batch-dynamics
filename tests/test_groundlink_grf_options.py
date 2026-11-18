import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "metrics_tau_claude"))
from loso_prototypes import apply_variant, horizontal_features, horizontal_local_features
from opensim_batch_dynamics.inverse_dynamics_no_grf import (
    _cop_progress_points,
    _horizontal_split_targets,
    _parse_cop_progress_model,
    _phase_cop_progress,
)


def sample_data():
    n = 4
    activation = np.ones((n, 2, 2))
    body_force = np.zeros((n, 4, 3))
    body_force[:, :, 2] = 100.0
    body_point = np.zeros((n, 4, 3))
    positions = np.zeros((n, 4, 3))
    positions[:, 0, 0] = -0.1
    positions[:, 1, 0] = 0.1
    positions[:, 2, 0] = 0.5
    positions[:, 3, 0] = 0.7
    times = np.arange(n, dtype=float) / 100
    net_force = np.tile([5.0, -2.0, 400.0], (n, 1))
    return {
        "times": times,
        "activation": activation,
        "body_force": body_force,
        "body_point": body_point,
        "body_positions": positions,
        "com": np.tile([0.2, 0.0, 0.9], (n, 1)),
        "floor_height": 0.0,
        "net_force": net_force,
        "bw_n": 700.0,
        "base_contact": np.ones((n, 2), dtype=bool),
    }


class GroundLinkGRFOptionTests(unittest.TestCase):
    def test_default_variant_preserves_wrenches(self):
        data = sample_data()
        result = apply_variant(data, "BASELINE", {})
        np.testing.assert_array_equal(result["body_force"], data["body_force"])
        np.testing.assert_array_equal(result["body_point"], data["body_point"])

    def test_vertical_split_leaves_estimated_single_stance_unchanged(self):
        data = sample_data()
        data["activation"][2:, 1, :] = 0.0
        before = data["body_force"].copy()
        params = {"S_coefficients": [0.5, 0.0, 0.0, 0.0], "S_contact_threshold": 0.5}
        result = apply_variant(data, "S", params)
        np.testing.assert_array_equal(result["body_force"][2:, :, 2], before[2:, :, 2])
        np.testing.assert_allclose(result["foot_force"][:2, :, 2].sum(axis=1), 400.0)

    def test_horizontal_split_preserves_net_force(self):
        data = sample_data()
        width = horizontal_features(data).shape[1]
        params = {"H_model": {"mean": [0.0] * width, "scale": [1.0] * width,
                              "coefficients": [[0.0, 0.0]] * (width + 1)}}
        result = apply_variant(data, "H", params)
        np.testing.assert_allclose(result["foot_force"].sum(axis=1)[:, :2],
                                   data["net_force"][:, :2], atol=1e-12)

    def test_hsym_mirrors_lateral_features_and_preserves_net_force(self):
        data = sample_data()
        data["body_positions"][:, 2:, 1] = 0.2
        local = horizontal_local_features(data)
        expected_right_ml = -(0.2 - data["com"][0, 1]) / data["com"][0, 2]
        np.testing.assert_allclose(local[:, 1, 4], expected_right_ml)
        np.testing.assert_allclose(local[:, 1, 8], -data["net_force"][:, 1] / data["bw_n"])

        params = {"H_model": {
            "architecture": "shared_per_foot_mirrored_y",
            "residual_allocation": "equal",
            "mean": [0.0] * 9,
            "scale": [1.0] * 9,
            "coefficients": [[0.0, 0.0]] * 10,
        }}
        result = apply_variant(data, "Hsym", params)
        np.testing.assert_allclose(result["foot_force"].sum(axis=1)[:, :2],
                                   data["net_force"][:, :2], atol=1e-12)

    def test_phase_cop_schema_and_estimated_phase_selection(self):
        legacy = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
        np.testing.assert_array_equal(_parse_cop_progress_model({"coefficients": legacy.tolist()}), legacy)
        model = {"coefficients_by_phase": {
            "single_stance": [0.0] * 5, "double_stance": [1.0, 0.0, 0.0, 0.0, 0.0],
        }}
        parsed = _parse_cop_progress_model(model)
        self.assertIsInstance(parsed, dict)
        data = sample_data()
        data["activation"][1, 1, :] = 0.0
        data["activation"][2, :, :] = 0.0
        contact = data["activation"].reshape(4, 4).reshape(4, 2, 2).max(axis=2) > 0.5
        features = np.zeros((4, 4))
        progress = _phase_cop_progress(features, contact, parsed)
        np.testing.assert_allclose(progress[:2], [1.0, 0.0])
        self.assertTrue(np.isnan(progress[2]))
        np.testing.assert_allclose(progress[3], 1.0)

    def test_phase_cop_variant_uses_estimated_contacts_without_measured_arrays(self):
        data = sample_data()
        data["activation"][1, 1, :] = 0.0
        data["activation"][2, :, :] = 0.0
        params = {"Pphase_coefficients": {
            "single_stance": [0.0] * 5, "double_stance": [1.0, 0.0, 0.0, 0.0, 0.0],
        }}
        result = apply_variant(data, "Pphase", params)
        self.assertEqual(result["body_point"].shape, (4, 4, 3))
        np.testing.assert_allclose(result["body_point"][0, 0, 0], 0.1)
        np.testing.assert_allclose(result["body_point"][1, 0, 0], -0.1)
        # Flight frame keeps its default application points.
        np.testing.assert_array_equal(result["body_point"][2], data["body_point"][2])

    def test_cop_progress_uses_kinematics_only(self):
        heel = np.zeros((2, 3))
        toes = np.tile([1.0, 0.0, 0.0], (2, 1))
        features = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        points = _cop_progress_points(heel, toes, features, np.array([0.0, 1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(points[:, 0], [0.0, 1.0])

    def test_pipeline_split_sum_constraint_uses_only_estimated_inputs(self):
        net = np.tile([4.0, -3.0], (3, 1))
        contact = np.array([[True, True], [True, False], [False, True]])
        predicted_left = np.tile([10.0, 6.0], (3, 1))
        target = _horizontal_split_targets(net, contact, predicted_left)
        np.testing.assert_allclose(target.sum(axis=1), net, atol=1e-12)
        self.assertEqual(target.shape, (3, 2, 2))

    def test_inference_does_not_require_measured_inputs(self):
        data = sample_data()
        width = horizontal_features(data).shape[1]
        params = {"H_model": {"mean": [0.0] * width, "scale": [1.0] * width,
                              "coefficients": [[0.0, 0.0]] * (width + 1)}}
        result = apply_variant(data, "H", params)
        self.assertEqual(result["foot_force"].shape, (4, 2, 3))


if __name__ == "__main__":
    unittest.main()

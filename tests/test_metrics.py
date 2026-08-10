from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opensim_batch_dynamics.metrics import (  # noqa: E402
    binary_classification_metrics,
    include_in_precision_metrics,
    per_column_rmse,
    rmse,
    top_k_rmse,
)
from opensim_batch_dynamics.realtime_dynamics import (  # noqa: E402
    RealtimeConfig,
    RealtimeFrameResult,
    RealtimeState,
    RealtimeWorkspace,
    qpid,
)


class MetricsModuleTest(unittest.TestCase):
    def test_shared_metrics_and_wrapper_import(self) -> None:
        self.assertTrue(include_in_precision_metrics("hip_flexion_r"))
        self.assertFalse(include_in_precision_metrics("wrist_flexion_r"))
        self.assertAlmostEqual(rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])), np.sqrt(2.0))
        self.assertEqual(
            binary_classification_metrics(np.array([1, 0, 1]), np.array([1, 1, 0]))["f1"],
            0.5,
        )
        self.assertEqual(per_column_rmse(np.array([[1.0, 2.0]]), np.array([[0.0, 2.0]]), ["a", "b"]), [("a", 1.0), ("b", 0.0)])
        self.assertEqual(top_k_rmse(np.array([[1.0, 3.0]]), np.array([[0.0, 0.0]]), ["a", "b"], k=1), [("b", 3.0)])
        self.assertTrue(callable(qpid))

    def test_workspaces_do_not_share_cache_objects(self) -> None:
        left = RealtimeWorkspace()
        right = RealtimeWorkspace()

        self.assertIsNot(left.stage1_cache, right.stage1_cache)
        self.assertIsNot(left.stage2_cache, right.stage2_cache)

    def test_public_runtime_dataclasses_exist(self) -> None:
        state = RealtimeState(q=np.zeros(1), dq=np.zeros(1), ddq=np.zeros(1), tau_full=np.zeros(1), root_residual=np.zeros(6))
        config = RealtimeConfig()
        frame = RealtimeFrameResult(q=np.zeros(1), dq=np.zeros(1), ddq=np.zeros(1), tau=np.zeros(1), tau_full=np.zeros(1), root_residual=np.zeros(6))

        self.assertEqual(config.steps, 1)
        self.assertEqual(state.step_index, 0)
        self.assertEqual(frame.dynamics_residual_norm, 0.0)


if __name__ == "__main__":
    unittest.main()

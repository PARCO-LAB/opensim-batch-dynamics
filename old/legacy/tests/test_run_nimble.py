from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_nimble as nimble  # noqa: E402


class RunNimbleTest(unittest.TestCase):
    def test_build_positions_raises_with_row_and_column_context(self) -> None:
        rows = [{"time": "0.0", "hip": "bad"}]
        with self.assertRaisesRegex(ValueError, r"row 1, column 'hip'"):
            nimble._build_positions(rows, ["hip"], np.array([0.0], dtype=np.float64))

    def test_build_positions_keeps_translational_dofs_in_meters(self) -> None:
        rows = [{"time": "0.0", "hip": "180.0", "pelvis_tx": "1.5"}]
        q_matrix, time_values = nimble._build_positions(
            rows,
            ["hip", "pelvis_tx"],
            np.array([0.0, 0.0], dtype=np.float64),
        )

        self.assertIsNotNone(time_values)
        np.testing.assert_allclose(q_matrix[0], np.asarray([np.pi, 1.5], dtype=np.float64))
        np.testing.assert_allclose(time_values, np.asarray([0.0], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()

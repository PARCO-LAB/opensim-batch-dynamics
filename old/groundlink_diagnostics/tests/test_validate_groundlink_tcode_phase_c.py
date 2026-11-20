from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_groundlink_tcode_phase_c import _lag_correlation  # noqa: E402


class TCodePhaseCTest(unittest.TestCase):
    def test_lag_correlation_skips_lags_longer_than_a_segment(self) -> None:
        correlation, count = _lag_correlation(
            np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0]), 50
        )
        self.assertTrue(np.isnan(correlation))
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from unittest import mock

import numpy as np

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opensim_batch_dynamics.inverse_dynamics_no_grf import _lowpass_butterworth_4th as inverse_lowpass  # noqa: E402
from opensim_batch_dynamics.mot_to_csv import _lowpass_butterworth_4th as mot_lowpass  # noqa: E402


class LowpassButterworthTest(unittest.TestCase):
    def test_short_signals_are_returned_unchanged(self) -> None:
        values = np.arange(5.0)

        filtered = mot_lowpass(values, sample_rate_hz=120.0, cutoff_hz=12.0)

        self.assertIsNot(filtered, values)
        np.testing.assert_array_equal(filtered, values)

    def test_mot_converter_raises_when_scipy_is_missing(self) -> None:
        values = np.arange(24.0)

        with mock.patch.dict(sys.modules, {"scipy": None, "scipy.signal": None}):
            with self.assertRaises(RuntimeError):
                mot_lowpass(values, sample_rate_hz=120.0, cutoff_hz=12.0)

    def test_inverse_dynamics_falls_back_to_copy_when_scipy_is_missing(self) -> None:
        values = np.arange(24.0)

        with mock.patch.dict(sys.modules, {"scipy": None, "scipy.signal": None}):
            filtered = inverse_lowpass(
                values,
                sample_rate_hz=120.0,
                cutoff_hz=12.0,
                missing_scipy_returns_copy=True,
            )

        self.assertIsNot(filtered, values)
        np.testing.assert_array_equal(filtered, values)


if __name__ == "__main__":
    unittest.main()

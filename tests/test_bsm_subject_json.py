from __future__ import annotations

import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from opensim_batch_dynamics.amass_loader import AMASSSequence  # noqa: E402
from opensim_batch_dynamics.bsm_subject_json import (  # noqa: E402
    SubjectMeasurementEstimate,
    build_subject_json,
)


def _sequence() -> AMASSSequence:
    zeros = np.zeros((1, 3), dtype=np.float32)
    return AMASSSequence(Path("sample.npz"), "smplx", "female", 120.0, zeros, zeros,
                         np.zeros((1, 63), dtype=np.float32), np.zeros((1, 45), dtype=np.float32),
                         np.zeros((1, 45), dtype=np.float32), zeros, zeros, zeros, np.zeros(10))


class BsmSubjectJsonTest(unittest.TestCase):
    def test_explicit_measurements_bypass_estimator(self) -> None:
        with patch("opensim_batch_dynamics.bsm_subject_json.estimate_subject_measurements") as estimate:
            result = build_subject_json(_sequence(), "unused", subject_mass_kg=69.86, subject_height_m=1.68)
        estimate.assert_not_called()
        self.assertEqual(result["massKg"], 69.86)
        self.assertEqual(result["heightM"], 1.68)

    def test_missing_measurements_keep_estimator_fallback(self) -> None:
        with patch("opensim_batch_dynamics.bsm_subject_json.estimate_subject_measurements",
                   return_value=SubjectMeasurementEstimate(70.0, 1.7)):
            result = build_subject_json(_sequence(), "unused")
        self.assertEqual(result["massKg"], 70.0)
        self.assertEqual(result["heightM"], 1.7)


if __name__ == "__main__":
    unittest.main()

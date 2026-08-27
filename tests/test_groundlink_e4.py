from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_groundlink_e4 import _trial_metrics  # noqa: E402


class GroundLinkE4Test(unittest.TestCase):
    def test_zero_error_and_normalized_residuals(self) -> None:
        names = ["hip_flexion_l_tau", "knee_angle_l_tau", "ankle_angle_l_tau", "pelvis_tx_tau", "pelvis_ty_tau", "pelvis_tz_tau", "pelvis_tilt_tau", "pelvis_list_tau", "pelvis_rotation_tau"]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tau.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["time", *names]); writer.writeheader(); writer.writerow({"time": 0, **{name: 0 for name in names}})
            metrics = _trial_metrics(path, path, 10.0, 2.0)
        self.assertEqual(metrics["hip_rmse_mgh"], 0.0)
        self.assertEqual(metrics["residual_force_pct_bw"], 0.0)

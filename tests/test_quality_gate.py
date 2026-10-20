from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from opensim_batch_dynamics.quality import assess_csv


class QualityGateTest(unittest.TestCase):
    def _write_csv(self, rows: list[dict[str, float]]) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "trial.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_gimbal_coupling_is_not_a_physical_rotation_spike(self) -> None:
        rows = []
        for frame in range(300):
            time = frame * 0.004
            rows.append({
                "time": time,
                "pelvis_tilt": -4.0 - 0.132 * frame,
                "pelvis_list": 1.543,
                "pelvis_rotation": -0.4 + 0.132 * frame,
            })
        report = assess_csv(self._write_csv(rows))
        self.assertEqual(report["status"], "ok")

    def test_true_orientation_spike_is_rejected(self) -> None:
        rows = []
        for frame in range(300):
            rows.append({
                "time": frame * 0.004,
                "pelvis_tilt": 0.2 if frame >= 150 else 0.0,
                "pelvis_list": 0.0,
                "pelvis_rotation": 0.0,
            })
        report = assess_csv(self._write_csv(rows))
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["reasons"], ["pelvis_rotation_speed_gt_20rad_s"])


if __name__ == "__main__":
    unittest.main()

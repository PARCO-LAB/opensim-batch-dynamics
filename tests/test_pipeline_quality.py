from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_pipeline_quality import score_pipeline, score_wrench_csv  # noqa: E402


HEADER = [
    "time", "target_force_x", "target_force_y", "target_force_z",
    "achieved_force_x", "achieved_force_y", "achieved_force_z", "force_balance_residual_norm",
]


class PipelineQualityTest(unittest.TestCase):
    def _write_wrench(self, path: Path, rows: list[list[float]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            writer.writerows(rows)

    def test_score_is_one_for_balanced_supported_wrenches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wrench = Path(tmpdir) / "wrench.csv"
            self._write_wrench(wrench, [[0, 0, 0, 98.1, 0, 0, 98.1, 0], [0.01, 0, 0, 98.1, 0, 0, 98.1, 0]])
            score = score_wrench_csv(wrench, 10.0)
        self.assertEqual(score["quality_score"], 1.0)
        self.assertEqual(score["unsupported_target_fraction"], 0.0)

    def test_score_falls_for_unsupported_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            wrench = Path(tmpdir) / "wrench.csv"
            self._write_wrench(wrench, [[0, 0, 0, 98.1, 0, 0, 0, 98.1]])
            score = score_wrench_csv(wrench, 10.0)
        self.assertEqual(score["quality_score"], 0.0)
        self.assertEqual(score["unsupported_target_fraction"], 1.0)

    def test_pipeline_report_finds_hashed_wrench(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final = root / "s007_trial_stageii.csv"
            final.write_text("subject_mass_kg\n10\n", encoding="utf-8")
            wrench = root / "s007_trial_stageii_hash" / "results" / "ID_estimatedGRF" / "w_contact_wrenches_estimated.csv"
            wrench.parent.mkdir(parents=True)
            self._write_wrench(wrench, [[0, 0, 0, 98.1, 0, 0, 98.1, 0]])
            rows = score_pipeline(root)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["frames"], 1)


if __name__ == "__main__":
    unittest.main()

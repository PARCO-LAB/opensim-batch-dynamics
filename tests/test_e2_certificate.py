from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from certify_groundlink_e2 import METHODS, METRICS, SUBJECTS, render, summarize  # noqa: E402


class E2CertificateTest(unittest.TestCase):
    def test_summary_requires_and_aggregates_all_seven_subjects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index, subject in enumerate(SUBJECTS, start=1):
                path = root / subject / "metrics_per_trial.csv"; path.parent.mkdir()
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["method", "metric", "value"]); writer.writeheader()
                    for method in METHODS:
                        for metric, _ in METRICS:
                            if metric != "pfa_planar_error" or method == "pipeline":
                                writer.writerow({"method": method, "metric": metric, "value": index / 10})
            summary = summarize(root)
        self.assertAlmostEqual(summary["pipeline"]["contact_f1"]["all"][0], 0.4)
        self.assertIn("**PASS:**", render(summary))


if __name__ == "__main__":
    unittest.main()

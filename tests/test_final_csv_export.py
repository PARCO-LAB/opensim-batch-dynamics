from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opensim_batch_dynamics.final_csv_export import export_final_csv


class FinalCsvExportTest(unittest.TestCase):
    def test_export_final_csv_keeps_blank_numeric_fields_as_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dof_csv = root / "dofs.csv"
            torque_csv = root / "torques.csv"
            output_csv = root / "final.csv"

            dof_csv.write_text(
                "\n".join(
                    [
                        "frame,time,hip,hip_vel,hip_acc",
                        "0,0.0,1.0,2.0,3.0",
                        "1,0.5,,4.0,5.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            torque_csv.write_text(
                "\n".join(
                    [
                        "time,hip_tau",
                        "0.0,10.0",
                        "0.5,11.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = export_final_csv(dof_csv, torque_csv, output_csv, contact_wrench_csv_path=None)

            with output_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(summary.frames, 2)
        self.assertEqual(summary.dof_names, ("hip",))
        self.assertEqual(summary.contact_body_names, ("calcn_l", "calcn_r"))
        self.assertEqual(rows[0][:6], ["frame", "time", "subject_mass_kg", "subject_height_m", "hip", "hip_vel"])
        self.assertAlmostEqual(float(rows[1][4]), 1.0)
        self.assertTrue(math.isnan(float(rows[2][4])))


if __name__ == "__main__":
    unittest.main()

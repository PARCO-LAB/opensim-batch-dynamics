from __future__ import annotations

import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opensim_batch_dynamics.addbio_csv_export import export_addbiomechanics_csv  # noqa: E402
from opensim_batch_dynamics.addbio_runner import _merge_segment_mots  # noqa: E402
from opensim_batch_dynamics.mot_to_csv import extract_coordinate_names_from_osim  # noqa: E402


class AddBiomechanicsCsvExportTest(unittest.TestCase):
    def test_merge_segment_mots_keeps_one_label_row_and_updates_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "segment_0.mot"
            second = root / "segment_1.mot"
            body = "name x\ndatacolumns 2\ndatarows 1\nrange 0 0\nendheader\ntime q\n"
            first.write_text(body + "0 1\n", encoding="utf-8")
            second.write_text(body.replace("range 0 0", "range 1 1") + "1 2\n", encoding="utf-8")
            merged = _merge_segment_mots([first, second], root / "merged.mot")
            text = merged.read_text(encoding="utf-8")
        self.assertIn("datarows 2", text)
        self.assertIn("range 0.00000000 1.00000000", text)
        self.assertEqual(text.count("time q"), 1)
        self.assertIn("0 1\n1 2", text)

    def test_export_addbiomechanics_csv_keeps_model_order_and_nan_fills_missing_dofs(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        model_path = repo_root / "model" / "bsm" / "bsm.osim"
        dof_names = extract_coordinate_names_from_osim(model_path)
        tracked_dof = dof_names[0]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            mot_path = tmp / "input.mot"
            mot_path.write_text(
                "\n".join(
                    [
                        "name test",
                        "endheader",
                        f"time /jointset/{tracked_dof}/value",
                        "0.0 0.0",
                        "0.5 1.0",
                        "1.0 2.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output_csv = tmp / "out.csv"

            summary = export_addbiomechanics_csv(model_path, mot_path, output_csv)

            with output_csv.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(summary.output_csv_path, output_csv.resolve())
        self.assertEqual(summary.model_path, model_path.resolve())
        self.assertEqual(summary.mot_path, mot_path.resolve())
        self.assertEqual(summary.dof_names, tuple(dof_names))
        self.assertEqual(summary.frames, 3)
        self.assertEqual(summary.velocity_source, "numerical_derivative_fallback")
        self.assertEqual(rows[0][:4], ["frame", "time", dof_names[0], dof_names[1]])

        pos_offset = 2
        vel_offset = 2 + len(dof_names)
        acc_offset = 2 + 2 * len(dof_names)

        self.assertEqual(rows[1][0], "0")
        self.assertEqual(rows[1][1], "0.0")
        self.assertAlmostEqual(float(rows[1][pos_offset]), 0.0)
        self.assertTrue(math.isnan(float(rows[1][pos_offset + 1])))
        self.assertAlmostEqual(float(rows[1][vel_offset]), 2.0)
        self.assertTrue(math.isnan(float(rows[1][vel_offset + 1])))
        self.assertAlmostEqual(float(rows[1][acc_offset]), 0.0)
        self.assertTrue(math.isnan(float(rows[1][acc_offset + 1])))


if __name__ == "__main__":
    unittest.main()

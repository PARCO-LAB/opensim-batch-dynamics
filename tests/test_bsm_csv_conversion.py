from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opensim_batch_dynamics.bsm_csv_conversion import (  # noqa: E402
    CONVERTED_COLUMNS,
    convert_bsm_csv_to_bsm2,
)


class BsmCsvConversionTest(unittest.TestCase):
    @staticmethod
    def _model_path() -> Path:
        return Path(__file__).resolve().parents[1] / "model" / "bsm" / "bsm2.osim"

    def test_negates_only_reparameterized_columns_and_preserves_power(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "trial_stageii.csv"
            destination = root / "trial_stageii_bsm2.csv"
            header = ["frame", "unchanged", *CONVERTED_COLUMNS]
            values = [
                "0",
                "keep",
                "0.25",
                "2.0",
                "-3.0",
                "4.0",
                "-0.5",
                "-6.0",
                "7.0",
                "-8.0",
            ]
            with source.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows([header, values])

            summary = convert_bsm_csv_to_bsm2(
                source,
                destination,
                target_model_path=self._model_path(),
            )
            with destination.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(summary.rows, 1)
        self.assertEqual(rows[0], header)
        self.assertEqual(rows[1][0:2], ["0", "keep"])
        self.assertEqual(
            rows[1][2:],
            ["-0.25", "-2.0", "3.0", "-4.0", "0.5", "6.0", "-7.0", "8.0"],
        )
        old_power = float(values[3]) * float(values[5]) + float(values[7]) * float(
            values[9]
        )
        new_power = float(rows[1][3]) * float(rows[1][5]) + float(rows[1][7]) * float(
            rows[1][9]
        )
        self.assertAlmostEqual(old_power, new_power)

    def test_rejects_converted_values_outside_clamped_target_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "trial_stageii.csv"
            destination = root / "trial_stageii_bsm2.csv"
            row = {column: "0" for column in CONVERTED_COLUMNS}
            row["shoulder_l_x"] = "2.0"  # Converts to -2.0; bsm2 lower limit is -1.5.
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            with self.assertRaisesRegex(ValueError, "exceeds clamped target range"):
                convert_bsm_csv_to_bsm2(
                    source,
                    destination,
                    target_model_path=self._model_path(),
                )
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()

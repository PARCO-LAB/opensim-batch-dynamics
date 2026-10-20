from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from batch_common import (  # noqa: E402
    DYNAMICS_PIPELINE_VERSION,
    excluded_trial_reason,
    is_nonempty_file,
    is_quality_pass,
    is_valid_groundlink_csv,
    read_manifest_record,
)


class BatchCommonTest(unittest.TestCase):
    def test_excluded_trial_reason_matches_dataset_key(self) -> None:
        self.assertEqual(
            excluded_trial_reason(Path("s002_20220520_balletsmalljump_1_stageii.npz")),
            "nonballistic_flight_kinematics",
        )
        self.assertIsNone(
            excluded_trial_reason(Path("s002_20220520_lambadadance_1_stageii.npz"))
        )

    def test_read_manifest_record_returns_requested_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            manifest_path.write_text(
                "\n".join(
                    [
                        '{"index": 0, "name": "first"}',
                        '{"index": 1, "name": "second"}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            record = read_manifest_record(manifest_path, 1)

        self.assertEqual(record["index"], 1)
        self.assertEqual(record["name"], "second")

    def test_read_manifest_record_raises_for_out_of_range_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.jsonl"
            manifest_path.write_text('{"index": 0}\n', encoding="utf-8")

            with self.assertRaises(IndexError):
                read_manifest_record(manifest_path, 1)

    def test_is_nonempty_file_checks_non_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.csv"
            self.assertFalse(is_nonempty_file(path))

            path.write_text("", encoding="utf-8")
            self.assertFalse(is_nonempty_file(path))

            path.write_text("value\n1\n", encoding="utf-8")
            self.assertTrue(is_nonempty_file(path))

    def test_is_valid_groundlink_csv_rejects_nonfinite_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.csv"
            path.write_text(
                "subject_mass_kg,subject_height_m\n66.68,nan\n",
                encoding="utf-8",
            )
            self.assertFalse(is_valid_groundlink_csv(path))

            path.write_text(
                "subject_mass_kg,subject_height_m\n66.68,1.70\n",
                encoding="utf-8",
            )
            self.assertTrue(is_valid_groundlink_csv(path))

    def test_is_quality_pass_requires_current_dynamics_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.csv"
            sidecar = path.with_suffix(path.suffix + ".quality.json")
            path.write_text("value\n1\n", encoding="utf-8")

            sidecar.write_text('{"status": "ok"}\n', encoding="utf-8")
            self.assertFalse(is_quality_pass(path))

            sidecar.write_text(
                '{"status": "ok", "dynamics_pipeline_version": "old"}\n',
                encoding="utf-8",
            )
            self.assertFalse(is_quality_pass(path))

            sidecar.write_text(
                '{"status": "ok", "dynamics_pipeline_version": "'
                + DYNAMICS_PIPELINE_VERSION
                + '"}\n',
                encoding="utf-8",
            )
            self.assertTrue(is_quality_pass(path))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from batch_common import is_nonempty_file, read_manifest_record  # noqa: E402


class BatchCommonTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

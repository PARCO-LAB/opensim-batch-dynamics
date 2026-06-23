import pickle
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opensim_batch_dynamics.carepd_gait_severity import (  # noqa: E402
    ScoreRow,
    extract_score_rows,
    write_score_csv,
)


class CarepdGaitSeverityTest(unittest.TestCase):
    def test_extract_score_rows_uses_canonical_filename_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "3DGait_canonical.pkl"
            payload = {
                0: {
                    "vid0073_0055": {
                        "pose": np.zeros((2, 66), dtype=np.float32),
                        "trans": np.zeros((2, 3), dtype=np.float32),
                        "beta": np.zeros((10,), dtype=np.float32),
                        "fps": 30,
                        "UPDRS_GAIT": 0,
                    },
                    "vid0073_0152": {
                        "pose": np.zeros((1, 66), dtype=np.float32),
                        "trans": np.zeros((1, 3), dtype=np.float32),
                        "beta": np.zeros((10,), dtype=np.float32),
                        "fps": 30,
                        "UPDRS_GAIT": None,
                    },
                },
                1: {
                    "vid0077_0061": {
                        "pose": np.zeros((3, 66), dtype=np.float32),
                        "trans": np.zeros((3, 3), dtype=np.float32),
                        "beta": np.zeros((10,), dtype=np.float32),
                        "fps": 30,
                        "UPDRS_GAIT": 2,
                    }
                },
            }
            with source_path.open("wb") as handle:
                pickle.dump(payload, handle)

            rows, stats = extract_score_rows(root)

        self.assertEqual(
            [(row.filename, row.score) for row in rows],
            [
                ("3DGait_canonical__0__vid0073_0055", 0),
                ("3DGait_canonical__1__vid0077_0061", 2),
            ],
        )
        self.assertEqual(stats.pickle_files, 1)
        self.assertEqual(stats.rows_written, 2)
        self.assertEqual(stats.skipped_missing, 1)
        self.assertEqual(stats.skipped_invalid, 0)

    def test_write_score_csv_uses_semicolon_delimiter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "scores.csv"
            write_score_csv(
                [
                    ScoreRow(filename="a", score=1),
                    ScoreRow(filename="b", score=3),
                ],
                output_csv,
            )

            lines = output_csv.read_text(encoding="utf-8").splitlines()

        self.assertEqual(
            lines,
            [
                "filename;MDS-UPDRS gait severity score",
                "a;1",
                "b;3",
            ],
        )


if __name__ == "__main__":
    unittest.main()

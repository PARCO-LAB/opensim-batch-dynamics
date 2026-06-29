import csv
import pickle
import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_carepd_mocha_training_dataset as builder  # noqa: E402


def _write_labels(path: Path, rows: list[tuple[str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["filename", "MDS-UPDRS gait severity score"])
        writer.writerows(rows)


def _write_motion_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


class CarepdMochaTrainingDatasetTest(unittest.TestCase):
    def test_build_dataset_treats_blank_numeric_cells_as_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "features"
            output_dir = root / "out"
            input_dir.mkdir()
            labels_csv = root / "labels.csv"

            _write_labels(labels_csv, [("Study__S01__a", 0)])
            _write_motion_csv(
                input_dir / "Study__S01__a.csv",
                ["frame", "time", "hip", "hip_vel", "hip_acc"],
                [
                    [0, 0.0, 1.0, 2.0, 3.0],
                    [1, 0.5, "", 4.0, 5.0],
                ],
            )

            args = builder.parse_args(
                [
                    "--input-dir",
                    str(input_dir),
                    "--labels-csv",
                    str(labels_csv),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            summary = builder.build_dataset(args)

            self.assertEqual(summary["exported_samples"], 1)
            with (output_dir / "manifest.csv").open(encoding="utf-8") as handle:
                manifest_rows = list(csv.DictReader(handle))
            with np.load(output_dir / manifest_rows[0]["input_npz"]) as npz:
                self.assertEqual(npz["x"].shape, (2, 3))
                self.assertTrue(np.isnan(npz["x"][1, 0]))

    def test_build_dataset_writes_manifest_sequences_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "features"
            output_dir = root / "out"
            input_dir.mkdir()
            labels_csv = root / "labels.csv"

            _write_labels(
                labels_csv,
                [
                    ("Study_canonical__S01__walk_a", 2),
                    ("Study_canonical__S02__walk_b", 1),
                    ("Study_canonical__S99__missing", 3),
                ],
            )

            header = [
                "frame",
                "time",
                "subject_mass_kg",
                "hip_flexion_r",
                "hip_flexion_r_vel",
                "hip_flexion_r_acc",
                "knee_angle_r",
                "knee_angle_r_vel",
                "knee_angle_r_acc",
                "hip_flexion_r_tau",
                "calcn_r_grf_x",
                "calcn_r_contact",
            ]
            _write_motion_csv(
                input_dir / "Study_canonical__S01__walk_a.csv",
                header,
                [
                    [0, 0.0, 70.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 0.0, 0],
                    [1, 0.5, 70.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 11.0, 1.0, 1],
                ],
            )
            _write_motion_csv(
                input_dir / "Study_canonical__S02__walk_b.csv",
                header,
                [
                    [0, 0.0, 71.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 12.0, 0.0, 0],
                    [1, 0.25, 71.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 13.0, 1.0, 1],
                ],
            )
            _write_motion_csv(input_dir / "unlabeled.csv", header, [[0, 0.0, 70.0, 1, 2, 3, 4, 5, 6, 7, 0, 0]])

            args = builder.parse_args(
                [
                    "--input-dir",
                    str(input_dir),
                    "--labels-csv",
                    str(labels_csv),
                    "--output-dir",
                    str(output_dir),
                    "--norm-split",
                    "all",
                ]
            )
            summary = builder.build_dataset(args)

            self.assertEqual(summary["exported_samples"], 2)
            self.assertEqual(summary["feature_count"], 6)
            self.assertEqual(summary["label_without_csv_count"], 1)
            self.assertEqual(summary["csv_without_label_count"], 1)

            with (output_dir / "manifest.csv").open(encoding="utf-8") as handle:
                manifest_rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in manifest_rows], ["Study_canonical__S01__walk_a", "Study_canonical__S02__walk_b"])
            self.assertEqual(manifest_rows[0]["subject_id"], "S01")
            self.assertEqual(manifest_rows[0]["walk_id"], "walk_a")
            self.assertEqual(manifest_rows[0]["feature_count"], "6")

            with np.load(output_dir / manifest_rows[0]["input_npz"]) as npz:
                self.assertEqual(npz["x"].shape, (2, 6))
                self.assertEqual(npz["label"].item(), 2)
                self.assertEqual(npz["feature_names"].tolist(), builder.build_feature_names(["hip_flexion_r", "knee_angle_r"]))
                np.testing.assert_allclose(npz["time"], np.asarray([0.0, 0.5], dtype=np.float32))

            with np.load(output_dir / "norm_stats.npz") as stats:
                self.assertEqual(stats["feature_count"].item(), 6)
                self.assertEqual(stats["feature_names"].tolist(), builder.build_feature_names(["hip_flexion_r", "knee_angle_r"]))

    def test_feature_mismatch_fill_keeps_reference_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "features"
            output_dir = root / "out"
            input_dir.mkdir()
            labels_csv = root / "labels.csv"
            _write_labels(labels_csv, [("Study__S01__a", 0), ("Study__S02__b", 1)])

            full_header = ["frame", "time", "hip", "hip_vel", "hip_acc", "knee", "knee_vel", "knee_acc"]
            short_header = ["frame", "time", "hip", "hip_vel", "hip_acc"]
            _write_motion_csv(input_dir / "Study__S01__a.csv", full_header, [[0, 0.0, 1, 2, 3, 4, 5, 6]])
            _write_motion_csv(input_dir / "Study__S02__b.csv", short_header, [[0, 0.0, 7, 8, 9]])

            args = builder.parse_args(
                [
                    "--input-dir",
                    str(input_dir),
                    "--labels-csv",
                    str(labels_csv),
                    "--output-dir",
                    str(output_dir),
                    "--on-feature-mismatch",
                    "fill",
                ]
            )
            summary = builder.build_dataset(args)

            self.assertEqual(summary["exported_samples"], 2)
            self.assertEqual(summary["skipped_feature_mismatch_count"], 1)
            with (output_dir / "manifest.csv").open(encoding="utf-8") as handle:
                manifest_rows = list(csv.DictReader(handle))
            with np.load(output_dir / manifest_rows[1]["input_npz"]) as npz:
                self.assertEqual(npz["x"].shape, (1, 6))
                self.assertTrue(np.isnan(npz["x"][0, 3:]).all())

    def test_carepd_folds_assign_train_and_eval_by_participant(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_dir = root / "features"
            output_dir = root / "out"
            folds_dir = root / "folds"
            fold_subdir = folds_dir / "UPDRS_Datasets"
            input_dir.mkdir()
            fold_subdir.mkdir(parents=True)
            labels_csv = root / "labels.csv"
            _write_labels(labels_csv, [("Study_canonical__S01__a", 0), ("Study_canonical__S02__b", 1)])

            header = ["frame", "time", "hip", "hip_vel", "hip_acc"]
            _write_motion_csv(input_dir / "Study_canonical__S01__a.csv", header, [[0, 0.0, 1, 2, 3]])
            _write_motion_csv(input_dir / "Study_canonical__S02__b.csv", header, [[0, 0.0, 4, 5, 6]])

            with (fold_subdir / "Study_fixed.pkl").open("wb") as handle:
                pickle.dump({1: {"train": ["S01"], "eval": ["S02"]}}, handle)

            args = builder.parse_args(
                [
                    "--input-dir",
                    str(input_dir),
                    "--labels-csv",
                    str(labels_csv),
                    "--output-dir",
                    str(output_dir),
                    "--split-source",
                    "carepd-folds",
                    "--folds-dir",
                    str(folds_dir),
                ]
            )
            summary = builder.build_dataset(args)

            self.assertEqual(summary["split_source"], "carepd-folds")
            self.assertIn("Study", summary["carepd_fold_files"])
            with (output_dir / "manifest.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["split"] for row in rows], ["train", "eval"])


if __name__ == "__main__":
    unittest.main()

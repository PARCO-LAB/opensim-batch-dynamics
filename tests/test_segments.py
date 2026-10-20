from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from opensim_batch_dynamics.segments import (
    discover_segment_mot_paths,
    merge_csv_segments,
)


class SegmentUtilitiesTest(unittest.TestCase):
    def test_discovers_segments_in_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            final = root / "trial_ik.mot"
            final.write_text("placeholder", encoding="utf-8")
            for index in (10, 2, 0):
                (root / f"trial_segment_{index}_ik.mot").write_text("x", encoding="utf-8")
            paths = discover_segment_mot_paths(final)
        self.assertEqual([path.name for path in paths], [
            "trial_segment_0_ik.mot",
            "trial_segment_2_ik.mot",
            "trial_segment_10_ik.mot",
        ])

    def test_csv_merge_trims_internal_edges_and_renumbers_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = []
            for segment, times in enumerate(((0.0, 0.2, 0.4, 0.6), (0.8, 1.0, 1.2, 1.4))):
                path = root / f"segment_{segment}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["frame", "time", "value"])
                    writer.writerows((index, time, segment) for index, time in enumerate(times))
                paths.append(path)
            output, frames = merge_csv_segments(paths, root / "merged.csv", edge_trim_s=0.2)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(frames, 6)
        self.assertEqual([int(row["frame"]) for row in rows], list(range(6)))
        self.assertEqual([float(row["time"]) for row in rows], [0.0, 0.2, 0.4, 1.0, 1.2, 1.4])

    def test_csv_merge_keeps_a_short_tail_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = []
            segment_times = (
                (0.0, 0.2, 0.4, 0.6),
                (0.8, 0.804, 0.808),
            )
            for segment, times in enumerate(segment_times):
                path = root / f"segment_{segment}.csv"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["frame", "time", "value"])
                    writer.writerows((index, time, segment) for index, time in enumerate(times))
                paths.append(path)
            output, frames = merge_csv_segments(paths, root / "merged.csv", edge_trim_s=0.2)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(frames, 6)
        self.assertEqual([float(row["time"]) for row in rows], [0.0, 0.2, 0.4, 0.8, 0.804, 0.808])


if __name__ == "__main__":
    unittest.main()

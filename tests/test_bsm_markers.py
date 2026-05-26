from pathlib import Path
import unittest

from opensim_batch_dynamics.bsm_markers import load_bsm_marker_map


class BsmMarkerMapTest(unittest.TestCase):
    def test_bsm_toe_markers_use_distinct_smplx_vertices(self) -> None:
        marker_map = load_bsm_marker_map(Path("assets/smpl2ab/bsm_markers_smplx.yaml"))

        self.assertEqual(marker_map["LTOE"], 5770)
        self.assertEqual(marker_map["RTOE"], 8480)
        self.assertNotEqual(marker_map["LTOE"], marker_map["RTOE"])

    def test_default_bsm_toe_markers_match_asset_map(self) -> None:
        asset_map = load_bsm_marker_map(Path("assets/smpl2ab/bsm_markers_smplx.yaml"))
        fallback_map = load_bsm_marker_map(None)

        self.assertEqual(fallback_map["LTOE"], asset_map["LTOE"])
        self.assertEqual(fallback_map["RTOE"], asset_map["RTOE"])


if __name__ == "__main__":
    unittest.main()

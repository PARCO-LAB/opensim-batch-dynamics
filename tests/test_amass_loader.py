from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from opensim_batch_dynamics.amass_loader import load_amass_npz  # noqa: E402


class AmassLoaderTest(unittest.TestCase):
    def test_load_amass_npz_reads_zero_dim_frame_rate_scalar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.npz"
            np.savez(
                path,
                surface_model_type=np.array("smplx", dtype=object),
                gender=np.array("female", dtype=object),
                mocap_frame_rate=np.array(120.0, dtype=np.float32),
                trans=np.zeros((2, 3), dtype=np.float32),
                root_orient=np.zeros((2, 3), dtype=np.float32),
                pose_body=np.zeros((2, 63), dtype=np.float32),
                pose_hand=np.zeros((2, 90), dtype=np.float32),
                pose_jaw=np.zeros((2, 3), dtype=np.float32),
                pose_eye=np.zeros((2, 6), dtype=np.float32),
                betas=np.zeros((10,), dtype=np.float32),
            )

            sequence = load_amass_npz(path)

        self.assertEqual(sequence.frame_rate_hz, 120.0)
        self.assertEqual(sequence.gender, "female")
        self.assertEqual(sequence.n_frames, 2)
        self.assertEqual(sequence.body_pose.shape, (2, 63))


if __name__ == "__main__":
    unittest.main()

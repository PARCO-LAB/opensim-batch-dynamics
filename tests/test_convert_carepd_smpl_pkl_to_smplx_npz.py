from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class ConvertCarepdSmplPklTest(unittest.TestCase):
    def test_extract_take_record_reads_frame_rate_scalar(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = repo_root / "scripts"

        fake_smplx = types.ModuleType("smplx")
        fake_smplx.__path__ = []  # type: ignore[attr-defined]
        fake_transfer_model = types.ModuleType("smplx.transfer_model")
        fake_transfer_model.__path__ = []  # type: ignore[attr-defined]
        fake_transfer_model.merge_output = lambda *args, **kwargs: None
        fake_transfer_model_transfer = types.ModuleType("smplx.transfer_model.transfer_model")
        fake_transfer_model_transfer.prepare_fitting_assets = lambda *args, **kwargs: None
        fake_transfer_model_transfer.run_fitting = lambda *args, **kwargs: None
        fake_transfer_utils = types.ModuleType("smplx.transfer_model.utils")
        fake_transfer_utils.read_deformation_transfer = lambda *args, **kwargs: None
        fake_smplx.transfer_model = fake_transfer_model  # type: ignore[attr-defined]
        fake_transfer_model.transfer_model = fake_transfer_model_transfer  # type: ignore[attr-defined]
        fake_transfer_model.utils = fake_transfer_utils  # type: ignore[attr-defined]

        with mock.patch.dict(
            sys.modules,
            {
                "smplx": fake_smplx,
                "smplx.transfer_model": fake_transfer_model,
                "smplx.transfer_model.transfer_model": fake_transfer_model_transfer,
                "smplx.transfer_model.utils": fake_transfer_utils,
            },
        ):
            sys.path.insert(0, str(scripts_dir))
            module = importlib.import_module("convert_carepd_smpl_pkl_to_smplx_npz")

        record = {
            "gender": np.array("female", dtype=object),
            "mocap_frame_rate": np.array(60.0, dtype=np.float32),
            "trans": np.zeros((1, 3), dtype=np.float32),
            "root_orient": np.zeros((1, 3), dtype=np.float32),
            "body_pose": np.zeros((1, 63), dtype=np.float32),
            "betas": np.zeros((16,), dtype=np.float32),
        }

        poses, betas, gender, frame_rate_hz, diagnostics = module._extract_take_record(  # type: ignore[attr-defined]
            record=record,
            source_path=repo_root / "sample.pkl",
            take_name="take_001",
            fallback_frame_rate=30.0,
            num_betas=16,
        )

        self.assertEqual(frame_rate_hz, 60.0)
        self.assertEqual(gender, "female")
        self.assertEqual(poses.shape, (1, 66))
        self.assertEqual(betas.shape, (16,))
        self.assertEqual(diagnostics["source_frame_rate_hz"], 60.0)


if __name__ == "__main__":
    unittest.main()

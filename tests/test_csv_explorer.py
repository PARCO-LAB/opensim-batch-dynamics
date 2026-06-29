from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import types
from unittest import mock

import numpy as np


class CsvExplorerTest(unittest.TestCase):
    def test_load_motion_csv_treats_blank_numeric_cells_as_nan(self) -> None:
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.use = lambda *args, **kwargs: None  # type: ignore[assignment]
        pyplot = types.ModuleType("matplotlib.pyplot")
        backends = types.ModuleType("matplotlib.backends")
        backend_pdf = types.ModuleType("matplotlib.backends.backend_pdf")

        class _DummyPdfPages:
            pass

        backend_pdf.PdfPages = _DummyPdfPages

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "frame,time,hip,hip_vel,hip_acc",
                        "0,0.0,1.0,2.0,3.0",
                        "1,0.5,,4.0,5.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with mock.patch.dict(
                sys.modules,
                {
                    "matplotlib": matplotlib,
                    "matplotlib.pyplot": pyplot,
                    "matplotlib.backends": backends,
                    "matplotlib.backends.backend_pdf": backend_pdf,
                },
            ):
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
                from csv_explorer import load_motion_csv  # noqa: E402

                data = load_motion_csv(csv_path)

        self.assertEqual(data.n_frames, 2)
        self.assertEqual(data.time_source, "time_column")
        self.assertEqual(data.dof_names, ["hip"])
        self.assertTrue(np.isnan(data.values["hip"][1]))
        np.testing.assert_allclose(data.time, np.asarray([0.0, 0.5], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()

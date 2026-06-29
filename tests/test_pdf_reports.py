from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class PdfReportTest(unittest.TestCase):
    def _fake_matplotlib_modules(self) -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
        class FakeAxes:
            def __getattr__(self, _name: str):
                def _noop(*_args, **_kwargs) -> None:
                    return None

                return _noop

        class FakeFigure:
            def suptitle(self, *_args, **_kwargs) -> None:
                return None

            def text(self, *_args, **_kwargs) -> None:
                return None

            def add_subplot(self, *_args, **_kwargs) -> FakeAxes:
                return FakeAxes()

            def tight_layout(self, *_args, **_kwargs) -> None:
                return None

        class FakePdfPages:
            def __init__(self, output_path: Path) -> None:
                self.output_path = Path(output_path)
                self.handle = None

            def __enter__(self) -> "FakePdfPages":
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.handle = self.output_path.open("wb")
                self.handle.write(b"%PDF-FAKE\n")
                return self

            def savefig(self, *_args, **_kwargs) -> None:
                assert self.handle is not None
                self.handle.write(b"page\n")

            def __exit__(self, *_exc) -> None:
                assert self.handle is not None
                self.handle.close()

        def fake_figure(*_args, **_kwargs) -> FakeFigure:
            return FakeFigure()

        def fake_subplots(*args, **kwargs):
            nrows = int(args[0] if args else kwargs.get("nrows", 1))
            ncols = int(args[1] if len(args) > 1 else kwargs.get("ncols", 1))
            fig = FakeFigure()
            if nrows == 1 and ncols == 1:
                return fig, FakeAxes()
            if ncols == 1:
                axes = np.empty(nrows, dtype=object)
                for idx in range(nrows):
                    axes[idx] = FakeAxes()
                return fig, axes
            axes = np.empty((nrows, ncols), dtype=object)
            for row in range(nrows):
                for col in range(ncols):
                    axes[row, col] = FakeAxes()
            return fig, axes

        matplotlib = ModuleType("matplotlib")
        matplotlib.__path__ = []  # type: ignore[attr-defined]
        matplotlib.use = lambda *_args, **_kwargs: None  # type: ignore[assignment]
        pyplot = ModuleType("matplotlib.pyplot")
        pyplot.figure = fake_figure  # type: ignore[assignment]
        pyplot.subplots = fake_subplots  # type: ignore[assignment]
        pyplot.close = lambda *_args, **_kwargs: None  # type: ignore[assignment]
        backends = ModuleType("matplotlib.backends")
        backends.__path__ = []  # type: ignore[attr-defined]
        backend_pdf = ModuleType("matplotlib.backends.backend_pdf")
        backend_pdf.PdfPages = FakePdfPages
        return matplotlib, pyplot, backends, backend_pdf

    def _write_csv_explorer_fixture(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "frame,time,pelvis_tx,pelvis_tx_vel,pelvis_ty,pelvis_ty_vel,pelvis_tz,pelvis_tz_vel,pelvis_list,pelvis_list_vel,pelvis_rotation,pelvis_rotation_vel,pelvis_bend,pelvis_bend_vel",
                    "0,0.0,0.0,0.1,0.1,0.2,0.2,0.3,0.3,0.4,0.4,0.5,0.5,0.6",
                    "1,0.5,0.2,0.1,0.3,0.2,0.4,0.3,0.5,0.4,0.6,0.5,0.7,0.6",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def _write_realtime_fixture(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "frame,time,hip_flexion_r,hip_flexion_r_vel,hip_flexion_r_acc,hip_flexion_r_tau",
                    "0,0.0,0.0,0.1,0.2,0.3",
                    "1,0.5,0.2,0.1,0.2,0.3",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_csv_explorer_build_pdf_report_writes_nonempty_pdf(self) -> None:
        matplotlib, pyplot, backends, backend_pdf = self._fake_matplotlib_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "motion.csv"
            output_pdf = root / "motion_report.pdf"
            self._write_csv_explorer_fixture(csv_path)

            with mock.patch.dict(
                sys.modules,
                {
                    "matplotlib": matplotlib,
                    "matplotlib.pyplot": pyplot,
                    "matplotlib.backends": backends,
                    "matplotlib.backends.backend_pdf": backend_pdf,
                },
            ):
                import csv_explorer

                with mock.patch.object(csv_explorer, "plt", pyplot):
                    data = csv_explorer.load_motion_csv(csv_path)
                    summary = csv_explorer.build_pdf_report(
                        data,
                        output_pdf=output_pdf,
                        title="Motion",
                        max_dofs=1,
                        root_force_warning_n=75.0,
                        root_moment_warning_nm=25.0,
                    )

            self.assertTrue(output_pdf.exists())
            self.assertGreater(output_pdf.stat().st_size, 0)
            self.assertEqual(summary["dof_count"], 6)
            self.assertEqual(summary["dof_pages"], 1)
            self.assertEqual(summary["grf_pages"], 0)

    def test_realtime_vs_offline_build_pdf_report_writes_nonempty_pdf(self) -> None:
        matplotlib, pyplot, backends, backend_pdf = self._fake_matplotlib_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            offline_csv = root / "offline.csv"
            realtime_csv = root / "realtime.csv"
            output_pdf = root / "compare.pdf"
            self._write_realtime_fixture(offline_csv)
            self._write_realtime_fixture(realtime_csv)

            with mock.patch.dict(
                sys.modules,
                {
                    "matplotlib": matplotlib,
                    "matplotlib.pyplot": pyplot,
                    "matplotlib.backends": backends,
                    "matplotlib.backends.backend_pdf": backend_pdf,
                },
            ):
                import realtime_vs_offline_pdf

                with mock.patch.object(realtime_vs_offline_pdf, "plt", pyplot):
                    report = realtime_vs_offline_pdf.build_report(
                        Namespace(
                            offline_csv=offline_csv,
                            realtime_csv=realtime_csv,
                            output_pdf=None,
                            title="Compare",
                            max_dofs=1,
                        )
                    )
                    summary = realtime_vs_offline_pdf.build_pdf_report(
                        report,
                        output_pdf=output_pdf,
                        max_dofs=1,
                    )

            self.assertTrue(output_pdf.exists())
            self.assertGreater(output_pdf.stat().st_size, 0)
            self.assertEqual(report["frames"], 2)
            self.assertEqual(summary["diagnostics_pages"], 0)
            self.assertEqual(summary["dof_pages"], 1)
            self.assertEqual(summary["grf_pages"], 2)


if __name__ == "__main__":
    unittest.main()

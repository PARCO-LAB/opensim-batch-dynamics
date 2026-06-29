from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import types
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from report_common import (  # noqa: E402
    add_text_page,
    binary_classification_metrics,
    compute_smart_ylim,
    format_float,
    include_in_precision_metrics,
    is_translational_dof,
    mae,
    rmse,
    write_pdf_report,
)


class ReportCommonTest(unittest.TestCase):
    def test_format_and_metrics_helpers(self) -> None:
        self.assertEqual(format_float(None), "n/a")
        self.assertEqual(format_float(1.23456, 2), "1.23")
        self.assertTrue(is_translational_dof("pelvis_tx"))
        self.assertFalse(is_translational_dof("pelvis_rx"))
        self.assertTrue(include_in_precision_metrics("hip_flexion_r"))
        self.assertFalse(include_in_precision_metrics("wrist_flexion_r"))

        self.assertAlmostEqual(rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])), np.sqrt(2.0))
        self.assertAlmostEqual(mae(np.array([1.0, 2.0]), np.array([1.0, 4.0])), 1.0)

        metrics = binary_classification_metrics(np.array([1, 0, 1]), np.array([1, 1, 0]))
        self.assertAlmostEqual(metrics["accuracy"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_compute_smart_ylim_centers_when_requested(self) -> None:
        ylim = compute_smart_ylim(np.array([-2.0, 3.0]), min_span=1.0, center_on_zero=True)
        self.assertIsNotNone(ylim)
        assert ylim is not None
        self.assertAlmostEqual(abs(ylim[0]), abs(ylim[1]))
        self.assertLess(ylim[0], 0.0)
        self.assertGreater(ylim[1], 0.0)

    def test_add_text_page_renders_title_and_body(self) -> None:
        class FakeAxes:
            def __init__(self) -> None:
                self.axis_calls: list[str] = []
                self.text_args = None

            def axis(self, value: str) -> None:
                self.axis_calls.append(value)

            def text(self, *args, **kwargs) -> None:
                self.text_args = (args, kwargs)

        class FakeFigure:
            def __init__(self) -> None:
                self.suptitle_args = None
                self.axes = FakeAxes()

            def suptitle(self, *args, **kwargs) -> None:
                self.suptitle_args = (args, kwargs)

            def add_subplot(self, *_args, **_kwargs) -> FakeAxes:
                return self.axes

        class FakePdf:
            def __init__(self) -> None:
                self.saved = []

            def savefig(self, fig, bbox_inches=None) -> None:
                self.saved.append((fig, bbox_inches))

        fake_figure = FakeFigure()
        fake_pdf = FakePdf()
        fake_pyplot = types.ModuleType("matplotlib.pyplot")
        fake_pyplot.figure = lambda *args, **kwargs: fake_figure  # type: ignore[assignment]
        fake_pyplot.close = lambda fig: None  # type: ignore[assignment]
        fake_matplotlib = types.ModuleType("matplotlib")
        fake_matplotlib.__path__ = []  # type: ignore[attr-defined]

        with mock.patch.dict(
            sys.modules,
            {
                "matplotlib": fake_matplotlib,
                "matplotlib.pyplot": fake_pyplot,
            },
        ):
            add_text_page(fake_pdf, "Report Title", ["line 1", "line 2"])

        self.assertEqual(fake_figure.suptitle_args[0][0], "Report Title")
        self.assertEqual(fake_figure.axes.text_args[0][2], "line 1\nline 2")
        self.assertEqual(fake_pdf.saved[0][1], "tight")

    def test_write_pdf_report_uses_shared_backend(self) -> None:
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

        backend_pdf = types.ModuleType("matplotlib.backends.backend_pdf")
        backend_pdf.PdfPages = FakePdfPages

        with tempfile.TemporaryDirectory() as tmpdir:
            output_pdf = Path(tmpdir) / "report.pdf"
            with mock.patch.dict(
                sys.modules,
                {
                    "matplotlib.backends.backend_pdf": backend_pdf,
                },
            ):
                results = write_pdf_report(
                    output_pdf,
                    [
                        lambda pdf: pdf.savefig(object()) or "first",
                        lambda pdf: pdf.savefig(object()) or "second",
                    ],
                )
            self.assertEqual(results, ["first", "second"])
            self.assertGreater(output_pdf.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

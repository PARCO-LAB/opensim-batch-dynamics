import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree as ET

try:
    import opensim  # noqa: F401
except ImportError:
    opensim = None

from opensim_batch_dynamics.inverse_dynamics_no_grf import (
    _auto_contact_body_names,
    _prepare_model_for_inverse_dynamics,
    run_inverse_dynamics_with_estimated_grf,
)


@unittest.skipUnless(opensim is not None, "opensim is not installed")
class InverseDynamicsModelPrepTest(unittest.TestCase):
    def test_prepared_model_uses_z_up_gravity_for_pipeline_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            prepared = _prepare_model_for_inverse_dynamics(
                model_path=Path("model/bsm/bsm.osim"),
                output_dir=Path(tmpdir),
                trial_name="gravity_check",
            )

            gravity = ET.parse(prepared).getroot().findtext(".//gravity")

        self.assertEqual(gravity, "0 0 -9.8066499999999994")

    def test_auto_contact_bodies_keeps_all_non_ground_segments(self) -> None:
        names = [
            "ground",
            "pelvis",
            "talus_r",
            "calcn_r",
            "toes_r",
            "talus_l",
            "calcn_l",
            "toes_l",
        ]

        self.assertEqual(
            _auto_contact_body_names(names),
            ["pelvis", "talus_r", "calcn_r", "toes_r", "talus_l", "calcn_l", "toes_l"],
        )

    def test_estimated_grf_contact_detection_uses_original_marker_model(self) -> None:
        class FakeInverseDynamicsTool:
            def setName(self, value):
                pass

            def setModelFileName(self, value):
                pass

            def setCoordinatesFileName(self, value):
                pass

            def setExternalLoadsFileName(self, value):
                pass

            def setStartTime(self, value):
                pass

            def setEndTime(self, value):
                pass

            def setLowpassCutoffFrequency(self, value):
                pass

            def setResultsDir(self, value):
                pass

            def setOutputGenForceFileName(self, value):
                pass

            def printToXML(self, value):
                Path(value).write_text("<OpenSimDocument />", encoding="utf-8")

        fake_opensim = types.SimpleNamespace(
            Logger=types.SimpleNamespace(setLevelString=lambda value: None),
            InverseDynamicsTool=FakeInverseDynamicsTool,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_model = root / "marker_model.osim"
            original_model.write_text("<OpenSimDocument />", encoding="utf-8")
            prepared_model = root / "id_model_nomarkers.osim"
            prepared_model.write_text("<OpenSimDocument />", encoding="utf-8")
            ik_mot = root / "ik.mot"
            ik_mot.write_text("time\n0\n", encoding="utf-8")
            output_dir = root / "id"

            def touch_id_output(setup_xml: Path) -> None:
                (output_dir / "trial_id_estimatedGRF.sto").write_text("", encoding="utf-8")

            with mock.patch.dict("sys.modules", {"opensim": fake_opensim}):
                with mock.patch(
                    "opensim_batch_dynamics.inverse_dynamics_no_grf._prepare_model_for_inverse_dynamics",
                    return_value=prepared_model,
                ):
                    with mock.patch(
                        "opensim_batch_dynamics.inverse_dynamics_no_grf._infer_time_window_from_mot",
                        return_value=(0.0, 1.0),
                    ):
                        with mock.patch(
                            "opensim_batch_dynamics.inverse_dynamics_no_grf._estimate_contact_wrenches_from_kinematics",
                            return_value=(
                                output_dir / "grf.mot",
                                output_dir / "external.xml",
                                output_dir / "wrenches.csv",
                            ),
                        ) as estimate:
                            with mock.patch(
                                "opensim_batch_dynamics.inverse_dynamics_no_grf._run_opensim_tool",
                                side_effect=touch_id_output,
                            ):
                                run_inverse_dynamics_with_estimated_grf(
                                    model_path=original_model,
                                    ik_mot_path=ik_mot,
                                    output_dir=output_dir,
                                    trial_name="trial",
                                    cutoff_hz=12.0,
                                )

        self.assertEqual(
            estimate.call_args.kwargs["model_path"],
            original_model.resolve(),
        )


if __name__ == "__main__":
    unittest.main()

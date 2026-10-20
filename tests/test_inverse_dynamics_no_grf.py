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
    _contact_force_target_and_activation,
    _prepare_model_for_inverse_dynamics,
    _project_force_to_friction_cone,
    _root_force_target_from_id_sto,
    run_inverse_dynamics_with_estimated_grf,
)
import numpy as np


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

    def test_contact_confidence_does_not_scale_required_force(self) -> None:
        raw = np.array([10.0, 20.0, 981.0])
        target, activation, fallback = _contact_force_target_and_activation(
            raw,
            np.array([0.1, 0.1]),
            np.array([0.0, 0.01]),
            np.array([0.0, 0.0, 1.0]),
            100.0,
            9.81,
        )
        np.testing.assert_allclose(target, raw)
        np.testing.assert_allclose(activation, [0.1, 0.1])
        self.assertFalse(fallback)

    def test_contact_gap_falls_back_to_lowest_support(self) -> None:
        raw = np.array([0.0, 0.0, 981.0])
        target, activation, fallback = _contact_force_target_and_activation(
            raw,
            np.zeros(2),
            np.array([0.02, -0.01]),
            np.array([0.0, 0.0, 1.0]),
            100.0,
            9.81,
        )
        np.testing.assert_allclose(target, raw)
        np.testing.assert_allclose(activation, [0.0, 1.0])
        self.assertTrue(fallback)

    def test_airborne_gap_does_not_invent_foot_force(self) -> None:
        raw = np.array([0.0, 0.0, 981.0])
        target, activation, fallback = _contact_force_target_and_activation(
            raw,
            np.zeros(2),
            np.array([0.12, 0.18]),
            np.array([0.0, 0.0, 1.0]),
            100.0,
            9.81,
        )
        self.assertIsNone(target)
        np.testing.assert_allclose(activation, [0.0, 0.0])
        self.assertFalse(fallback)

    def test_fast_low_foot_does_not_trigger_contact_fallback(self) -> None:
        raw = np.array([0.0, 0.0, 981.0])
        target, activation, fallback = _contact_force_target_and_activation(
            raw,
            np.zeros(2),
            np.array([0.01, 0.03]),
            np.array([0.0, 0.0, 1.0]),
            100.0,
            9.81,
            low_speed=np.array([1.2, 0.8]),
            fallback_speed_threshold_mps=0.5,
        )
        self.assertIsNone(target)
        np.testing.assert_allclose(activation, [0.0, 0.0])
        self.assertFalse(fallback)

    def test_contact_target_with_tensile_normal_is_rejected(self) -> None:
        target, activation, fallback = _contact_force_target_and_activation(
            np.array([0.0, 0.0, -1.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            100.0,
            9.81,
        )
        self.assertIsNone(target)
        np.testing.assert_allclose(activation, [1.0, 0.0])
        self.assertFalse(fallback)

    def test_friction_projection_removes_tension_and_zero_normal_shear(self) -> None:
        up = np.array([0.0, 0.0, 1.0])

        np.testing.assert_allclose(
            _project_force_to_friction_cone(
                np.array([100.0, 0.0, 0.0]), up, friction_coeff=0.8
            ),
            np.zeros(3),
        )
        np.testing.assert_allclose(
            _project_force_to_friction_cone(
                np.array([100.0, 0.0, -20.0]), up, friction_coeff=0.8
            ),
            np.zeros(3),
        )
        np.testing.assert_allclose(
            _project_force_to_friction_cone(
                np.array([100.0, 0.0, 100.0]), up, friction_coeff=0.8
            ),
            np.array([80.0, 0.0, 100.0]),
        )

    def test_root_force_target_uses_no_grf_translation_forces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sto = Path(tmpdir) / "no_grf.sto"
            sto.write_text(
                "name no_grf\n"
                "datacolumns 4\n"
                "datarows 2\n"
                "range 0 1\n"
                "endheader\n"
                "time pelvis_tx_force pelvis_ty_force pelvis_tz_force\n"
                "0 1 2 3\n"
                "1 11 12 13\n",
                encoding="utf-8",
            )
            target = _root_force_target_from_id_sto(
                sto,
                np.array([0.0, 0.5, 1.0]),
            )

        np.testing.assert_allclose(
            target,
            [[1.0, 2.0, 3.0], [6.0, 7.0, 8.0], [11.0, 12.0, 13.0]],
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
                                "opensim_batch_dynamics.inverse_dynamics_no_grf.run_inverse_dynamics_no_grf",
                                return_value=(
                                    output_dir / "target_setup.xml",
                                    output_dir / "target.sto",
                                    12.0,
                                ),
                            ) as target_id:
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
        self.assertEqual(
            estimate.call_args.kwargs["force_target_id_sto_path"],
            output_dir / "target.sto",
        )
        self.assertEqual(target_id.call_args.kwargs["cutoff_hz"], 12.0)
        self.assertEqual(target_id.call_args.kwargs["ik_mot_path"], ik_mot.resolve())


if __name__ == "__main__":
    unittest.main()

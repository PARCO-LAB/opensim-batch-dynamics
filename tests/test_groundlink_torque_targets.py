from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from extract_groundlink_torque_targets import (  # noqa: E402
    _load_h1_mask,
    _load_cop_shifts,
    _load_measured_wrenches,
    _patch_original_setup,
    _clone_setup_with_forces_to_exclude,
    _resample_wrenches,
    _filter_ik_mot_explicit_order4,
    _discover_segment_setup_paths,
    _resume_output_is_current,
    _write_measured_load_files,
    _shift_mot_add_tcode,
    _verify_mot_add_tcode,
    _max_abs_id_csv_difference,
    TARGET_COORDINATES,
)
from apply_groundlink_cop_shifts import _shift_mot, _verify_mot  # noqa: E402
from rerun_groundlink_id_nomuscles import (  # noqa: E402
    _ankle_drift_metrics,
    _cop_invalid_mask,
    _tcode_from_marker_min_y,
)
from opensim_batch_dynamics.mot_to_csv import _lowpass_butterworth_4th


class GroundLinkTorqueTargetsTest(unittest.TestCase):
    def test_resume_rejects_cop_shifted_copy_until_target_id_is_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            baseline = Path(tmp) / "torque_targets.csv"
            baseline.write_text("baseline", encoding="utf-8")
            fields = {
                "frame", "time", "segment", "valid", "cop_invalid",
                *[f"{name}_tau_target" for name in TARGET_COORDINATES],
                *[f"{name}_tau_hat" for name in TARGET_COORDINATES],
            }
            metadata = {
                "protocol": "original_id_setup",
                "cop_shift_m": [0.01, 0.02],
                "source_hashes": {},
            }
            common = {
                "expected_frames": 10,
                "protocol": "original_id_setup",
                "expected_shift": [0.01, 0.02],
                "nomuscles_targets_path": baseline,
            }

            self.assertFalse(_resume_output_is_current(metadata, fields, 10, **common))
            metadata.update({
                "target_id_regenerated": True,
                "tau_hat_source": "nomuscles torque_targets.csv, copied unchanged",
                "source_hashes": {
                    "nomuscles_torque_targets_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                },
            })
            self.assertTrue(_resume_output_is_current(metadata, fields, 10, **common))

    def test_revised_ankle_drift_criterion_records_trial_side_counts(self) -> None:
        baseline = []
        output = []
        for frame in range(100):
            old = {"ankle_angle_l_tau_target": "100", "ankle_angle_l_tau_hat": "100",
                   "ankle_angle_r_tau_target": "100", "ankle_angle_r_tau_hat": "100"}
            new = dict(old)
            if frame < 3:
                new["ankle_angle_r_tau_target"] = str(101 if frame == 0 else 100.000002)
            baseline.append(old)
            output.append(new)
        result = _ankle_drift_metrics(baseline, output)
        right = result["sides"]["r"]
        self.assertTrue(right["criteria_pass"])
        self.assertEqual(right["frames_with_any_abs_delta_gt_1e-6"], 3)
        self.assertEqual(right["series"]["target"]["frames_abs_delta_gt_1e-6"], 3)
        self.assertEqual(right["series"]["hat"]["frames_abs_delta_gt_1e-6"], 0)
        self.assertEqual(right["max_abs_delta_nm"], 1.0)
        self.assertEqual(right["series"]["target"]["baseline_abs_tau_p99_nm"], 100.0)

    def test_revised_ankle_drift_limit_is_strict_and_nonmutating(self) -> None:
        baseline = [{
            "ankle_angle_l_tau_target": "100", "ankle_angle_l_tau_hat": "100",
            "ankle_angle_r_tau_target": "100", "ankle_angle_r_tau_hat": "100",
        }]
        output = [{**baseline[0], "ankle_angle_r_tau_hat": "102"}]
        result = _ankle_drift_metrics(baseline, output)
        self.assertFalse(result["sides"]["r"]["criteria_pass"])
        self.assertEqual(result["sides"]["r"]["max_abs_delta_nm"], 2.0)
        self.assertEqual(baseline[0]["ankle_angle_r_tau_hat"], "100")

    def _save_force(self, path: Path, *, add_moment: bool = False) -> None:
        frames = 250
        grf = np.zeros((frames, 2, 3), dtype=float)
        cop = np.zeros_like(grf)
        grf[20:220, :, 2] = 0.1  # GroundLink stores kN.
        cop[20:220, :, 0] = 0.25
        payload = {"GRF": grf, "CoP": cop}
        if add_moment:
            payload["M"] = np.zeros_like(grf)
            payload["M"][20:220, :, 2] = 0.2
            payload["M_unit"] = "kNm"
        np.save(path, payload, allow_pickle=True)

    def test_measured_loads_apply_threshold_filter_and_zero_missing_free_moment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "force.npy"
            self._save_force(source)
            cop, grf, moment, moment_source, short_runs = _load_measured_wrenches(source)
            self.assertEqual(grf.shape, (250, 2, 3))
            self.assertEqual(cop.shape, grf.shape)
            self.assertEqual(moment.shape, (250, 2))
            self.assertEqual(moment_source, "zero_no_free_moment_field")
            self.assertEqual(short_runs, [])
            self.assertTrue(np.all(moment == 0.0))
            self.assertGreater(float(np.max(grf[:20, :, 2])), 0.0)
            unloaded_times = np.arange(10, dtype=float) / 250.0
            cop_unloaded, grf_unloaded, _ = _resample_wrenches(
                cop, grf, moment, unloaded_times
            )
            self.assertTrue(np.all(grf_unloaded == 0.0))
            self.assertTrue(np.all(cop_unloaded == 0.0))
            self.assertAlmostEqual(float(np.median(grf[50:190, 0, 2])), 100.0, places=3)
            self.assertAlmostEqual(float(np.median(cop[50:190, 0, 0])), 0.25, places=3)

    def test_resample_applies_trial_cop_shift_only_to_valid_contact_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "force.npy"
            self._save_force(source)
            cop, grf, moment, _, _ = _load_measured_wrenches(source)
            cop[80, 0] = 0.0  # Invalid CoP while force remains in contact.
            sample_times = np.arange(20, 220, dtype=float) / 250.0
            baseline_cop, baseline_grf, _ = _resample_wrenches(cop, grf, moment, sample_times)
            shifted_cop, shifted_grf, _ = _resample_wrenches(
                cop, grf, moment, sample_times, cop_shift_m=(0.10, 0.04)
            )

            np.testing.assert_array_equal(shifted_grf, baseline_grf)
            valid_cop = np.any(baseline_cop[:, :, :2] != 0.0, axis=2)
            np.testing.assert_allclose(
                shifted_cop[valid_cop, :2], baseline_cop[valid_cop, :2] - np.array([0.10, 0.04])
            )
            np.testing.assert_array_equal(shifted_cop[60, 0], np.zeros(3))
            np.testing.assert_array_equal(shifted_cop[:, :, 2], baseline_cop[:, :, 2])

    def test_cop_shift_csv_requires_unique_finite_trial_shifts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shifts.csv"
            path.write_text(
                "subject,trial,dx_m,dy_m\ns001,walk_1,0.12,-0.34\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _load_cop_shifts(path),
                {("s001", "walk_1"): (0.12, -0.34)},
            )
            path.write_text(
                "subject,trial,dx_m,dy_m\ns001,walk_1,0.12,-0.34\ns001,walk_1,0.01,0.02\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate CoP shift"):
                _load_cop_shifts(path)
            path.write_text(
                "subject,trial,dx_m,dy_m\ns001,walk_1,nan,0.02\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Non-finite CoP shift"):
                _load_cop_shifts(path)

    def test_b2_mot_translation_changes_only_valid_cop_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_grf_measured.mot"
            output = root / "output_grf_measured.mot"
            columns = [
                "time",
                "ground_force_calcn_l_vz", "ground_force_calcn_l_px", "ground_force_calcn_l_py",
                "ground_force_calcn_r_vz", "ground_force_calcn_r_px", "ground_force_calcn_r_py",
            ]
            source.write_text(
                "name test\nendheader\n"
                + "\t".join(columns) + "\n"
                + "0.0\t0\t0\t0\t0\t0\t0\n"
                + "0.1\t100\t0.5\t0.25\t0\t0\t0\n"
                + "0.2\t120\t0\t0\t100\t-0.5\t-0.1\n",
                encoding="utf-8",
            )
            output.write_bytes(source.read_bytes())

            self.assertEqual(_shift_mot(output, 0.1, 0.2), 2)
            self.assertEqual(_verify_mot(source, output, 0.1, 0.2), 2)
            source_rows = [line.split() for line in source.read_text().splitlines()[3:]]
            output_rows = [line.split() for line in output.read_text().splitlines()[3:]]
            self.assertEqual(source_rows[1][1], output_rows[1][1])
            self.assertEqual(output_rows[1][2:4], ["0.400000000000", "0.050000000000"])
            self.assertEqual(source_rows[2][1], output_rows[2][1])
            self.assertEqual(source_rows[2][2:4], ["0", "0"])
            self.assertEqual(output_rows[2][2:4], ["0", "0"])
            self.assertEqual(source_rows[2][4], output_rows[2][4])
            self.assertEqual(output_rows[2][5:], ["-0.600000000000", "-0.300000000000"])

    def test_tcode_mot_translation_adds_only_to_valid_contact_cop_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_grf_measured.mot"
            output = root / "output_grf_measured.mot"
            columns = [
                "time",
                "ground_force_calcn_l_vz", "ground_force_calcn_l_px", "ground_force_calcn_l_py",
                "ground_force_calcn_r_vz", "ground_force_calcn_r_px", "ground_force_calcn_r_py",
            ]
            source.write_text(
                "name test\nendheader\n"
                + "\t".join(columns) + "\n"
                + "0.0\t0\t0\t0\t0\t0\t0\n"
                + "0.1\t100\t0.5\t0.25\t0\t0\t0\n"
                + "0.2\t120\t0\t0\t100\t-0.5\t-0.1\n",
                encoding="utf-8",
            )
            output.write_bytes(source.read_bytes())

            self.assertEqual(_shift_mot_add_tcode(output, 0.0, 0.04), 2)
            self.assertEqual(_verify_mot_add_tcode(source, output, 0.0, 0.04), 2)
            source_rows = [line.split() for line in source.read_text().splitlines()[3:]]
            output_rows = [line.split() for line in output.read_text().splitlines()[3:]]
            self.assertEqual(source_rows[1][1:3], output_rows[1][1:3])
            self.assertEqual(output_rows[1][3], "0.290000000000")
            self.assertEqual(output_rows[1][4:], source_rows[1][4:])
            self.assertEqual(output_rows[2][1:4], source_rows[2][1:4])
            self.assertEqual(output_rows[2][4:6], source_rows[2][4:6])
            self.assertEqual(output_rows[2][6], "-0.060000000000")

    def test_tcode_reconstruction_matches_b1bis_reference_with_strict_threshold(self) -> None:
        reference = {"T_x": "0", "T_y": "-0.123", "T_z": "0"}
        transform, delta = _tcode_from_marker_min_y(0.133, reference)
        self.assertEqual((transform[0], transform[2]), (0.0, 0.0))
        self.assertAlmostEqual(transform[1], -0.123, places=12)
        self.assertLess(delta, 1e-6)
        with self.assertRaisesRegex(ValueError, "differs"):
            _tcode_from_marker_min_y(0.1330011, reference)

    def test_free_moment_unit_is_converted_and_resampled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "force.npy"
            self._save_force(source, add_moment=True)
            cop, grf, moment, source_label, _ = _load_measured_wrenches(source)
            self.assertEqual(source_label, "M:knm")
            self.assertAlmostEqual(float(np.mean(moment[50:190])), 200.0, delta=0.1)
            sample_times = np.arange(20, 220, dtype=float) / 250.0
            sampled_cop, sampled_grf, sampled_moment = _resample_wrenches(
                cop, grf, moment, sample_times
            )
            self.assertEqual(sampled_grf.shape, (200, 2, 3))
            self.assertAlmostEqual(float(np.mean(sampled_moment[80:180, 0])), 200.0, delta=0.2)
            np.testing.assert_allclose(sampled_cop[:, :, 2], 0.0, atol=0.0)

    def test_setup_clone_preserves_original_id_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = root / "original.xml"
            tool = ET.Element("InverseDynamicsTool")
            for name, value in (
                ("model_file", "/removed/model.osim"),
                ("coordinates_file", "/removed/ik.mot"),
                ("external_loads_file", "/removed/forces.xml"),
                ("results_directory", "/removed"),
                ("output_gen_force_file", "estimated.sto"),
                ("lowpass_cutoff_frequency_for_coordinates", "12"),
                ("time_range", "0 1"),
                ("forces_to_exclude", ""),
            ):
                ET.SubElement(tool, name).text = value
            doc = ET.Element("OpenSimDocument")
            doc.append(tool)
            ET.ElementTree(doc).write(setup)

            destination = root / "copy" / "Setup.xml"
            cutoff = _patch_original_setup(
                setup,
                destination,
                root / "model.osim",
                root / "ik.mot",
                root / "measured.xml",
                root / "target.sto",
            )
            copied = ET.parse(destination).getroot().find("InverseDynamicsTool")
            self.assertEqual(cutoff, 12.0)
            self.assertEqual(copied.findtext("lowpass_cutoff_frequency_for_coordinates"), "12")
            self.assertEqual(copied.findtext("time_range"), "0 1")
            self.assertEqual(copied.findtext("forces_to_exclude"), "")
            self.assertEqual(copied.findtext("model_file"), "../model.osim")
            self.assertEqual(copied.findtext("coordinates_file"), "../ik.mot")
            self.assertEqual(copied.findtext("external_loads_file"), "../measured.xml")
            self.assertEqual(copied.findtext("results_directory"), ".")
            self.assertEqual(copied.findtext("output_gen_force_file"), "target.sto")

            explicit = root / "copy" / "Setup_explicit.xml"
            cutoff = _patch_original_setup(
                setup,
                explicit,
                root / "model.osim",
                root / "ik.mot",
                root / "measured.xml",
                root / "target.sto",
                cutoff_override_hz=-1.0,
            )
            explicit_tool = ET.parse(explicit).getroot().find("InverseDynamicsTool")
            self.assertEqual(cutoff, 12.0)
            self.assertEqual(explicit_tool.findtext("lowpass_cutoff_frequency_for_coordinates"), "-1.0")

    def test_setup_clone_can_exclude_actuators_without_changing_other_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.xml"
            tool = ET.Element("InverseDynamicsTool")
            for name, value in (
                ("model_file", "model.osim"),
                ("coordinates_file", "ik.mot"),
                ("external_loads_file", "loads.xml"),
                ("results_directory", "."),
                ("output_gen_force_file", "id.sto"),
                ("lowpass_cutoff_frequency_for_coordinates", "12"),
                ("time_range", "0 1"),
                ("forces_to_exclude", ""),
            ):
                ET.SubElement(tool, name).text = value
            doc = ET.Element("OpenSimDocument")
            doc.append(tool)
            ET.ElementTree(doc).write(source)

            output = root / "copy.xml"
            _clone_setup_with_forces_to_exclude(source, output, "Actuators")
            original_tool = ET.parse(source).getroot().find("InverseDynamicsTool")
            copied_tool = ET.parse(output).getroot().find("InverseDynamicsTool")
            self.assertEqual(copied_tool.findtext("forces_to_exclude"), "Actuators")
            for name in (
                "model_file", "coordinates_file", "external_loads_file",
                "results_directory", "output_gen_force_file",
                "lowpass_cutoff_frequency_for_coordinates", "time_range",
            ):
                self.assertEqual(copied_tool.findtext(name), original_tool.findtext(name))

    def test_cop_mask_requires_force_strictly_above_threshold_and_zero_cop(self) -> None:
        fz = np.asarray([20.0, 20.001, 40.0, 40.0, 40.0])
        cop_x = np.asarray([0.0, 0.0, 0.0, 1e-10, 0.0])
        cop_y = np.asarray([0.0, 0.0, 1e-10, 0.0, 0.0])
        np.testing.assert_array_equal(
            _cop_invalid_mask(fz, cop_x, cop_y),
            [False, True, False, False, True],
        )

    def test_identical_load_id_comparison_covers_all_six_target_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "target.csv"
            comparison = root / "identity.csv"
            target_columns = [f"{name}_tau_target" for name in TARGET_COORDINATES]
            hat_columns = [f"{name}_tau_hat" for name in TARGET_COORDINATES]
            reference.write_text(
                ",".join(["frame", "time", "segment", *target_columns])
                + "\n0,0.0,0,"
                + ",".join("1.0" for _ in TARGET_COORDINATES)
                + "\n",
                encoding="utf-8",
            )
            comparison.write_text(
                ",".join(["frame", "time", "segment", *hat_columns])
                + "\n0,0.0,0,"
                + ",".join("1.0" for _ in TARGET_COORDINATES)
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _max_abs_id_csv_difference(
                    reference, comparison, "target", "hat", "s001", "trial", 0
                ),
                0.0,
            )
            comparison.write_text(
                ",".join(["frame", "time", "segment", *hat_columns])
                + "\n0,0.0,0,"
                + ",".join(["1.0", "1.0", "1.0", "1.0", "1.0", "1.1"])
                + "\n",
                encoding="utf-8",
            )
            self.assertAlmostEqual(
                _max_abs_id_csv_difference(
                    reference, comparison, "target", "hat", "s001", "trial", 0
                ),
                0.1,
            )

    def test_explicit_filter_matches_pipeline_filter_with_timestamp_jitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ik.mot"
            destination = root / "filtered.mot"
            frames = 300
            nominal_step = 1.0 / 250.0
            jitter = 0.0015 * np.sin(np.arange(frames - 1, dtype=float) * 0.17)
            times = np.r_[0.0, np.cumsum(nominal_step * (1.0 + jitter))]
            values = np.sin(2.0 * np.pi * 2.0 * times) + 0.5 * np.sin(2.0 * np.pi * 35.0 * times)
            source.write_text(
                "name ik\ndatacolumns 2\ndatarows 300\nendheader\n"
                "time hip_flexion_l\n"
                + "".join(f"{time:.12g} {value:.12g}\n" for time, value in zip(times, values)),
                encoding="utf-8",
            )
            filtered_path, filtered_times = _filter_ik_mot_explicit_order4(source, destination)
            self.assertEqual(filtered_path, destination)
            np.testing.assert_allclose(filtered_times, times, rtol=0.0, atol=5e-12)
            filtered = np.loadtxt(destination, skiprows=5)
            np.testing.assert_allclose(filtered[:, 0], times, rtol=0.0, atol=5e-12)
            expected = _lowpass_butterworth_4th(values, 1.0 / np.median(np.diff(times)), 12.0)
            np.testing.assert_allclose(filtered[:, 1], expected, rtol=0.0, atol=2e-11)

    def test_butterworth_filter_has_no_sample_lag(self) -> None:
        sample_rate = 250.0
        times = np.arange(0.0, 10.0, 1.0 / sample_rate)
        reference = np.sin(2.0 * np.pi * 2.0 * times)
        signal = reference + 0.5 * np.sin(2.0 * np.pi * 35.0 * times)
        filtered = _lowpass_butterworth_4th(signal, sample_rate, 12.0)
        interior = slice(200, -200)
        correlations = np.correlate(filtered[interior], reference[interior], mode="full")
        lag = int(np.argmax(correlations) - (len(correlations) - 1) // 2)
        self.assertEqual(lag, 0)

    def test_threshold_boundary_and_contact_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "force.npy"
            grf = np.zeros((250, 2, 3), dtype=float)
            cop = np.zeros_like(grf)
            grf[20:60, 0, 2] = 0.019999  # Below 20 N after kN -> N conversion.
            cop[20:60, 0, 0] = 0.7
            grf[90:130, 1, 2] = 0.020  # Exactly 20 N remains in contact.
            cop[90:130, 1, 0] = 0.3
            grf[150:190, 1, 2] = 0.1
            cop[150:190, 1, 0] = 0.5
            np.save(source, {"GRF": grf, "CoP": cop}, allow_pickle=True)
            cop_out, grf_out, _, _, _ = _load_measured_wrenches(source)
        self.assertTrue(np.all(cop_out[20:60, 0, 0] == 0.0))
        self.assertTrue(np.all(grf_out[20:60, 0, 2] == 0.0))
        self.assertGreater(float(np.median(cop_out[100:120, 1, 0])), 0.2)
        self.assertGreater(float(np.median(grf_out[100:120, 1, 2])), 19.0)
        self.assertTrue(np.all(cop_out[130:150, 1, 0] == 0.0))
        self.assertGreater(float(np.median(cop_out[160:180, 1, 0])), 0.4)

    def test_short_ik_segment_matches_pipeline_filter_passthrough(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, destination = root / "short.mot", root / "filtered.mot"
            times = np.arange(12, dtype=float) / 250.0
            values = np.linspace(-0.2, 0.4, len(times))
            source.write_text(
                "name short\ndatacolumns 2\ndatarows 12\nendheader\n"
                "time hip_flexion_l\n"
                + "".join(f"{time:.12g} {value:.12g}\n" for time, value in zip(times, values)),
                encoding="utf-8",
            )
            _filter_ik_mot_explicit_order4(source, destination)
            filtered = np.loadtxt(destination, skiprows=5)
        np.testing.assert_allclose(filtered[:, 1], values, rtol=0.0, atol=1e-12)

    def test_short_cop_contacts_are_preserved_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "force.npy"
            grf = np.zeros((250, 2, 3), dtype=float)
            cop = np.zeros_like(grf)
            grf[30:32, 0, 2] = 0.1
            grf[60:65, 1, 2] = 0.1
            cop[30:32, 0, 0] = 0.4
            cop[60:65, 1, 0] = 0.2
            np.save(source, {"GRF": grf, "CoP": cop}, allow_pickle=True)
            cop_out, _, _, _, short_runs = _load_measured_wrenches(source)
        self.assertEqual(short_runs, [{
            "foot": "l", "start_index": 30, "stop_index_exclusive": 32, "samples": 2,
        }])
        np.testing.assert_array_equal(cop_out[30:32, 0, 0], [0.4, 0.4])
        self.assertTrue(np.all(np.isfinite(cop_out[60:65, 1, 0])))

    def test_measured_external_loads_are_assigned_to_matching_calcn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            times = np.array([0.0, 0.004, 0.008])
            grf = np.zeros((3, 2, 3))
            cop = np.zeros_like(grf)
            moment = np.zeros((3, 2))
            _, xml_path = _write_measured_load_files(root, "trial", times, grf, cop, moment)
            forces = ET.parse(xml_path).getroot().findall(".//ExternalForce")
        self.assertEqual(len(forces), 2)
        for force, side in zip(forces, ("l", "r")):
            self.assertEqual(force.findtext("applied_to_body"), f"calcn_{side}")
            self.assertEqual(force.findtext("force_expressed_in_body"), "ground")
            self.assertEqual(force.findtext("point_expressed_in_body"), "ground")
            self.assertEqual(force.findtext("force_identifier"), f"ground_force_calcn_{side}_v")
            self.assertEqual(force.findtext("point_identifier"), f"ground_force_calcn_{side}_p")
            self.assertEqual(force.findtext("torque_identifier"), f"ground_force_calcn_{side}_m")

    def test_h1_mask_reads_times_segments_and_boolean_validity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.csv"
            path.write_text(
                "time,segment,valid\n0.0,0,1\n0.004,0,0\n0.5,1,1\n",
                encoding="utf-8",
            )
            times, segments, valid = _load_h1_mask(path)
        np.testing.assert_allclose(times, [0.0, 0.004, 0.5])
        np.testing.assert_array_equal(segments, [0, 0, 1])
        np.testing.assert_array_equal(valid, [True, False, True])

    def test_discover_setups_supports_single_and_multi_segment_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp)
            id_root = artifact / "ID_estimatedGRF"
            direct = id_root / "Setup_ID_estimatedGRF_trial.xml"
            direct.parent.mkdir(parents=True)
            direct.touch()
            self.assertEqual(_discover_segment_setup_paths(artifact, 1), [direct])
            segment_zero = id_root / "segments" / "segment_0" / "Setup_ID_estimatedGRF_trial_segment_0.xml"
            segment_one = id_root / "segments" / "segment_1" / "Setup_ID_estimatedGRF_trial_segment_1.xml"
            segment_zero.parent.mkdir(parents=True)
            segment_one.parent.mkdir(parents=True)
            segment_zero.touch()
            segment_one.touch()
            self.assertEqual(
                _discover_segment_setup_paths(artifact, 2), [segment_zero, segment_one]
            )


if __name__ == "__main__":
    unittest.main()

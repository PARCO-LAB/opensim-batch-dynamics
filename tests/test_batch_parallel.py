from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from batch_common import BatchTask, build_trial_name  # noqa: E402
from run_amass_batch_parallel import _run_single_task  # noqa: E402
from run_amass_to_bsm_csv import _publish_trial_artifacts  # noqa: E402


class BatchParallelTest(unittest.TestCase):
    def test_publish_trial_artifacts_skips_geometry_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "stage"
            target = root / "published"
            (source / "results" / "ID_estimatedGRF").mkdir(parents=True)
            (source / "results" / "ID_estimatedGRF" / "trial.sto").write_text("sto")
            (source / "Geometry").mkdir()
            (source / "Geometry" / "mesh.vtp").write_text("mesh")

            _publish_trial_artifacts(source, target)

            self.assertTrue((target / "results" / "ID_estimatedGRF" / "trial.sto").exists())
            self.assertFalse((target / "Geometry").exists())

    def test_excluded_trial_gets_explicit_ineligible_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            relative = Path("s002_20220520_balletsmalljump_1_stageii.npz")
            task = BatchTask(
                input_path=root / relative,
                relative_path=relative,
                output_csv_path=root / "out" / relative.with_suffix(".csv").name,
                log_path=root / "out" / "logs" / relative.with_suffix(".log").name,
                trial_name=build_trial_name(relative),
            )
            result = _run_single_task(argparse.Namespace(), root / "unused.py", root / "out", task)

            self.assertEqual(result.status, "excluded")
            sidecar = json.loads(
                task.output_csv_path.with_suffix(task.output_csv_path.suffix + ".quality.json").read_text()
            )
            self.assertEqual(sidecar["status"], "excluded")
            self.assertEqual(sidecar["reason"], "nonballistic_flight_kinematics")

    def test_no_cleanup_keeps_unique_open_sim_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake = root / "fake_pipeline.py"
            fake.write_text(
                "import argparse, json\n"
                "from pathlib import Path\n"
                "p = argparse.ArgumentParser()\n"
                "p.add_argument('--output-dir', required=True)\n"
                "p.add_argument('--trial', required=True)\n"
                "p.add_argument('--final-csv-path', required=True)\n"
                "p.add_argument('--engine-work-root')\n"
                "a, _ = p.parse_known_args()\n"
                "if a.engine_work_root:\n"
                "    work = Path(a.engine_work_root)\n"
                "    work.mkdir(parents=True, exist_ok=True)\n"
                "    (work / 'used').write_text('yes')\n"
                "(Path(a.output_dir) / a.trial / 'results').mkdir(parents=True)\n"
                "out = Path(a.final_csv_path)\n"
                "out.parent.mkdir(parents=True, exist_ok=True)\n"
                "out.write_text('value\\n1\\n')\n"
                "(out.with_suffix(out.suffix + '.quality.json')).write_text(json.dumps({\n"
                "    'status': 'ok',\n"
                "    'dynamics_pipeline_version': 'opensim_root_force_target_zup_v7',\n"
                "}))\n",
                encoding="utf-8",
            )
            input_path = root / "s002_20220520_hopping_1_stageii.npz"
            input_path.write_bytes(b"placeholder")
            relative = Path(input_path.name)
            task = BatchTask(
                input_path=input_path,
                relative_path=relative,
                output_csv_path=root / "out" / relative.with_suffix(".csv").name,
                log_path=root / "out" / "logs" / relative.with_suffix(".log").name,
                trial_name=build_trial_name(relative),
            )
            args = argparse.Namespace(
                skip_existing=False,
                rerun_walking=False,
                require_intermediate_artifacts=False,
                cleanup_intermediate=False,
                python_exe=sys.executable,
                smplx_model_dir="smpl",
                bsm_model="bsm.osim",
                device="cpu",
                id_filter_mode="auto",
                id_grf_mode="estimated",
                id_contact_bodies="calcn_l,toes_l",
                id_friction_coeff=0.8,
                id_contact_height_threshold_m=0.04,
                id_contact_speed_threshold_mps=0.5,
                walking_contact_height_threshold_m=None,
                walking_contact_speed_threshold_mps=None,
                addbio_root=None,
                sex=None,
                subject_mass_kg=None,
                subject_height_m=None,
                id_cutoff_hz=None,
                skip_inverse_dynamics=False,
                engine_work_root=str(root / "engine-work"),
            )
            result = _run_single_task(args, fake, root / "out", task)

            self.assertEqual(result.status, "ok")
            self.assertTrue(task.output_csv_path.exists())
            artifacts = list((root / "out").glob(f"{task.output_csv_path.stem}_*/results"))
            self.assertEqual(len(artifacts), 1)
            self.assertEqual((root / "engine-work" / "used").read_text(), "yes")


if __name__ == "__main__":
    unittest.main()

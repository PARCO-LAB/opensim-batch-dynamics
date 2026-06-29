from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


class AmassBatchRunnerCliTest(unittest.TestCase):
    def test_parallel_dry_run_reports_planned_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sample.npz").touch()
            output_dir = (root / "out").resolve()

            result = _run_script(
                "scripts/run_amass_batch_parallel.py",
                "--input-root",
                str(root),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--limit",
                "1",
            )

        expected_csv = output_dir / "sample.csv"
        self.assertIn("Discovered 1 .npz/.npy files under:", result.stdout)
        self.assertIn("Skip existing CSVs: enabled (0 already present)", result.stdout)
        self.assertIn(f"[DRY 1] sample.npz -> {expected_csv}", result.stdout)
        self.assertIn("trial=sample_8151325d", result.stdout)

    def test_slurm_dry_run_reports_manifest_and_planned_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "sample.npz").touch()
            output_dir = (root / "out").resolve()

            result = _run_script(
                "scripts/run_amass_batch_slurm.py",
                "submit",
                "--input-root",
                str(root),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--limit",
                "1",
            )

        expected_csv = output_dir / "sample.csv"
        expected_manifest = output_dir / "slurm" / "manifest.jsonl"
        expected_plan = output_dir / "slurm" / "submit_plan.json"
        expected_sbatch = output_dir / "slurm" / "run_batch.sbatch"
        self.assertIn("Discovered 1 .npz files under:", result.stdout)
        self.assertIn(f"Manifest: {expected_manifest}", result.stdout)
        self.assertIn(f"SBATCH script: {expected_sbatch}", result.stdout)
        self.assertIn(f"Plan summary: {expected_plan}", result.stdout)
        self.assertIn(f"[DRY 1] sample.npz -> {expected_csv}", result.stdout)
        self.assertIn("trial=sample_8151325d", result.stdout)

    def test_ntu_dry_run_reports_manifest_and_output_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            root = Path(tmpdir)
            (root / "sample.skeleton").touch()
            input_root = root.relative_to(REPO_ROOT)
            output_dir = input_root / "out"

            result = _run_script(
                "scripts/run_ntu_skeleton_batch_slurm.py",
                "submit",
                "--input-root",
                str(input_root),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--limit",
                "1",
            )

        expected_output_npz = (root / "out" / "sample.npz").resolve()
        self.assertIn("Found .skeleton: 1", result.stdout)
        self.assertIn(f"Manifest: {(root / 'out' / 'slurm' / 'manifest.jsonl').resolve()}", result.stdout)
        self.assertIn("SBATCH: ", result.stdout)
        self.assertIn(f"[DRY] sample.skeleton -> {expected_output_npz}", result.stdout)

    def test_humanml_dry_run_reports_manifest_and_output_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            root = Path(tmpdir)
            (root / "joints" / "sample.npy").parent.mkdir(parents=True, exist_ok=True)
            (root / "joints" / "sample.npy").touch()
            input_root = root.relative_to(REPO_ROOT)
            output_dir = input_root / "out"

            result = _run_script(
                "scripts/run_humanml3d_joints_batch_slurm.py",
                "submit",
                "--input-root",
                str(input_root),
                "--output-dir",
                str(output_dir),
                "--dry-run",
                "--limit",
                "1",
            )

        expected_scan_root = (root / "joints").resolve()
        expected_output_npz = (root / "out" / "sample.npz").resolve()
        self.assertIn(f"Scan root: {expected_scan_root}", result.stdout)
        self.assertIn("Found .npy: 1", result.stdout)
        self.assertIn(f"Manifest: {(root / 'out' / 'slurm' / 'manifest.jsonl').resolve()}", result.stdout)
        self.assertIn("SBATCH: ", result.stdout)
        self.assertIn(f"[DRY] sample.npy -> {expected_output_npz}", result.stdout)


if __name__ == "__main__":
    unittest.main()

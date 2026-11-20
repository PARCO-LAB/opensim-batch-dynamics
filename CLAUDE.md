# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Converts AMASS / SMPL-X motion (`.npz`, also Motion-X `.npy`) into OpenSim BSM dynamics datasets: one unified CSV per trial with DOF `q`/`_vel`/`_acc`/`_tau`, estimated GRFs per contact body, and 0/1 contact codes. `knee_angle_{r,l}_beta` are always dropped, so the final CSV has 49 effective DOFs. Separate subsystems: a causal real-time inverse-dynamics prototype (`RT/`), converters from other datasets (NTU, HumanML3D, CARE-PD) into AMASS-like `.npz`, and GroundLink validation / torque-target workflows (`scripts/*groundlink*`).

Deeper context (read when needed, trust code over these): `CODEX.md` (long technical handoff: offline pipeline, RT algorithm, known limitations), `DESCRIPTION.md` (math of each pipeline stage), `PONYTAIL.md` (ranked refactor/deletion backlog), `RT/RESULTS.md` and `RT/NEXT_STEPS.md` (RT tuning history), `docs/groundlink_torque_targets_nomuscles_README.md`. `TODO.md` and `MotionX_FIX.md` are in Italian. README links use stale macOS absolute paths; treat them as repo-relative.

## Environment

- Conda env `opensim-torque` from `environment.yml` (Python 3.11, `opensim=4.5.2`, `numpy<2`, torch, `smplx`, `nimblephysics`). `environment-rt.yml` adds RT deps (`cvxpy`, `open3d`, ...).
- No `python` on PATH outside the env: use `conda activate opensim-torque` or `~/miniconda3/envs/opensim-torque/bin/python`.
- No packaging (`pyproject`/`setup.py`). Library code lives in `src/opensim_batch_dynamics/`; scripts insert `src/` into `sys.path` themselves (`_add_src_to_path()`), and tests insert `scripts/` or `src/`.
- External: AddBiomechanics checkout passed via `--addbio-root` or `$ADDBIO_ENGINE_ROOT`. Assets not in git: `model/smpl/SMPLX_NEUTRAL.npz`, `smplx/`, `transfer_data/`. `data/` and `outputs/` are gitignored.
- `KMP_DUPLICATE_LIB_OK=TRUE` is set by scripts to avoid OpenMP clashes between torch, OpenSim and Nimble.

## Tests

Tests use `unittest` (pytest is not installed in the env).

```bash
# full suite (needs opensim; tests/test_inverse_dynamics_no_grf.py uses the real model)
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'

# single module / single test
PYTHONPATH=src python -m unittest tests.test_batch_common
PYTHONPATH=src python -m unittest tests.test_batch_common.BatchCommonTest.test_excluded_trial_reason_matches_dataset_key
```

No linter or formatter is configured.

## Offline pipeline architecture

Entry point: `scripts/run_amass_to_bsm_csv.py` (`--trial all` exports one CSV per trial in a multi-trial file). Flow, mapped to `src/opensim_batch_dynamics/` modules:

1. `amass_loader.py`: parse Stage-II or legacy `poses/trans` `.npz`, sibling `shape.npz` fallback, Motion-X loading (includes the AIST translation fix).
2. `smplx_forward.py` then `bsm_markers.py`: SMPL-X forward pass, virtual markers = fixed vertices from `assets/smpl2ab/bsm_markers_smplx.yaml`.
3. `trc_export.py`, `bsm_subject_json.py`, `addbio_subject_folder.py`, `addbio_runner.py`: write TRC + subject JSON, run local AddBiomechanics (scaling + kinematic fit) in a local scratch dir (`--engine-work-root`, default `/tmp/groundlink-engine-work`, kept off NAS/CIFS).
4. `addbio_csv_export.py` / `mot_to_csv.py`: fitted `.mot` to model-ordered DOF CSV with filtered velocities/accelerations.
5. `inverse_dynamics_no_grf.py`: core dynamics. Estimates contact wrenches/GRF from COM dynamics (`--id-grf-mode estimated`, contact candidates default `all` bodies), then runs OpenSim ID. Effective cutoff is `min(cutoff, 0.8*Nyquist)` and the same value must reach CSV conversion and OpenSim ID.
6. `final_csv_export.py`: merge into the unified CSV. GRF columns are zero-filled when no contact, so schemas stay constant.
7. `quality.py`: quality gate. Output is written to `.csv.partial`, gated, then atomically renamed only on `ok`; a `.quality.json` sidecar is kept.

## Batch / HPC runners

- `scripts/batch_common.py` is shared by `run_amass_batch_parallel.py` (local process pool) and `run_amass_batch_slurm.py` (manifest JSONL + SLURM array, chunked by `--slurm-max-array-size`). It holds trial naming, excluded trials, and skip-existing logic.
- `DYNAMICS_PIPELINE_VERSION` in `batch_common.py`: an existing CSV is reused by skip-existing only if its quality sidecar is `ok` AND records this version. Bump it whenever a change makes old outputs invalid.
- SLURM defaults target the UniVR HPC layout (`~/storage/emartini/AMASS/AMASS` input, `~/storage/emartini/AMASS_torque` output). `.sbatch` files in `scripts/` are hand-written job scripts for GroundLink and AMASS retry/feeder workflows.
- Dataset converters (`convert_ntu_skeleton_to_smplx_npz.py`, `convert_humanml3d_joints_to_smplx_npz.py`, `convert_carepd_smpl_pkl_to_smplx_npz.py`) fit SMPL-X and export Z-up AMASS-like `.npz`; their output feeds `run_amass_to_bsm_csv.py` unchanged. Z-up is the gravity convention for BSM/Nimble/OpenSim everywhere.

## Real-time subsystem

`RT/rt_library.py` (`qpid`): causal two-stage estimator. Stage 1 = robust IK from 12 observed keypoints + kinematic filter + contact cues; stage 2 = dynamic contact QP (cvxpy/OSQP). Root residual is kept intentionally as a diagnostic. `RT/real_time_test.py` benchmarks RT against an offline CSV (`--csv`, `--noise-std`, `--drop-joint-prob`). RT outputs 51 DOFs vs 49 offline.

## Legacy

`run_amass_to_opensim.py`, `run_amass_to_opencap_legacy.py`, `assets/opencap/`, and `model/LaiUhlrich2022_torque_only.osim` are the old OpenCap/LaiUhlrich pipeline, slated for deletion in `PONYTAIL.md`. Do not extend them; the target model is `model/bsm/bsm.osim`.

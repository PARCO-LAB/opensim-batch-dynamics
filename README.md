# AMASS -> BSM OpenSim Unified CSV Pipeline

This repository provides an end-to-end pipeline that takes an AMASS `.npz` file and produces a **single unified `.csv`** with:

- DOF positions
- DOF velocities
- DOF accelerations
- DOF torques
- Ground reaction forces (per foot + total)
- Binary contact codes (`0/1`) per foot

`knee_angle_r_beta` and `knee_angle_l_beta` are removed from the final output (they are always `NaN`), so the final CSV contains **49 effective DOFs**.

Main entrypoint:

- [run_amass_to_bsm_csv.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/scripts/run_amass_to_bsm_csv.py)
- `--trial all` runs every trial found inside a multi-trial `.npz` and exports one CSV per trial.

Additional NTU entrypoints:

- [convert_ntu_skeleton_to_smplx_npz.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/scripts/convert_ntu_skeleton_to_smplx_npz.py): converts NTU RGB+D `.skeleton` files to AMASS-like SMPL-X `.npz`.
- [run_ntu_skeleton_batch_slurm.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/scripts/run_ntu_skeleton_batch_slurm.py): SLURM helper for NTU `.skeleton -> .npz` conversion.
- [convert_humanml3d_joints_to_smplx_npz.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/scripts/convert_humanml3d_joints_to_smplx_npz.py): fits SMPL-X params from HumanML3D `joints/*.npy` 52-joint files.
- [run_humanml3d_joints_batch_slurm.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/scripts/run_humanml3d_joints_batch_slurm.py): SLURM helper for HumanML3D `joints/*.npy -> .npz` conversion.

## Required Inputs and Assets

- AMASS input file (example): `data/A3-_Swing_arms_stageii.npz`
- SMPL-X model file: `model/smpl/SMPLX_NEUTRAL.npz`
- BSM OpenSim model: `model/bsm/bsm.osim`
- BSM geometry folder: `model/bsm/Geometry/`
- Marker map: `assets/smpl2ab/bsm_markers_smplx.yaml`
- Local AddBiomechanics checkout (see setup below)

AMASS compatibility note:

- The loader supports both Stage-II style `.npz` files and legacy AMASS files with `poses/trans`.
- If present, sibling `shape.npz` is used automatically as fallback for `gender` and `betas`.

NTU compatibility note:

- NTU RGB+D `.skeleton` files contain 25 3D joints in camera coordinates.
- The NTU converter fits an AMASS-like SMPL-X parameter file with `trans`, `root_orient`, `pose_body`, zero hand/face pose, `betas`, `gender`, `mocap_frame_rate`, and NTU metadata.
- Raw NTU skeletons are treated as Y-up; by default the converter exports **Z-up** motion (`--target-frame z-up`) to match this repo's BSM/Nimble/OpenSim gravity convention.
- NTU left/right labels are swapped by default before fitting (`--swap-left-right`) because the sample files are mirrored relative to the SMPL-X left/right convention.
- Subject size is handled by fitting SMPL-X `betas` from metric 3D limb lengths.

HumanML3D compatibility note:

- `new_joint_vecs/*.npy` are feature vectors and are not direct SMPL parameters.
- `new_joints/*.npy` are normalized 22-joint positions.
- `joints/*.npy` are the preferred source for this repo because they contain 52 SMPL/SMPL-H-style joints, including hands.
- HumanML3D motion is 20 fps by default.

## Ubuntu Setup

1. Install system dependencies:

```bash
sudo apt update
sudo apt install -y git wget
```

2. Install Conda (Miniconda), if needed:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
```

3. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate opensim-torque
```

4. Clone AddBiomechanics:

```bash
git clone https://github.com/keenon/AddBiomechanics.git "$HOME/AddBiomechanics"
```

## macOS Setup (Apple Silicon + Intel)

1. Install Conda (Miniconda or Miniforge).
2. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate opensim-torque
```

3. Clone AddBiomechanics:

```bash
git clone https://github.com/keenon/AddBiomechanics.git "$HOME/AddBiomechanics"
```

Notes for macOS:

- The pipeline script sets `KMP_DUPLICATE_LIB_OK=TRUE` automatically to avoid OpenMP runtime conflicts.
- If `python` is not available in your shell, use `python3` (inside the activated conda env, `python` is usually available).

## One Command End-to-End

Run the full pipeline with one Python command:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/A3-_Swing_arms_stageii.npz \
  --trial A3_swing_full \
  --output-dir outputs/bsm \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

If your shell does not resolve `python`, run the same command with `python3`.

To export one CSV per trial from a multi-trial `.npz`, use:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/your_multitrial_file.npz \
  --trial all \
  --output-dir outputs/bsm \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

Final output:

- `outputs/bsm/A3_swing_full.csv`

With `--cleanup-intermediate`, all temporary files under `outputs/bsm/<trial>/` are removed after a successful run, leaving only the final CSV.

## NTU `.skeleton` to SMPL-X `.npz`

NTU RGB+D 60/120 skeleton files can be converted to AMASS-like SMPL-X `.npz` files, then passed to `run_amass_to_bsm_csv.py`.

Single-file conversion:

```bash
python scripts/convert_ntu_skeleton_to_smplx_npz.py \
  --input-file data/ntu/S001C001P001R001A001.skeleton \
  --output-dir data/ntu_smplx_npz \
  --smplx-model-dir model/smpl \
  --target-frame z-up
```

Folder conversion:

```bash
python scripts/convert_ntu_skeleton_to_smplx_npz.py \
  --input-dir data/ntu \
  --output-dir data/ntu_smplx_npz \
  --smplx-model-dir model/smpl \
  --recursive \
  --target-frame z-up
```

Output example:

- `data/ntu_smplx_npz/S001C001P001R001A001.npz`
- `data/ntu_smplx_npz/conversion_summary.json`
- `data/ntu_smplx_npz/_shape_cache/P001.npy`

Important options:

- `--target-frame z-up` is the default and should be used before BSM/OpenSim dynamics.
- `--target-frame y-up` preserves raw NTU vertical and is mainly for debugging.
- `--actor-mode primary` keeps the main tracked body per file.
- `--actor-mode all` exports additional tracked bodies as `_bodyXX`.
- `--performer P001` filters conversion to one NTU performer.
- `--swap-left-right` is enabled by default; use `--no-swap-left-right` only when visual inspection shows the source skeleton already matches SMPL-X left/right.
- `--fit-shapes-only` fits/writes performer beta cache, then stops before pose fitting.
- `--force` recomputes existing outputs and shape cache entries.

The generated `.npz` can be fed directly into the existing BSM pipeline:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/ntu_smplx_npz/S001C001P001R001A001.npz \
  --trial S001C001P001R001A001 \
  --output-dir outputs/ntu_bsm \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

## NTU SLURM Batch Runner

Preferred HPC mode is one SLURM array task per NTU performer:

```bash
python scripts/run_ntu_skeleton_batch_slurm.py submit-subjects \
  --input-root data/ntu \
  --output-dir data/ntu_smplx_npz \
  --smplx-model-dir model/smpl \
  --python-exe python \
  --slurm-setup-cmd 'source "$HOME/miniconda3/etc/profile.d/conda.sh"' \
  --slurm-setup-cmd 'conda activate opensim-torque' \
  --slurm-array-parallelism 32 \
  --submit
```

What `submit-subjects` does:

- groups input files by NTU performer key (`P001`, `P002`, ...)
- launches one SLURM array task per performer
- inside each task, fits one common SMPL-X shape/beta vector for that performer
- converts all that performer's `.skeleton` files to `.npz`
- writes logs under `data/ntu_smplx_npz/logs/`
- writes SLURM manifests and result JSON files under `data/ntu_smplx_npz/slurm/`

This is usually cleaner than one job per file because each performer gets a shared shape estimate before pose fitting.

Dry run:

```bash
python scripts/run_ntu_skeleton_batch_slurm.py submit-subjects \
  --input-root data/ntu \
  --output-dir data/ntu_smplx_npz \
  --smplx-model-dir model/smpl \
  --limit 1 \
  --dry-run
```

Constrain the job to a specific node:

```bash
python scripts/run_ntu_skeleton_batch_slurm.py submit-subjects \
  --input-root data/ntu \
  --output-dir data/ntu_smplx_npz \
  --smplx-model-dir model/smpl \
  --slurm-node node001 \
  --slurm-array-parallelism 32 \
  --slurm-setup-cmd 'source "$HOME/miniconda3/etc/profile.d/conda.sh"' \
  --slurm-setup-cmd 'conda activate opensim-torque' \
  --submit
```

`--slurm-node node001` is an alias for `--slurm-nodelist node001` and writes `#SBATCH -w node001`.

Alternative two-step mode:

```bash
python scripts/run_ntu_skeleton_batch_slurm.py prefit-shapes \
  --input-root data/ntu \
  --output-dir data/ntu_smplx_npz \
  --smplx-model-dir model/smpl \
  --python-exe python

python scripts/run_ntu_skeleton_batch_slurm.py submit \
  --input-root data/ntu \
  --output-dir data/ntu_smplx_npz \
  --smplx-model-dir model/smpl \
  --python-exe python \
  --slurm-setup-cmd 'source "$HOME/miniconda3/etc/profile.d/conda.sh"' \
  --slurm-setup-cmd 'conda activate opensim-torque' \
  --slurm-array-parallelism 32 \
  --submit
```

Use this only if you explicitly want global prefit first and then one array task per file.

## HumanML3D 52-Joint to SMPL-X `.npz`

For HumanML3D, use the `joints/` folder, not `new_joint_vecs/`. The `joints/*.npy` files have shape `(T, 52, 3)` and provide the richest direct joint target.

```bash
python scripts/convert_humanml3d_joints_to_smplx_npz.py \
  --input data/HumanML3D/joints/000000.npy \
  --output-dir data/humanml3d_smplx_npz \
  --smplx-model-dir model/smpl \
  --target-frame z-up
```

Output example:

- `data/humanml3d_smplx_npz/000000.npz`

On HPC/SLURM, submit the full `joints/` folder with:

```bash
python scripts/run_humanml3d_joints_batch_slurm.py submit \
  --input-root data/HumanML3D \
  --output-dir data/humanml3d_smplx_npz \
  --smplx-model-dir model/smpl \
  --python-exe python \
  --slurm-nodelist 'node[001-002],blade[010-012]' \
  --slurm-setup-cmd 'source "$HOME/miniconda3/etc/profile.d/conda.sh"' \
  --slurm-setup-cmd 'conda activate opensim-torque' \
  --slurm-max-array-size 1000 \
  --slurm-array-parallelism 32 \
  --submit
```

The script automatically uses `--input-root/joints` when that folder exists. For large datasets, for example 17k HumanML3D files, it follows the same scheduling pattern as `run_amass_batch_slurm.py`: it splits the task list into multiple SLURM array chunks with `--slurm-max-array-size` (default `1000`) and submits them one at a time with retry/backoff. Use `--slurm-nodelist` or the alias `--slurm-node` to constrain execution to a node or node range.

HumanML3D fitting defaults are conservative for visual stability: `--input-smooth-passes 1`, `--pose-iters 2600`, `--no-fit-hands`, `--root-up-prior-weight 10`, `--torso-frame-prior-weight 2`, `--pose-prior-weight 0.05`, `--temporal-smooth-weight 0.03`, `--temporal-joint-smooth-weight 500`, `--temporal-joint-jerk-weight 50`, and `--foot-flat-prior-weight 25`. The hand fit is disabled by default because the HumanML3D 52-joint hands come from SMPL/SMPL-H-style joints and can create unnatural SMPL-X finger/body twists. The torso-frame prior matches hip, spine, collar, and shoulder directions from the source joints, the joint smooth/jerk priors suppress short flicker bursts, and the foot-flat prior keeps SMPL-X ankle, heel, and toe joints coplanar during detected contact frames.

It writes:

- `data/humanml3d_smplx_npz/slurm/manifest.jsonl`
- `data/humanml3d_smplx_npz/slurm/run_humanml3d_joints_to_npz.sbatch`
- additional chunk scripts such as `run_humanml3d_joints_to_npz_0001.sbatch` when needed
- `data/humanml3d_smplx_npz/slurm/results/task_*.json`
- per-file logs under `data/humanml3d_smplx_npz/logs/`

The generated `.npz` contains `trans`, `root_orient`, `pose_body`, `pose_hand`, `pose_jaw`, `pose_eye`, `betas`, `gender`, and `mocap_frame_rate`, so it can be passed to `run_amass_to_bsm_csv.py`:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/humanml3d_smplx_npz/000000.npz \
  --trial humanml3d_000000 \
  --output-dir outputs/humanml3d_bsm \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

## Batch Parallel Runner

To process an entire AMASS folder in parallel, use:

```bash
python scripts/run_amass_batch_parallel.py \
  --input-root /path/to/AMASS \
  --output-dir outputs/bsm_batch \
  --workers 8 \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

What it does:

- recursively scans `--input-root` for `.npz` files
- runs the full pipeline in parallel (`--workers N`)
- writes one final CSV per input `.npz` (preserving folder structure in `--output-dir`)
- writes per-file logs under `outputs/bsm_batch/logs/`
- writes a global summary JSON at `outputs/bsm_batch/batch_summary.json`

Useful options:

- `--dry-run` to preview discovered files and planned outputs
- `--limit N` to run only the first `N` discovered files
- `--skip-existing-csv` (default) skips files when the destination CSV already exists and is non-empty
- `--no-skip-existing-csv` to force re-run even when output CSV already exists
- `--no-cleanup-intermediate` to keep intermediate artifacts

Note on SMB paths:

- If your dataset path is like `smb://parconas.di.univr.it/MAEVE/dataset/AMASS`, mount it first (for example to `/Volumes/AMASS`) and pass the mounted local path to `--input-root`.

## SLURM Parallel Runner

For HPC clusters, use the SLURM helper script:

```bash
python scripts/run_amass_batch_slurm.py submit \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate \
  --slurm-job-name amass_bsm \
  --slurm-partition cpu \
  --slurm-time 08:00:00 \
  --slurm-cpus-per-task 4 \
  --slurm-mem 16G \
  --slurm-array-parallelism 32 \
  --slurm-setup-cmd 'source "$HOME/miniconda3/etc/profile.d/conda.sh"' \
  --slurm-setup-cmd 'conda activate opensim-torque' \
  --submit
```

On the UniVR HPC layout described in this project, the script now defaults to:

- input root: `~/storage/emartini/AMASS/AMASS`
- output root: `~/storage/emartini/AMASS_torque`

Assuming `~/storage` is a symlink to `/storage-large/shared-folders/gr_bombieri/emartini`, the shortest working submit command is:

```bash
python scripts/run_amass_batch_slurm.py submit \
  --addbio-root "$HOME/AddBiomechanics" \
  --slurm-job-name amass_bsm \
  --slurm-nodelist 'node[001-002],blade[010-012]' \
  --slurm-time 08:00:00 \
  --slurm-cpus-per-task 4 \
  --slurm-mem 16G \
  --slurm-array-parallelism 32 \
  --slurm-setup-cmd 'source "$HOME/miniconda3/etc/profile.d/conda.sh"' \
  --slurm-setup-cmd 'conda activate opensim-torque' \
  --submit
```

If your cluster requires an explicit partition or account, add `--slurm-partition ...` and/or `--slurm-account ...`.
If admins ask you to constrain jobs to a subset of hosts, add `--slurm-nodelist 'node[001-002],blade[010-012]'`.
If the preferred queue is overloaded, consider switching some batches to `--slurm-partition low`.

What this script does:

- discovers `.npz` files recursively
- skips existing destination CSVs by default (`--skip-existing-csv`)
- writes a task manifest at `outputs/bsm_batch/slurm/manifest.jsonl`
- writes an SBATCH script at `outputs/bsm_batch/slurm/run_batch.sbatch`
- launches a SLURM array job (if `--submit` is passed)
- stores per-task worker status JSON files under `outputs/bsm_batch/slurm/results/`

Useful SLURM workflow:

- use `--dry-run` first to generate and inspect manifest + SBATCH script without submitting
- omit `--submit` if you want to manually run the printed `sbatch ...` command
- use `--no-skip-existing-csv` when you intentionally want to recompute all outputs
- tune `--slurm-array-parallelism` to cap concurrent jobs (for example `%32`)

## Output Format

The unified CSV includes:

- `frame`, `time`
- Subject metadata: `subject_mass_kg`, `subject_height_m`
- Body scaling metadata: `<body>_scale_x`, `<body>_scale_y`, `<body>_scale_z` (one triplet for each model body)
- For each DOF: `<dof>`, `<dof>_vel`, `<dof>_acc`, `<dof>_tau`
- GRF columns (for every detected contact body and total)
- Contact code columns (for every detected contact body, for example `hand_l_contact`, `tibia_r_contact`, `calcn_l_contact`)

GRF values are explicitly encoded with zeros when no contact is detected, so CSV schemas stay consistent across motions (including jumps/flight phases).

## Pipeline Summary

For native AMASS input:

1. Load AMASS (`SMPL-X`) from `.npz`.
2. Run SMPL-X forward pass.
3. Extract BSM virtual markers from SMPL-X vertices.
4. Export `markers.trc`.
5. Run AddBiomechanics scaling + kinematic fit.
6. Export fitted `.osim` and `.mot`.
7. Compute DOF kinematics (`q`, `qdot`, `qddot`).
8. Estimate contact wrenches/GRF from motion.
9. Run inverse dynamics in OpenSim.
10. Merge kinematics + torques + GRF/contact into one final CSV.

For NTU input, prepend:

1. Read NTU RGB+D `.skeleton` 25-joint tracks.
2. Convert raw NTU Y-up camera coordinates to Z-up world coordinates.
3. Fit SMPL-X `betas` from metric segment lengths.
4. Fit SMPL-X root/body pose over time.
5. Export AMASS-like `.npz`, then run the same AMASS/BSM pipeline above.

For HumanML3D `joints/*.npy` input, prepend:

1. Read HumanML3D 52-joint SMPL/SMPL-H-style positions.
2. Convert HumanML3D Y-up coordinates to Z-up world coordinates.
3. Fit SMPL-X `betas` from 52-joint segment lengths.
4. Fit SMPL-X root and body pose over time, with hands neutral by default.
5. Export AMASS-like `.npz`, then run the same AMASS/BSM pipeline above.

## Key CLI Options

- `--addbio-root /path/to/AddBiomechanics`
- `--id-grf-mode {estimated,none}` (default: `estimated`)
- `--id-contact-bodies all` (default; uses all model body nodes as contact candidates)
- `--id-friction-coeff 0.8` (default)
- `--id-filter-mode {auto,walking,dynamic,none}` (default: `auto`)
- `--id-cutoff-hz <float>` (optional custom cutoff)
- `--final-csv-path /custom/output.csv` (optional)
- `--cleanup-intermediate` (keep only final CSV)

## CSV Explorer (PDF Report)

You can generate a full multi-page PDF report from any unified output CSV with:

```bash
python scripts/csv_explorer.py \
  --input-csv outputs/bsm/A3-_Swing_arms_stageii.csv \
  --output-pdf outputs/bsm/A3-_Swing_arms_stageii_report.pdf
```

The report includes:

- global summary (frames, duration, sample rate, detected DOFs and GRFs)
- motion overview statistics
- one page per DOF with position/velocity/acceleration/(torque if present)
- GRF pages for each contact body and total GRF

## Legacy Scripts

The previous pipelines are still available:

- [run_amass_to_opensim.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/scripts/run_amass_to_opensim.py)
- [run_amass_to_opencap_legacy.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/scripts/run_amass_to_opencap_legacy.py)

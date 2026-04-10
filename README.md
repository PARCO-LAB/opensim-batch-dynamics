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

## Required Inputs and Assets

- AMASS input file (example): `data/A3-_Swing_arms_stageii.npz`
- SMPL-X model file: `model/smpl/SMPLX_NEUTRAL.npz`
- BSM OpenSim model: `model/bsm/bsm.osim`
- BSM geometry folder: `model/bsm/Geometry/`
- Marker map: `assets/smpl2ab/bsm_markers_smplx.yaml`
- Local AddBiomechanics checkout (see setup below)

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

## Output Format

The unified CSV includes:

- `frame`, `time`
- For each DOF: `<dof>`, `<dof>_vel`, `<dof>_acc`, `<dof>_tau`
- GRF columns (per contact body and total)
- Contact code columns (for example `calcn_l_contact`, `calcn_r_contact`)

GRF values are explicitly encoded with zeros when no contact is detected, so CSV schemas stay consistent across motions (including jumps/flight phases).

## Pipeline Summary

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

## Key CLI Options

- `--addbio-root /path/to/AddBiomechanics`
- `--id-grf-mode {estimated,none}` (default: `estimated`)
- `--id-contact-bodies calcn_l,calcn_r` (default)
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

# AMASS -> OpenSim Torque-Only Pipeline

This repository contains only the files needed to convert an AMASS `.npz` sequence into:

- `.mot` (OpenSim inverse kinematics motion)
- `.osim` (scaled torque-only model, no muscles)
- `.csv` (frame-by-frame DOF table for the torque-only model, including velocity and acceleration columns)

The default model is `model/LaiUhlrich2022_torque_only.osim`.

## Minimal Repository Structure

- `scripts/run_amass_to_opensim.py`: single CLI entrypoint
- `src/opensim_batch_dynamics/`: pipeline implementation
- `assets/opencap/`: XML configs and marker mapping used for scaling and IK
- `model/LaiUhlrich2022_torque_only.osim`: torque-actuated model (no muscles)
- `model/Geometry/`: OpenSim mesh files required by the model
- `model/smpl/SMPLX_NEUTRAL.npz`: SMPL-X model file

## Setup with Conda

### Ubuntu

1. Install Miniconda (if needed):

```bash
sudo apt update
sudo apt install -y wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
```

2. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate opensim-torque
```

### macOS (Apple Silicon and Intel)

1. Install Miniconda for your architecture:

- Apple Silicon (M1/M2/M3):

```bash
curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.zsh hook)"
```

- Intel:

```bash
curl -L -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash miniconda.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.zsh hook)"
```

2. Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate opensim-torque
```

### Quick Environment Check

```bash
python -c "import opensim, torch, smplx; print('Environment OK')"
```

## Run the Pipeline

```bash
python scripts/run_amass_to_opensim.py \
  --input data/A3-_Swing_arms_stageii.npz \
  --trial demo_torque \
  --output-dir outputs \
  --mass-kg 75 \
  --height-m 1.75 \
  --sex neutral
```

## Generated Outputs

- `outputs/MarkerData/<trial>/<trial>.trc`
- `outputs/OpenSim/Model/<trial>/LaiUhlrich2022_torque_only_scaled.osim`
- `outputs/OpenSim/IK/<trial>/<trial>.mot`
- `outputs/OpenSim/IK/<trial>/<trial>_dofs.csv`

The script prints a final JSON payload with the absolute paths of all generated files.

The CSV contains:

- Position columns: `<dof>`
- Velocity columns: `<dof>_vel`
- Acceleration columns: `<dof>_acc`

Velocity and acceleration are computed from filtered DOF trajectories.
Filtering uses a 4th-order zero-phase Butterworth low-pass filter.

## Useful CLI Options

- `--skip-scale`: skip scaling and use `--model-path` directly
- `--skip-ik`: do not run IK (so no `.mot` and no `.csv`)
- `--trc-only`: export only TRC files
- `--csv-path /path/file.csv`: custom CSV output path
- `--csv-model-path /path/model.osim`: model used to order CSV DOF columns
- `--missing-fill 0`: fill value for DOFs missing in `.mot` (default: `nan`)
- `--filter-mode auto|walking|dynamic|none`: filtering strategy
- `--filter-cutoff-hz 20`: explicit cutoff override in Hz
- `--no-velocity-columns`: disable `<dof>_vel` columns
- `--no-acceleration-columns`: disable `<dof>_acc` columns

## Practical Notes

- This repo includes only `SMPLX_NEUTRAL.npz`. For male/female AMASS files, you can still run with `--sex neutral`.
- If OpenSim reports missing mesh files, make sure `model/Geometry` is present next to the `.osim` model.

## Replay in Nimble (optional)

You can replay the generated DOF CSV on the OpenSim model with:

```bash
python scripts/run_nimble.py \
  --osim model/LaiUhlrich2022_torque_only.osim \
  --csv outputs/OpenSim/IK/demo_torque/demo_torque_dofs.csv \
  --max-frames 200
```

By default geometry loading is disabled (faster and quieter).  
If you want mesh loading too, add `--load-geometry --geometry-dir model/Geometry`.

Optional realtime playback:

```bash
python scripts/run_nimble.py \
  --osim model/LaiUhlrich2022_torque_only.osim \
  --csv outputs/OpenSim/IK/demo_torque/demo_torque_dofs.csv \
  --realtime --speed 1.0
```
- `auto` filter mode uses 12 Hz for gait-like trial names and 30 Hz otherwise.

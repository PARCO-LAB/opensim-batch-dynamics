#!/usr/bin/env bash
# Read-only investigation of the blocking zero-Fz target check. node008 excluded.
#SBATCH --job-name=gl-tcode-swingdiag
#SBATCH --partition=medium
#SBATCH --exclude=node008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/storage-large/homedirs/mrtnrc33/.codex_phase_c/swing_diag_%j.out

set -euo pipefail
python="/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python"
repo="/home/accounts/personale/mrtnrc33/opensim-batch-dynamics"
root="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque"
cd "$repo"
exec "$python" scripts/diagnose_groundlink_tcode_swing.py \
  --source-dir "$root/torque_targets_12hz_nomuscles" \
  --output-dir "$root/torque_targets_12hz_nomuscles_copfix_tcode" \
  --index "$root/h1_labels/index.csv"

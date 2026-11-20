#!/usr/bin/env bash
# Read-only per-trial minimum marker-axis audit. node008 is explicitly excluded.
#SBATCH --job-name=gl-axis-trc
#SBATCH --partition=medium
#SBATCH --exclude=node008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/storage-large/homedirs/mrtnrc33/.codex_phase_c/axis_trc_%j.out

set -euo pipefail

python="/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python"
repo="/home/accounts/personale/mrtnrc33/opensim-batch-dynamics"
root="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque"

cd "$repo"
exec "$python" scripts/audit_groundlink_vertical_trc_minima.py \
  --source-dir "$root/torque_targets_12hz_nomuscles" \
  --index "$root/h1_labels/index.csv" \
  --output-dir "$root/torque_targets_12hz_nomuscles_copfix_tcode"

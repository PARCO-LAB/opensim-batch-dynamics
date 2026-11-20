#!/usr/bin/env bash
# Cohort validation, gated side_invalid mask, and Phase C report generation.
# Run only after all six T_code target arrays complete. node008 is excluded.
#SBATCH --job-name=gl-tcode-validate
#SBATCH --partition=medium
#SBATCH --exclude=node008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/storage-large/homedirs/mrtnrc33/.codex_phase_c/tcode_validation_%j.out

set -euo pipefail

python="/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python"
repo="/home/accounts/personale/mrtnrc33/opensim-batch-dynamics"
root="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque"
source="$root/torque_targets_12hz_nomuscles"
output="$root/torque_targets_12hz_nomuscles_copfix_tcode"
index="$root/h1_labels/index.csv"
tcode_audit="/storage-large/homedirs/mrtnrc33/.codex_phase_c/tcode_recomputed_by_trial.csv"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$repo"
exec "$python" scripts/validate_groundlink_tcode_phase_c.py \
  --source-dir "$source" \
  --output-dir "$output" \
  --index "$index" \
  --tcode-audit "$tcode_audit"

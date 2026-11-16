#!/usr/bin/env bash
#SBATCH --job-name=gl-cop-shift
#SBATCH --partition=low
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/torque_targets_12hz_nomuscles_copfix/logs/b2_%A_%a.out

set -euo pipefail

subjects=(s001 s002 s003 s004 s005 s006)
subject="${subjects[${SLURM_ARRAY_TASK_ID}]}"
python="/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python"
repo="/home/accounts/personale/mrtnrc33/opensim-batch-dynamics"
source_root="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/torque_targets_12hz_nomuscles"
output_root="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/torque_targets_12hz_nomuscles_copfix"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$repo"
exec "$python" scripts/apply_groundlink_cop_shifts.py \
  --source-dir "$source_root" \
  --output-dir "$output_root" \
  --subject "$subject"

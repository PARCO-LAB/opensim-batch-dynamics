#!/usr/bin/env bash
#SBATCH --job-name=gl-kin-cache
#SBATCH --partition=low
#SBATCH --array=0-5%2
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study/logs/phase_b_cache/%A_%a.out

set -euo pipefail

study_root=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study
script_root=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study/scripts
python_bin=/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" "$script_root/cache_kinematics.py" \
  --torque-root /storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque \
  --study-root "$study_root" \
  --subject-index "$SLURM_ARRAY_TASK_ID"

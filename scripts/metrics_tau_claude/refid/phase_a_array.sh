#!/usr/bin/env bash
#SBATCH --job-name=gl-hybrid-id
#SBATCH --partition=low
#SBATCH --array=0-65%4
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study/logs/phase_a/%A_%a.out

set -euo pipefail

study_root=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study
script_root=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study/scripts/refid
python_bin=/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" "$script_root/run_id_group.py" \
  --manifest "$study_root/phase_a/id_manifest.csv" \
  --group-index "$SLURM_ARRAY_TASK_ID"

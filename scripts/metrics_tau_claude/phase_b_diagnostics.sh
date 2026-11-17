#!/usr/bin/env bash
#SBATCH --job-name=gl-diagnostics
#SBATCH --partition=low
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study/logs/phase_b_diagnostics/%j.out

set -euo pipefail

study_root=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study
source_root=/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque
script_root="$study_root/scripts"
python_bin=/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" "$script_root/diagnose_modes.py" \
  --torque-root "$source_root" \
  --study-root "$study_root" \
  --output-dir "$study_root/phase_b/diagnostics"

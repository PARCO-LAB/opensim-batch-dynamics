#!/usr/bin/env bash
#SBATCH --job-name=gl-hybrid-eval
#SBATCH --partition=low
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study/logs/phase_a_eval/%j.out

set -euo pipefail

study_root=/storage-large/homedirs/mrtnrc33/groundlink_grf_tuning/2026-11-17_study
source_root=/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque
script_root="$study_root/scripts/refid"
python_bin=/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$python_bin" "$script_root/evaluate_hybrid_ids.py" \
  --manifest "$study_root/phase_a/id_manifest.csv" \
  --torque-root "$source_root" \
  --output-dir "$study_root/phase_a/results"

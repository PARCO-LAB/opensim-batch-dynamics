#!/bin/bash
#SBATCH --job-name=amass_oom_retry
#SBATCH --partition=gpuRTX
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-15%16
#SBATCH --output=/storage-exp-large/shared-folders/gr_bombieri/emartini/AMASS_torque_nomuscles/slurm/logs/oom_retry_%A_%a.out
#SBATCH --error=/storage-exp-large/shared-folders/gr_bombieri/emartini/AMASS_torque_nomuscles/slurm/logs/oom_retry_%A_%a.err

set -euo pipefail
cd /home/accounts/personale/mrtnrc33/opensim-batch-dynamics
unset PYTHONPATH
export PYTHONNOUSERSITE=1

INDEX_FILE=/storage-exp-large/shared-folders/gr_bombieri/emartini/AMASS_torque_nomuscles/slurm/oom_retry_indices.txt
MANIFEST=/storage-exp-large/shared-folders/gr_bombieri/emartini/AMASS_torque_nomuscles/slurm/manifest.jsonl
mapfile -t task_indices < "$INDEX_FILE"
TASK_INDEX="${task_indices[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "$TASK_INDEX" ]]; then
  echo "No task index for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python \
  scripts/run_amass_batch_slurm.py worker \
  --manifest "$MANIFEST" \
  --task-index "$TASK_INDEX" \
  --no-skip-existing-csv

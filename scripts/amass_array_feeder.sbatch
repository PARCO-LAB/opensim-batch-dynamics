#!/bin/bash
#SBATCH --job-name=amass_feeder
#SBATCH --partition=gpuRTX
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/storage-exp-large/shared-folders/gr_bombieri/emartini/AMASS_torque_nomuscles/slurm/logs/amass_feeder_%j.out
#SBATCH --error=/storage-exp-large/shared-folders/gr_bombieri/emartini/AMASS_torque_nomuscles/slurm/logs/amass_feeder_%j.err

set -euo pipefail
cd /storage-large/homedirs/mrtnrc33/opensim-batch-dynamics
unset PYTHONPATH
export PYTHONNOUSERSITE=1
/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python \
  scripts/submit_amass_arrays_with_capacity.py \
  --plan /storage-exp-large/shared-folders/gr_bombieri/emartini/AMASS_torque_nomuscles/slurm/submit_plan.json \
  --max-job-count 10000 \
  --safety-slots 100 \
  --poll-seconds 30 \
  --min-array-size 32 \
  --array-throttle 72 \
  --partition gpuRTX

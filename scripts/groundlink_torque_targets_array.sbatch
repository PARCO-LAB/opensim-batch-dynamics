#!/usr/bin/env bash
# One independent SLURM task per GroundLink subject. Each task writes its own
# manifest so parallel workers never contend over the same file.
#SBATCH --job-name=gl-torque-original
#SBATCH --partition=low
#SBATCH --array=0-5%6
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/storage-large/homedirs/mrtnrc33/storage/emartini/GroundLink_torque/torque_targets_12hz/logs/%A_%a.out

set -euo pipefail

subjects=(s001 s002 s003 s004 s005 s006)
subject="${subjects[${SLURM_ARRAY_TASK_ID}]}"
python="/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python"
repo="/home/accounts/personale/mrtnrc33/opensim-batch-dynamics"
dataset="/storage-large/homedirs/mrtnrc33/storage/emartini/GroundLink"
pipeline="/storage-large/homedirs/mrtnrc33/storage/emartini/GroundLink_torque/pipeline"
index="/storage-large/homedirs/mrtnrc33/storage/emartini/GroundLink_torque/h1_labels/index.csv"
output="/storage-large/homedirs/mrtnrc33/storage/emartini/GroundLink_torque/torque_targets_12hz"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$repo"
exec srun --ntasks=1 --cpus-per-task=1 "$python" \
  scripts/extract_groundlink_torque_targets.py \
  --dataset-root "$dataset" \
  --pipeline-root "$pipeline" \
  --index "$index" \
  --output-dir "$output" \
  --subject "$subject" \
  --resume \
  --protocol original_id_setup

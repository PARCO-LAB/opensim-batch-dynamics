#!/usr/bin/env bash
# One isolated subject task per array element; node008 is explicitly excluded.
# Submit only after read-only T_code recomputation and the Phase C preflight pass.
#SBATCH --job-name=gl-cop-tcode
#SBATCH --partition=medium
#SBATCH --exclude=node008
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/storage-large/homedirs/mrtnrc33/.codex_phase_c/tcode_%A_%a.out

set -euo pipefail

subjects=(s001 s002 s003 s004 s005 s006)
subject="${subjects[${SLURM_ARRAY_TASK_ID}]}"
python="/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python"
repo="/home/accounts/personale/mrtnrc33/opensim-batch-dynamics"
dataset="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink"
pipeline="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/pipeline"
index="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/h1_labels/index.csv"
nomuscles="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/torque_targets_12hz_nomuscles"
output="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/torque_targets_12hz_nomuscles_copfix_tcode"
tcode_reference="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/cop_diagnosis_B1bis/code_transform_by_trial.csv"
smplx_models="$repo/model/smpl"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$repo"
exec "$python" scripts/extract_groundlink_torque_targets.py \
  --dataset-root "$dataset" \
  --pipeline-root "$pipeline" \
  --index "$index" \
  --output-dir "$output" \
  --nomuscles-root "$nomuscles" \
  --cop-shift-source tcode \
  --tcode-reference-csv "$tcode_reference" \
  --smplx-model-dir "$smplx_models" \
  --subject "$subject" \
  --resume

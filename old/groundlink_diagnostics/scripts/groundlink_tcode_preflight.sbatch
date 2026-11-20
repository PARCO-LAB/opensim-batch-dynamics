#!/usr/bin/env bash
# Read-only 206-trial T_code reconstruction. Do not submit correction arrays
# until this task succeeds. node008 is explicitly excluded.
#SBATCH --job-name=gl-tcode-audit
#SBATCH --partition=medium
#SBATCH --exclude=node008
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/storage-large/homedirs/mrtnrc33/.codex_phase_c/tcode_preflight_%j.out

set -euo pipefail

python="/home/accounts/personale/mrtnrc33/miniconda3/envs/opensim-torque/bin/python"
repo="/home/accounts/personale/mrtnrc33/opensim-batch-dynamics"
dataset="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink"
pipeline="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/pipeline"
index="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/h1_labels/index.csv"
nomuscles="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/torque_targets_12hz_nomuscles"
tcode_reference="/storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/cop_diagnosis_B1bis/code_transform_by_trial.csv"
smplx_models="$repo/model/smpl"
audit="/storage-large/homedirs/mrtnrc33/.codex_phase_c/tcode_recomputed_by_trial.csv"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$repo"
exec "$python" scripts/extract_groundlink_torque_targets.py \
  --dataset-root "$dataset" \
  --pipeline-root "$pipeline" \
  --index "$index" \
  --output-dir "$nomuscles" \
  --nomuscles-root "$nomuscles" \
  --cop-shift-source tcode \
  --tcode-reference-csv "$tcode_reference" \
  --smplx-model-dir "$smplx_models" \
  --tcode-preflight-only \
  --tcode-preflight-output "$audit"

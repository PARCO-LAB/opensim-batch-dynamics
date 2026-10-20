#!/usr/bin/env bash
set -euo pipefail

RESUME=0
if [[ "${1:-}" == "--resume" ]]; then
  RESUME=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0 [--resume]" >&2
  exit 2
fi

# Recompute the complete GroundLink pipeline and the E4 inverse-dynamics
# reports for all seven subjects.
#
# Optional environment variables:
#   GROUNDLINK_ROOT     input dataset root
#   OUTPUT_ROOT         output root
#   ENV_NAME            micromamba environment name
#   MAMBA_ROOT_PREFIX   micromamba package/environment root
#   WORKERS             parallel pipeline workers
#   REFERENCE_THRESHOLD measured-vs-kinematic force mismatch threshold in BW
#   BSM_MODEL           scaled BSM model used by the pipeline
#   SMPLX_MODEL_DIR     directory containing SMPL-X model files

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GROUNDLINK_ROOT="${GROUNDLINK_ROOT:-/home/emartini/nas/MAEVE/dataset/GroundLink}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/emartini/nas/MAEVE/HUMAN_MODEL/GroundLink_torque}"
ADDBIO_ENGINE_ROOT="${ADDBIO_ENGINE_ROOT:-/home/emartini/AddBiomechanics}"
PYTHON_BIN="${PYTHON_BIN:-/home/emartini/miniconda3/envs/opensim-torque/bin/python}"
MICROMAMBA="${MICROMAMBA:-/tmp/micromamba-extract/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/tmp/mamba-root}"
ENV_NAME="${ENV_NAME:-opensim-torque}"
WORKERS="${WORKERS:-2}"
REFERENCE_THRESHOLD="${REFERENCE_THRESHOLD:-0.2}"
SMPLX_MODEL_DIR="${SMPLX_MODEL_DIR:-${REPO_ROOT}/model/smpl}"
BSM_MODEL="${BSM_MODEL:-${REPO_ROOT}/model/bsm/bsm.osim}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_dir() {
  [[ -d "$1" ]] || die "directory not found: $1"
}

require_file() {
  [[ -f "$1" ]] || die "file not found: $1"
}

require_dir "$REPO_ROOT"
require_dir "$GROUNDLINK_ROOT/moshpp"
require_dir "$GROUNDLINK_ROOT/force"
require_dir "$SMPLX_MODEL_DIR"
require_file "$BSM_MODEL"
require_file "$ADDBIO_ENGINE_ROOT/server/engine/src/engine.py"
if [[ -n "$PYTHON_BIN" ]]; then
  [[ -x "$PYTHON_BIN" ]] || die "Python executable not found or not executable: $PYTHON_BIN"
else
  [[ -x "$MICROMAMBA" ]] || die "set PYTHON_BIN or provide executable MICROMAMBA"
fi

mkdir -p "$OUTPUT_ROOT"

echo "GroundLink input:  $GROUNDLINK_ROOT"
echo "GroundLink output: $OUTPUT_ROOT"
echo "AddBiomechanics:   $ADDBIO_ENGINE_ROOT"
echo "Environment:       $ENV_NAME"
echo "Workers:            $WORKERS"
echo

pipeline_args=()
if [[ "$RESUME" -eq 1 ]]; then
  pipeline_args+=(--skip-existing)
else
  pipeline_args+=(--rerun-walking)
fi

# This stage audits the trials, reruns AddBiomechanics/ID for every valid trial,
# and regenerates audit/, pipeline/, and metrics/.
set +e
GROUNDLINK_ROOT="$GROUNDLINK_ROOT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
ADDBIO_ENGINE_ROOT="$ADDBIO_ENGINE_ROOT" \
MICROMAMBA="$MICROMAMBA" \
PYTHON_BIN="$PYTHON_BIN" \
MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" \
ENV_NAME="$ENV_NAME" \
WORKERS="$WORKERS" \
bash "$REPO_ROOT/scripts/run_groundlink_s001_all.sh" \
  "${pipeline_args[@]}" \
  "$SMPLX_MODEL_DIR" \
  "$BSM_MODEL" \
  "$ADDBIO_ENGINE_ROOT"
pipeline_status=$?
set -e
if [[ "$pipeline_status" -ne 0 ]]; then
  echo "Warning: some trials failed their quality gate; E4 will skip explicit rejections." >&2
fi

echo
echo "Exporting E4 torque files and residual metrics..."

for subject in s001 s002 s003 s004 s005 s006 s007; do
  echo "[E4] $subject"
  env \
    PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts" \
    "$PYTHON_BIN" \
    "$REPO_ROOT/scripts/run_groundlink_e4.py" \
      --dataset-root "$GROUNDLINK_ROOT" \
      --pipeline-root "$OUTPUT_ROOT/pipeline" \
      --output-dir "$OUTPUT_ROOT/e4_id" \
      --subject "$subject" \
      --reference-force-balance-threshold-bw "$REFERENCE_THRESHOLD"
done

echo
echo "Done. Updated outputs:"
echo "  audit:   $OUTPUT_ROOT/audit"
echo "  pipeline: $OUTPUT_ROOT/pipeline"
echo "  metrics:  $OUTPUT_ROOT/metrics"
echo "  E4/ID:    $OUTPUT_ROOT/e4_id"

exit "$pipeline_status"

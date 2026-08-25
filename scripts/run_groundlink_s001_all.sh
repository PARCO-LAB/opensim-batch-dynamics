#!/usr/bin/env bash
set -euo pipefail

# Run the complete GroundLink pipeline for all seven subjects.
# Usage: run_groundlink_s001_all.sh [--subject <subject>] [--skip-existing] [--rerun-walking] <smplx-model-dir> <bsm-model.osim> [addbiomechanics-root]

SKIP_EXISTING=0
RERUN_WALKING=0
SUBJECT=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --rerun-walking) RERUN_WALKING=1; shift ;;
    --subject)
      [[ $# -ge 2 ]] || { echo "Missing value for --subject." >&2; exit 2; }
      SUBJECT="$2"
      shift 2
      ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
set -- "${POSITIONAL[@]}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 [--subject <subject>] [--skip-existing] [--rerun-walking] <smplx-model-dir> <bsm-model.osim> [addbiomechanics-root]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMPLX_MODEL_DIR="$1"
BSM_MODEL="$2"
ADDBIO_ROOT="${3:-${ADDBIO_ENGINE_ROOT:-/tmp/AddBiomechanics}}"
GROUNDLINK_ROOT="${GROUNDLINK_ROOT:-/home/emartini/nas/MAEVE/dataset/GroundLink}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/emartini/nas/MAEVE/HUMAN_MODEL/GroundLink_torque}"
WORKERS="${WORKERS:-2}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/tmp/mamba-root}"
MICROMAMBA="${MICROMAMBA:-/tmp/micromamba-extract/bin/micromamba}"
ENV_NAME="${ENV_NAME:-opensim-torque}"

if [[ ! -d "$SMPLX_MODEL_DIR" || ! -f "$BSM_MODEL" || ! -d "$GROUNDLINK_ROOT/moshpp" || ! -d "$GROUNDLINK_ROOT/force" ]]; then
  echo "Missing SMPL-X directory, BSM model, or GroundLink moshpp/force roots." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
AUDIT_DIR="$OUTPUT_ROOT/audit"
PIPELINE_DIR="$OUTPUT_ROOT/pipeline"
METRICS_DIR="$OUTPUT_ROOT/metrics"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/groundlink-all-stage.XXXXXX")"
trap 'rm -rf "$STAGE_DIR"' EXIT

run_python() {
  env MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/scripts" \
    "$MICROMAMBA" run -n "$ENV_NAME" python "$@"
}

subject_config() {
  case "$1" in
    s001) echo "female 69.86 1.68" ;;
    s002) echo "male 66.68 1.70" ;;
    s003) echo "female 53.07 1.59" ;;
    s004) echo "male 71.67 1.82" ;;
    s005) echo "male 90.70 1.95" ;;
    s006) echo "female 48.99 1.53" ;;
    s007) echo "female 63.96 1.75" ;;
    *) echo "Unknown subject: $1" >&2; exit 1 ;;
  esac
}

SUBJECTS=(s001 s002 s003 s004 s005 s006 s007)
[[ -n "$SUBJECT" ]] && SUBJECTS=("$SUBJECT")
BATCH_FAILURES=0
for subject in "${SUBJECTS[@]}"; do
  read -r sex mass height < <(subject_config "$subject")
  motion_root="$GROUNDLINK_ROOT/moshpp/$subject"
  force_root="$GROUNDLINK_ROOT/force/$subject"
  audit_subject_dir="$AUDIT_DIR/$subject"
  pipeline_subject_dir="$PIPELINE_DIR/$subject"
  metrics_subject_dir="$METRICS_DIR/$subject"
  stage_subject_dir="$STAGE_DIR/$subject"

  echo "[1/3] Auditing $subject"
  if ! run_python "$REPO_ROOT/scripts/validate_groundlink.py" \
      --input-root "$motion_root" \
      --force-root "$force_root" \
      --subject "$subject" \
      --mass-kg "$mass" \
      --height-m "$height" \
      --output-dir "$audit_subject_dir" \
      --mode audit; then
    echo "Warning: $subject has invalid trials; continuing with valid trials." >&2
  fi

  mkdir -p "$stage_subject_dir"
  awk -F',' -v sid="$subject" 'NR > 1 && $1 == sid && $7 == "ok" {print $3}' \
    "$audit_subject_dir/manifest.csv" |
  while IFS= read -r motion_path; do
    [[ -n "$motion_path" ]] || continue
    ln -s "$motion_path" "$stage_subject_dir/$(basename "$motion_path")"
  done

  staged_count="$(find "$stage_subject_dir" -type l | wc -l)"
  if [[ "$staged_count" -eq 0 ]]; then
    echo "No valid trials staged for $subject." >&2
    exit 1
  fi

  echo "[2/3] Running $subject batch ($staged_count trials)"
  if [[ "$SKIP_EXISTING" -eq 1 ]]; then
    skip_existing_arg="--skip-existing-csv"
  else
    skip_existing_arg="--no-skip-existing-csv"
  fi
  if run_python "$REPO_ROOT/scripts/run_amass_batch_parallel.py" \
      --input-root "$stage_subject_dir" \
      --output-dir "$pipeline_subject_dir" \
      --workers "$WORKERS" \
      "$skip_existing_arg" \
      $([[ "$RERUN_WALKING" -eq 1 ]] && echo "--rerun-walking") \
      --no-cleanup-intermediate \
      --smplx-model-dir "$SMPLX_MODEL_DIR" \
      --bsm-model "$BSM_MODEL" \
      --addbio-root "$ADDBIO_ROOT" \
      --sex "$sex" \
      --subject-mass-kg "$mass" \
      --subject-height-m "$height" \
      --id-cutoff-hz 12 \
      --walking-contact-height-threshold-m 0.025 \
      --walking-contact-speed-threshold-mps 0.35 \
      --id-contact-bodies calcn_l,toes_l,calcn_r,toes_r; then
    :
  else
    BATCH_FAILURES=1
    echo "Warning: $subject has failed batch trials; continuing with remaining subjects." >&2
  fi

  echo "[3/3] Evaluating $subject GRF, CoP, and contact"
  if ! run_python "$REPO_ROOT/scripts/validate_groundlink.py" \
      --input-root "$motion_root" \
      --force-root "$force_root" \
      --subject "$subject" \
      --mass-kg "$mass" \
      --height-m "$height" \
      --output-dir "$metrics_subject_dir" \
      --mode evaluate \
      --pipeline-dir "$pipeline_subject_dir"; then
    echo "Warning: $subject evaluation found invalid trials; continuing." >&2
  fi
done

echo "Aggregating subject metrics"
run_python - "$METRICS_DIR" "${SUBJECTS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for subject in sys.argv[2:]:
    path = root / subject / "metrics_per_trial.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        records.extend(csv.DictReader(handle))
if not records:
    raise SystemExit("No subject metrics found")
fields = list(records[0])
with (root / "metrics_per_trial.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
groups = {}
for row in records:
    try:
        value = float(row["value"])
    except ValueError:
        continue
    if value == value and value not in (float("inf"), float("-inf")):
        groups.setdefault((row["method"], row["metric"]), []).append(value)
with (root / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["method", "metric", "value", "unit"])
    writer.writeheader()
    for (method, metric), values in sorted(groups.items()):
        unit = next(row["unit"] for row in records if row["method"] == method and row["metric"] == metric)
        writer.writerow({"method": method, "metric": metric, "value": sum(values) / len(values), "unit": unit})
PY

echo "Done. Results:"
echo "  audit:   $AUDIT_DIR"
echo "  pipeline: $PIPELINE_DIR"
echo "  metrics: $METRICS_DIR"

if [[ "$BATCH_FAILURES" -ne 0 ]]; then
  echo "Completed with batch failures; inspect */batch_summary.json and logs." >&2
  exit 1
fi

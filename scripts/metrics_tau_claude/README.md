# GroundLink metrics and LOSO workflow

This folder contains the portable Table I evaluator, paired comparisons, extra GRF diagnostics, a T_code target-extraction entrypoint, and an alternative-load OpenSim ID runner. The scripts do not change files under the existing pipeline or AMASS directories. Write outputs into a new `pipeline_tuning/<run-name>/` directory.

## Environment and paths

Use Python 3.10+ with OpenSim 4.5.2, NumPy, pandas, SciPy, and (for ID runs) either the OpenSim Python bindings or `opensim-cmd` on `PATH`.

Typical roots on `parco07` are:

```text
NAS:       ~/nas/MAEVE/GroundLink_torque/
Force:     ~/nas/MAEVE/GroundLink/force/
HPC:       /storage-exp-large/shared-folders/gr_bombieri/emartini/GroundLink_torque/
Local run: ~/hdd/groundlink-grf-tuning/GroundLink_torque/
```

The NAS and local roots must contain `pipeline/`, `h1_labels/`, and `torque_targets_12hz_nomuscles_copfix_tcode/`. The measured force root contains `s001/` through `s006/` with the original `.npy` files. Paths in `run_metadata.json` point to the HPC tree; `tau_table1_all.py` maps them to `--torque-root` and `--force-root`.

## Baseline gate

Run the baseline first. Its output directory must be new:

```bash
python tau_table1_all.py \
  --torque-root ~/nas/MAEVE/GroundLink_torque \
  --force-root ~/nas/MAEVE/GroundLink/force \
  --output-dir ~/nas/MAEVE/GroundLink_torque/pipeline_tuning/2026-11-16_baseline
```

The evaluator reports per-foot force scores, measured-contact Fz RMSE/correlation, contact F1, net-force residual, torque RMSE and Pearson r by coordinate, CoP error, horizontal-force RMSE, false-contact rates, and lag. Contact uses measured `Fz > 0.05 BW`; measured force is zero-phase 4th-order Butterworth low-pass filtered at 12 Hz from 250 Hz. Torque rows use only valid frames where the corresponding measured foot is in stance. GRF normalization uses `subject_mass_kg` from the pipeline CSV; torque normalization uses the OpenSim model mass.

Check the six torque coordinates and the GRF, CoP, horizontal-force, and false-contact metrics against the baseline notes before proceeding. If any baseline is outside tolerance, stop and document the exact masks, denominators, trial coverage, and code/data versions. Do not start decomposition, estimator ablations, LOSO, or AMASS work until the gate passes.

Run optional summaries from the baseline CSV:

```bash
python tau_table1_paired.py --metrics ~/nas/MAEVE/GroundLink_torque/pipeline_tuning/2026-11-16_baseline/baseline_metrics.csv
python grf_extra.py --metrics ~/nas/MAEVE/GroundLink_torque/pipeline_tuning/2026-11-16_baseline/baseline_metrics.csv
```

## Target extraction and reference ID

`tau_extract_tcode.py` forwards to `scripts/extract_groundlink_torque_targets.py` in the active repository. Set `OPEN_SIM_BATCH_DYNAMICS_ROOT` or pass `--repo-root` before the extractor's normal arguments:

```bash
export OPEN_SIM_BATCH_DYNAMICS_ROOT=/storage-large/homedirs/mrtnrc33/opensim-batch-dynamics
python tau_extract_tcode.py --help
```

`refid/ref_id.py` runs a saved OpenSim `InverseDynamicsTool` setup with a supplied alternative ExternalLoads XML. It writes a new setup and the generated `.sto` only in `--output-dir`. `refid/batch.py` accepts a CSV with columns `setup,external_loads_file,output_dir` and optional `output_name,forces_to_exclude`. For SLURM arrays, set one zero-based manifest row per task (`--row-index` or `SLURM_ARRAY_TASK_ID`). Do not point any output directory at an existing result.

Example scheduler wrapper for independent ID rows:

```bash
#!/bin/bash
#SBATCH --job-name=groundlink-id
#SBATCH --array=0-5
#SBATCH --time=08:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/id-%A_%a.log
python refid/batch.py --manifest refid_manifest.csv --row-index "$SLURM_ARRAY_TASK_ID"
```

Submit long jobs with `sbatch` and keep the job ID, manifest, and logs in the run folder. Never evaluate a held-out subject while selecting parameters.

## Baseline numbers supplied for reproduction

```text
GRF (pipeline): Fz RMSE_contact 0.137 BW; r_contact 0.831; contact F1 0.957; rho 0.010 BW
Torque: mean RMSE 0.344 N·m/kg; mean Pearson r 0.547; ankle r about 0.36
Torque split references: BW 0.401 / 0.299; CoM 0.431 / 0.441
CoP point error 4.01 cm; horizontal-force RMSE 0.093 BW
False contacts: compare with the supplied subject-level 8–25% baseline and record the denominator
```

These are gate values, not expected improvements. Preserve the published masks and aggregation; report lag without shifting signals.

## Required outputs after the gate passes

Use a fresh `pipeline_tuning/<run-name>/` and include `report.md`, `baseline_metrics.csv`, `decomposition.csv`, `ablation.csv`, `loso_folds.csv`, `loso_params.csv`, and `amass_sample.csv`. The report order is: summary, code/data version, reproduced baseline, torque decomposition, single-method ablations, LOSO, AMASS transfer, recommendation and code diff. Copy results to the NAS and compare SHA-256 hashes. Do not overwrite the source target folders or AMASS.

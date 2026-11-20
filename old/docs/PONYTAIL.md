# PONYTAIL Refactor Brief

Repo: `opensim-batch-dynamics`

Scope: over-engineering, duplication, generated artifacts, and dependency bloat only. Do not use this brief for correctness, security, or performance rewrites.

Mode: ponytail full. Delete first. Reuse existing code before writing new code. Add the smallest black-box regression test before each behavior-preserving refactor. Run tests after every step and stop on any behavior change.

## Current Worktree Note

There are already local changes:

- `src/opensim_batch_dynamics/addbio_csv_export.py` delegates to `mot_to_csv.convert_mot_to_model_csv`.
- `tests/test_addbio_csv_export.py` pins the current AddBiomechanics CSV export behavior with the real `model/bsm/bsm.osim` fixture.

Do not undo those changes unless explicitly asked.

## Test Commands

Fast non-OpenSim check:

```bash
PYTHONPATH=src python -m unittest \
  tests.test_addbio_csv_export \
  tests.test_bsm_markers \
  tests.test_carepd_gait_severity \
  tests.test_carepd_mocha_training_dataset
```

Full suite in an environment with `opensim` installed:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Known local blocker: the full suite fails without the `opensim` Python package because `tests/test_inverse_dynamics_no_grf.py` imports and exercises OpenSim.

## Agent Rules

1. Do one item at a time.
2. Add or extend one black-box regression test before changing production code.
3. Prefer deletion over abstraction.
4. Keep public BSM behavior stable unless the item explicitly says it is a product decision.
5. If a step needs a product decision, stop and ask. Do not silently delete public behavior.
6. After each item, run the smallest relevant test command plus the fast non-OpenSim check.

## Ranked Refactor Items

### 1. `delete:` Legacy OpenCap/LaiUhlrich Pipeline

What to cut:

- `scripts/run_amass_to_opencap_legacy.py`
- `scripts/run_amass_to_opensim.py`
- `src/opensim_batch_dynamics/opensim_pipeline.py`
- `src/opensim_batch_dynamics/opencap_markers.py`
- `src/opensim_batch_dynamics/config.py`
- `assets/opencap/`
- `model/LaiUhlrich2022_torque_only.osim`

Why:

- The README says the main product is AMASS to BSM unified CSV.
- `TODO.md` already says the OpenCap/LaiUhlrich path is not the final target.
- This legacy path is about 8.6k tracked lines plus assets and keeps old concepts in the package export.
- `src/opensim_batch_dynamics/__init__.py` still exports `run_amass_to_opensim`, which makes the legacy path look primary.

Replacement:

- Keep `scripts/run_amass_to_bsm_csv.py` as the main entrypoint.
- Update `src/opensim_batch_dynamics/__init__.py` to export the BSM-facing API only, or export nothing if there is no stable library API yet.
- Update README legacy section to say the OpenCap path was removed, if deletion is accepted.

Test first:

- Add an import smoke test that imports `opensim_batch_dynamics` and the BSM modules used by `scripts/run_amass_to_bsm_csv.py`.
- Keep `tests/test_bsm_markers.py`, `tests/test_addbio_csv_export.py`, and `tests/test_inverse_dynamics_no_grf.py` passing.

Stop condition:

- This deletes public legacy CLI behavior. Ask before applying unless the user explicitly wants legacy removal.

### 2. `delete:` Duplicate Geometry Tree

What to cut:

- One of `model/Geometry/` or `model/bsm/Geometry/`.

Why:

- Both directories are 8.7 MB.
- They share 195 filenames.
- Most files are duplicated model geometry under two paths.

Replacement:

- Keep `model/bsm/Geometry/` because the BSM pipeline references it directly.
- Delete `model/Geometry/` after legacy OpenCap deletion.
- If legacy is still kept, use a symlink or update the old path to point at `model/bsm/Geometry/`.

Test first:

- Add a test around `default_bsm_asset_paths().ensure_exists()` and `build_addbiomechanics_subject_folder()` using the real BSM model and geometry path.
- Run `tests.test_inverse_dynamics_no_grf` in an OpenSim environment if model path assumptions change.

Stop condition:

- Do not delete `model/Geometry/` while the OpenCap/LaiUhlrich path remains active unless the replacement path is tested.

### 3. `delete:` Tracked Generated Artifacts And Local Junk

What to cut:

- `.DS_Store`
- `model/.DS_Store`
- `model/Geometry/.DS_Store`
- `model/bsm/.DS_Store`
- `outputs/.DS_Store`
- `src/.DS_Store`
- `TOFIX.md`
- `RT/tmp_rt_sequence.csv`
- `RT/tmp_rt_vs_offline.pdf`
- `RT/tmp_rt_vs_offline_all_dofs.pdf`
- `RT/tmp_rt_vs_offline_from_csv.pdf`
- `RT/amass_batch_results.json`
- `RT/amass_clean_artifacts.json`
- `RT/test_dynamics.ipynb`
- `data/A3-_Swing_arms_stageii.npz`

Why:

- These are generated artifacts, machine-local files, empty files, or sample data.
- They add noise and make audits slower.
- The sample `.npz` is 3.4 MB and belongs in download instructions or test fixtures only if required.

Replacement:

- Keep a tiny deterministic fixture if a test needs real NPZ behavior.
- Add ignore rules for `.DS_Store`, `RT/tmp_*`, generated PDFs, generated JSON reports, `outputs/`, and large local data.

Test first:

- Verify no tests reference these exact files.
- If `data/A3-_Swing_arms_stageii.npz` is needed by docs only, replace with README instructions.

Stop condition:

- If any test or script requires the sample NPZ as a fixture, replace it with the smallest fixture before deleting it.

### 4. `shrink:` Batch Runner Duplication

What to cut:

- Duplicated task discovery, trial naming, command building, skip checks, manifest writing, sbatch writing, worker JSON, retry logic.

Files:

- `scripts/run_amass_batch_parallel.py`
- `scripts/run_amass_batch_slurm.py`
- `scripts/run_ntu_skeleton_batch_slurm.py`
- `scripts/run_humanml3d_joints_batch_slurm.py`

Why:

- The same helper names and logic repeat across these scripts: `BatchTask`, `_build_tasks`, `_write_manifest`, `_read_manifest_record`, `_write_json`, `_write_sbatch_script`, `_resolve_submit_path`, `_should_retry_sbatch`, `_submit_chunk_with_retry`.
- Current duplication makes any scheduler fix a multi-file edit.

Replacement:

- Extract only the shared boring pieces into one small module, for example `scripts/batch_common.py`.
- Keep dataset-specific command builders inside each script.
- Do not invent classes beyond the existing dataclass shape unless it removes real duplication.

Test first:

- Add dry-run tests for each runner that do not submit jobs.
- Use temporary input trees with one fake input file each.
- Assert the generated manifest command and output path only.

Stop condition:

- If a shared helper needs more parameters than the duplicated code, do not extract it.

### 5. `yagni:` CARE-PD Converter Backend Modes

What to cut:

- `--pipeline classic`
- `--pipeline cpu-take`
- Classic temporary OBJ/PKL transfer helpers if not actively used:
  - `_write_smpl_meshes`
  - `_write_transfer_config`
  - `_run_transfer_model`
  - `_run_merge_output`
  - `_merged_pkl_to_amass_npz`
  - `_convert_one_take`
  - `_convert_one_take_cpu_takes`

File:

- `scripts/convert_carepd_smpl_pkl_to_smplx_npz.py`

Why:

- Default is `in-memory`.
- `MOCHA_CHALLENGE.md` says the classic path is too slow and fragile.
- Multiple backends inflate the converter to about 2k lines.

Replacement:

- Keep only `in-memory` unless there is current evidence that another backend is required.
- Keep CLI compatibility only if the user explicitly needs old commands. Otherwise fail fast on removed modes with a clear message for one release.

Test first:

- Add black-box tests around `parse_args()` and `--dry-run` planning for a tiny synthetic pickle.
- Pin the current default pipeline value before deleting alternatives.

Stop condition:

- This may change CLI compatibility. Ask before removing modes if old command support matters.

### 6. `delete:` Embedded BSM Marker Map Fallback

What to cut:

- `_DEFAULT_BSM_MARKERS_SMPLX_TEXT` in `src/opensim_batch_dynamics/bsm_markers.py`.

Why:

- The real asset exists at `assets/smpl2ab/bsm_markers_smplx.yaml`.
- The embedded fallback has drifted: the YAML has `RWRB`, fallback does not.
- Carrying two marker maps creates a hidden behavior split.

Replacement:

- Require the YAML asset by default.
- Resolve the default path from repo root if `yaml_path is None`.
- Keep the regex parser as the no-PyYAML fallback for the real YAML text.

Test first:

- Update `tests/test_bsm_markers.py` to assert `load_bsm_marker_map(None)` reads the same 105 markers as the asset and includes `RWRB`.
- Keep the toe-marker tests.

Stop condition:

- If the package must work when installed without repo assets, keep fallback but generate it from the asset in one place.

### 7. `shrink:` Duplicate Plot Report Code

What to cut:

- Repeated PDF/report helpers:
  - `add_title_page`
  - `add_overview_page`
  - `add_dof_pages`
  - `add_grf_pages`
  - `build_pdf_report`

Files:

- `scripts/csv_explorer.py`
- `scripts/realtime_vs_offline_pdf.py`

Why:

- Both scripts build similar multi-page matplotlib PDFs.
- This is lower priority because it is script-only, but the duplication is visible and large.

Replacement:

- Keep one PDF helper in `scripts/csv_report_common.py`.
- Let each script pass its domain-specific data structure to the common page functions.

Test first:

- Add a tiny CSV fixture and assert both scripts can generate a non-empty PDF.
- Do not inspect PDF pixels unless a layout bug is being fixed.

Stop condition:

- If common code needs a complex abstraction layer, leave duplication alone.

### 8. `shrink:` Duplicate Signal Helpers

What to cut:

- Duplicate `_lowpass_butterworth_4th` in `src/opensim_batch_dynamics/inverse_dynamics_no_grf.py`.
- Duplicate tiny `_to_float` helpers where a direct `float()` with local error handling is enough.

Files:

- `src/opensim_batch_dynamics/mot_to_csv.py`
- `src/opensim_batch_dynamics/inverse_dynamics_no_grf.py`
- `scripts/build_carepd_mocha_training_dataset.py`
- `scripts/run_nimble.py`
- `scripts/convert_carepd_smpl_pkl_to_smplx_npz.py`
- `src/opensim_batch_dynamics/final_csv_export.py`
- `src/opensim_batch_dynamics/amass_loader.py`

Why:

- `mot_to_csv.py` already owns filtering and differentiation for OpenSim motion files.
- Small repeated helpers are tolerable, but this is easy cleanup after bigger deletions.

Replacement:

- Reuse the `mot_to_csv.py` filter where behavior matches.
- Leave local helpers where exception behavior differs.

Test first:

- Add or extend a test that checks short signals are returned unchanged and `cutoff_hz=None` returns a copy.

Stop condition:

- Do not unify helpers if one call site intentionally swallows missing SciPy and another intentionally raises.

### 9. `delete:` Dependency Bloat In Base Environment

What to cut from the base `environment.yml` if the related feature is moved or deleted:

- `rtree`
- `shapely`
- `pyrender`
- `open3d`
- `loguru`
- `omegaconf`
- `cvxpy`
- `pandas`
- `pyyaml`

Observed imports outside vendored/local ignored trees:

- `cvxpy`: `RT/rt_library.py`
- `pandas`: `RT/real_time_test.py`
- `trimesh`: `src/opensim_batch_dynamics/bsm_subject_json.py`
- `meshio`: `scripts/run_nimble.py`

Why:

- Several environment entries support RT experiments, visualization, or local transfer tooling, not the BSM pipeline core.
- Heavy optional deps slow setup and make CI harder.

Replacement:

- Keep `trimesh` in base if subject mass estimation remains.
- Keep `meshio` only if `scripts/run_nimble.py` is considered core.
- Move RT-only deps to an `environment-rt.yml` or document them as optional.
- Remove `pyyaml` only after `bsm_markers.py` can parse the real YAML via stdlib fallback.

Test first:

- Run import smoke tests for the core package in a fresh env.
- Keep one test for marker YAML parsing without PyYAML if removing `pyyaml`.

Stop condition:

- Do not remove deps used by the selected main pipeline until a fresh environment run proves imports still work.

## Already Done In This Worktree

### `shrink:` AddBiomechanics CSV Export Duplicate Logic

Done:

- `src/opensim_batch_dynamics/addbio_csv_export.py` now uses `mot_to_csv.convert_mot_to_model_csv`.
- `tests/test_addbio_csv_export.py` pins current output with real `model/bsm/bsm.osim`.
- `src/opensim_batch_dynamics/bsm_markers.py` now reads the repo YAML asset by default instead of the embedded fallback, and `tests/test_bsm_markers.py` pins that behavior.
- `src/opensim_batch_dynamics/final_csv_export.py` now parses numeric CSV cells inline, and `tests/test_final_csv_export.py` pins blank-cell handling.
- `src/opensim_batch_dynamics/amass_loader.py` now parses the scalar frame-rate inline, and `tests/test_amass_loader.py` pins the zero-dim numpy scalar path.
- `scripts/build_carepd_mocha_training_dataset.py` now parses numeric cells inline, and `tests/test_carepd_mocha_training_dataset.py` pins blank-cell handling.
- `scripts/csv_explorer.py` now parses numeric cells inline, and `tests/test_csv_explorer.py` pins blank-cell handling with a minimal matplotlib stub.
- `scripts/report_common.py` now holds shared pure report helpers, and `tests/test_report_common.py` pins the shared math/format behavior.
- `scripts/report_common.py` now also holds the shared text-page renderer used by the two report scripts, and `tests/test_report_common.py` pins it with a fake `PdfPages`/`pyplot` harness.
- `scripts/report_common.py` now also holds the shared PDF wrapper used by `scripts/csv_explorer.py` and `scripts/realtime_vs_offline_pdf.py`, and `tests/test_report_common.py` plus `tests/test_pdf_reports.py` pin it with fake backends and tiny CSV fixtures.
- `scripts/run_nimble.py` now reuses the shared translational DOF check and inlines the row parser, and `tests/test_run_nimble.py` pins both the error message and translational conversion.
- `src/opensim_batch_dynamics/mot_to_csv.py` now owns the shared Butterworth low-pass helper, `src/opensim_batch_dynamics/inverse_dynamics_no_grf.py` reuses it with copy-on-missing-SciPy behavior, and `tests/test_lowpass_filter.py` pins both behaviors.
- `scripts/convert_carepd_smpl_pkl_to_smplx_npz.py` now inlines the single-use frame-rate fallback, and `tests/test_convert_carepd_smpl_pkl_to_smplx_npz.py` pins the record parser with stubbed `smplx` imports.
- `environment.yml` now keeps the base env on the core pipeline, `environment-rt.yml` holds RT-only extras, and `tests/test_inverse_dynamics_no_grf.py` skips cleanly when `opensim` is unavailable.
- `scripts/batch_common.py` now holds the shared batch submit-path resolver and JSON writer used by the AMASS, NTU, and HumanML SLURM launchers.
- `scripts/batch_common.py` also now holds the shared manifest JSONL reader used by the AMASS, NTU, and HumanML SLURM launchers, with `tests/test_batch_common.py` pinning it.
- `scripts/batch_common.py` now also holds the shared non-empty file check used by the batch runners, with `tests/test_batch_common.py` pinning it.
- Tracked generated junk is deleted from the worktree: `.DS_Store`, `RT/*` reports and notebooks, `TOFIX.md`, and `data/A3-_Swing_arms_stageii.npz`.
- `.gitignore` now ignores `.DS_Store` and the local `RT/` report junk that keeps reappearing during refactors.
- `tests/test_amass_batch_runners.py` pins dry-run behavior for the AMASS, NTU, and HumanML batch launchers with real temp fixtures.

Run:

```bash
PYTHONPATH=src python -m unittest tests.test_addbio_csv_export
```

## Suggested Order

1. Marker-map fallback cleanup.
2. Generated artifacts and junk cleanup.
3. Legacy OpenCap decision and deletion if approved.
4. Duplicate geometry after the legacy decision.
5. Batch runner common helpers.
6. CARE-PD backend mode reduction if CLI compatibility can change.
7. Optional report-code and signal-helper cleanup.
8. Environment split after code paths are settled.

## Net Estimate

Conservative possible reduction:

- About 8k lines if legacy OpenCap/LaiUhlrich is removed.
- About 8.7 MB if one geometry tree is removed.
- About 5 MB of tracked local/generated artifacts removable immediately if no tests depend on them.
- About 300 to 700 Python lines from runner/helper/backend simplification without changing the main BSM pipeline.
- About 5 to 8 base dependencies can move out of the core environment after RT/visualization paths are isolated.

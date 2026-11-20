# CLAUDE.md

## Project

This repository converts AMASS/SMPL-X motion into BSM dynamics CSVs through
AddBiomechanics, inverse dynamics, and a quality gate. The main entry point is
`scripts/run_amass_to_bsm_csv.py`; local and SLURM batch runners are in
`scripts/run_amass_batch_parallel.py` and `scripts/run_amass_batch_slurm.py`.

Core Python modules are in `src/opensim_batch_dynamics/`. The supported model
and geometry are under `model/bsm/`; calibrated GroundLink models are under
`model/groundlink_calibrated/`. Dataset converters and CSV/PDF reports remain
in `scripts/` and are documented in `docs/usage_reference.md`.

## Environment and tests

Use the `opensim-torque` Conda environment from `environment.yml`. Tests use
`unittest`:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

The `tests/` directory contains supported pipeline and conversion tests.
Unmaintained prototypes, legacy pipelines, and one-off utilities are preserved
under `old/` and are not part of test discovery.

## Core pipeline notes

- The single-file pipeline is `scripts/run_amass_to_bsm_csv.py`.
- `scripts/batch_common.py` is shared by local and SLURM batch runners.
- External AddBiomechanics assets are passed with `--addbio-root` or
  `$ADDBIO_ENGINE_ROOT`; local SMPL-X model assets are not tracked.
- GroundLink evaluation scripts and calibrated model files remain supported;
  one-off diagnostics are under `old/groundlink_diagnostics/`.

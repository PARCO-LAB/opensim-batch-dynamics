# Automated Dynamics Annotation of Human Motion Archives

An automated pipeline that converts SMPL/SMPL-X motion into musculoskeletal
kinematics, ground reaction forces and joint torques.

The pipeline labels SMPL/SMPL-X motion archives, such as AMASS, that have no
force measurements. It estimates ground reaction forces (GRF) from body motion
alone and computes joint torques by inverse dynamics. It requires no force
measurements and no training, so it scales to large archives, for example
to provide joint-torque targets for learning motion representations. Its
force and torque estimates are evaluated against force-plate measurements
from GroundLink.

For each input sequence, the pipeline writes one CSV with:

- joint angles, velocities and accelerations of a 49-DOF musculoskeletal model;
- estimated joint torques $\hat\tau$;
- estimated GRF for each contact body and in total;
- binary contact states for each contact body.

## Contents

- [Method overview](#method-overview)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Annotating AMASS](#annotating-amass)
- [Calibrated variant](#calibrated-variant)
- [Evaluation on GroundLink](#evaluation-on-groundlink)
- [Output format](#output-format)
- [Repository structure](#repository-structure)
- [License and acknowledgements](#license-and-acknowledgements)

## Method overview

```
SMPL-X motion (.npz)
  │  1. SMPL-X mesh → virtual markers → AddBiomechanics: scaled BSM model + IK
  ▼
Musculoskeletal kinematics q, q̇, q̈ (49 DOFs)
  │  2. Total contact force from whole-body dynamics
  │  3. Contact detection and distribution of the force over the feet
  ▼
Per-foot GRF and application points
  │  4. OpenSim inverse dynamics
  ▼
Joint torques τ̂  →  quality gate  →  unified CSV
```

**1. Biomechanical model.** We generate a body mesh from SMPL-X and place
virtual markers at fixed vertices
([`assets/smpl2ab/bsm_markers_smplx.yaml`](assets/smpl2ab/bsm_markers_smplx.yaml)).
[AddBiomechanics](https://github.com/keenon/AddBiomechanics) uses these
markers to scale the biomechanical skeleton model (BSM) of
[SKEL](https://skel.is.tue.mpg.de/), derived from the model of Rajagopal et
al., and solves inverse kinematics. The model has 49 DOFs, including six at
the pelvis, with consistent segment inertias. Body mass is computed from the
mesh volume with a constant density of 985 kg m⁻³, or estimated from stature
when the shape is unavailable.

**2. Total contact force.** After zero-phase fourth-order Butterworth
filtering, the total external force required by the motion is

$$\mathbf F_{\text{target}}(t) = m\,\bigl(\ddot c(t) - \mathbf g\bigr),$$

where $c$ is the center of mass. In the implementation, this force is taken
from a first OpenSim inverse-dynamics pass without external loads: the
residual force at the pelvis is exactly the external force that the motion
requires, so the final inverse dynamics uses the same filtered kinematics.

**3. Contact detection and load distribution.** The support plane is placed
at the first percentile of foot-marker heights. A foot is a contact candidate
when its lowest marker is within 4 cm of the plane and its speed is below
0.5 m s⁻¹; hysteresis and smoothing give continuous activations
$\alpha_i\in[0,1]$. The force is shared among the feet according to their
activation and their planar distance $d_i$ to the projected center of mass,

$$w_i = \frac{\alpha_i/(d_i+\epsilon)}{\sum_j \alpha_j/(d_j+\epsilon)},$$

and each foot receives $w_i\mathbf F_{\text{target}}$, so the per-foot forces
sum to the required total. Forces are applied at the contact-body origins
projected onto the support plane. The residual
$\rho=\lVert\mathbf F_{\text{target}}-\sum_i\mathbf F_i\rVert/\text{BW}$ is
recorded for every frame.

**4. Inverse dynamics.** OpenSim inverse dynamics computes the joint torques
$\hat\tau$ under the estimated loads. A quality gate rejects sequences that
are too short, with implausible global translation or pelvis rotation, or
with diverging torques. Accepted sequences are written to the unified CSV.

The pipeline assumes foot-only contact on a planar support, fixed application
points and zero free moment. Sequences that violate these assumptions, such as
sitting, crawling or stair climbing, should be excluded before use.

## Installation

Runs on Linux and macOS (Apple Silicon and Intel) with Python 3.11 and
OpenSim 4.5.2.

```bash
git clone https://github.com/PARCO-LAB/opensim-batch-dynamics.git
cd opensim-batch-dynamics

conda env create -f environment.yml
conda activate opensim-torque

# AddBiomechanics engine (scaling and inverse kinematics)
git clone https://github.com/keenon/AddBiomechanics.git "$HOME/AddBiomechanics"
```

Download the SMPL-X model from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/)
(registration required) and place `SMPLX_NEUTRAL.npz` (and the gendered
models, if needed) in `model/smpl/`. The BSM model and its geometry are
included in `model/bsm/`.

Optional dependencies for the dataset converters (NTU RGB+D, HumanML3D,
CARE-PD) are listed in [`docs/usage_reference.md`](docs/usage_reference.md).

## Quick start

Annotate one AMASS sequence:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/A3-_Swing_arms_stageii.npz \
  --trial A3_swing \
  --output-dir outputs/bsm \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

The result is `outputs/bsm/A3_swing.csv`, plus a `.quality.json` sidecar.
Use `--trial all` to export one CSV per trial from a multi-trial `.npz`.
With `--cleanup-intermediate`, only the final CSV is kept.

Main options:

| Option | Default | Description |
|---|---|---|
| `--id-grf-mode {estimated,none}` | `estimated` | Estimate GRF from motion, or run inverse dynamics without external loads. |
| `--id-filter-mode {auto,walking,dynamic,none}` | `auto` | Low-pass preset: 12 Hz (`walking`), 30 Hz (`dynamic`), or chosen from the trial name (`auto`). |
| `--id-cutoff-hz HZ` | – | Explicit cutoff; overrides `--id-filter-mode`. |
| `--id-contact-height-threshold-m` | `0.04` | Contact height threshold above the support plane. |
| `--id-contact-speed-threshold-mps` | `0.5` | Contact speed threshold. |
| `--id-forces-to-exclude {Actuators,Muscles}` | – | Exclude a force set of the model from the reported joint torques, e.g. passive muscle forces. |
| `--id-horizontal-split-model PATH` | – | Calibrated horizontal force sharing ([Calibrated variant](#calibrated-variant)). |
| `--id-cop-progress-model PATH` | – | Calibrated center of pressure ([Calibrated variant](#calibrated-variant)). |

## Annotating AMASS

Download AMASS (SMPL-X G) from the [AMASS website](https://amass.is.tue.mpg.de/).
To process a whole folder on one machine:

```bash
python scripts/run_amass_batch_parallel.py \
  --input-root /path/to/AMASS \
  --output-dir outputs/amass_bsm \
  --workers 8 \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --id-cutoff-hz 12 \
  --id-forces-to-exclude Actuators \
  --cleanup-intermediate
```

The output tree mirrors the input tree, with one CSV per sequence, per-file
logs and a `batch_summary.json`. Existing outputs that passed the quality
gate with the current pipeline version are skipped, so an interrupted run can
be resumed with the same command.

On a SLURM cluster, `scripts/run_amass_batch_slurm.py submit` writes a task
manifest and submits array jobs with the same pipeline options; for very
large runs, `scripts/submit_amass_arrays_with_capacity.py` keeps the queue
within the cluster's job limit, and `scripts/watch_amass_oom_retries.py`
resubmits tasks that ran out of memory. See
[`docs/usage_reference.md`](docs/usage_reference.md) for all options.

## Calibrated variant

The motion determines the total force, but not how its horizontal part is
shared between the feet or where each foot force is applied. The calibrated
variant learns these two rules from force-plate data with linear models;
vertical forces and contact states are unchanged.

- **Horizontal force sharing.** For each foot, a linear model predicts the
  horizontal force from stance progress, contact activation, the estimated
  vertical force, the foot position and velocity relative to the center of
  mass (normalized by center-of-mass height) and the total horizontal force.
  Both feet share the coefficients, with the mediolateral axis mirrored for
  the right foot. In double support, the two predictions are corrected
  equally so that they sum to the required horizontal force; in single
  support, the loaded foot receives the full horizontal force.
- **Center of pressure.** Each foot force is applied at
  $\mathbf h + s(\mathbf t-\mathbf h)$, where $\mathbf h$ and $\mathbf t$ are
  the heel and toe body positions and $s\in[0,1]$ is a clipped linear
  function of foot pitch, stance progress and contact activation.

The two models have 25 coefficients in total. The fitted coefficients, from
least squares on all six GroundLink subjects, are in
[`model/groundlink_calibrated/`](model/groundlink_calibrated/):

```bash
python scripts/run_amass_to_bsm_csv.py \
  ... \
  --id-horizontal-split-model model/groundlink_calibrated/hsym_horizontal_model.json \
  --id-cop-progress-model model/groundlink_calibrated/cop_progress_model.json
```

Without these two options, the pipeline is training-free and its output is
unchanged. The calibrated variant is evaluated on GroundLink
leave-one-subject-out; use the training-free pipeline when no force-plate data
are available for calibration.

## Evaluation on GroundLink

We evaluate the estimated forces and torques on GroundLink (Han et al., SIGGRAPH Asia 2023): 206 trials from six subjects
(s001–s006), after excluding subject s007, motion classes with non-foot
support, and trials with misaligned motion and centers of pressure.
Reference torques come from OpenSim inverse dynamics with the measured
forces and centers of pressure, using the same model and kinematics.

The main scripts are:

| Step | Script |
|---|---|
| Audit force/motion pairs | [`scripts/validate_groundlink.py`](scripts/validate_groundlink.py) |
| Run the pipeline on all subjects and the measured-force inverse dynamics | [`scripts/rerun_groundlink_all.sh`](scripts/rerun_groundlink_all.sh), [`scripts/run_groundlink_e4.py`](scripts/run_groundlink_e4.py) |
| Extract reference and estimated torques | [`scripts/extract_groundlink_torque_targets.py`](scripts/extract_groundlink_torque_targets.py) |
| Compute the force and torque metrics | [`scripts/evaluate_groundlink_table1.py`](scripts/evaluate_groundlink_table1.py) |

The metrics are computed with

```bash
python scripts/evaluate_groundlink_table1.py \
  --torque-root /path/to/GroundLink_torque \
  --force-root /path/to/GroundLink/force \
  --output-dir outputs/groundlink_table1
```

Metrics are computed on valid frames during measured contact (vertical force
above 0.05 BW), with measured forces filtered at 12 Hz, averaged per foot,
trial and subject, and reported as the mean and 95% confidence interval over
the six subjects. The leave-one-subject-out fitting and evaluation of the
calibrated variant are in
[`scripts/metrics_tau_claude/`](scripts/metrics_tau_claude/README.md).

## Output format

| Columns | Content |
|---|---|
| `frame`, `time` | Frame index and time (s) |
| `subject_mass_kg`, `subject_height_m` | Estimated body mass and stature |
| `<body>_scale_{x,y,z}` | Scale factors of each model body |
| `<dof>`, `<dof>_vel`, `<dof>_acc` | Joint angle (rad) or translation (m), velocity, acceleration |
| `<dof>_tau` | Estimated joint torque (N m) or pelvis residual force (N) |
| GRF columns | Estimated force for each contact body and in total (N); zero without contact |
| `<body>_contact` | Binary contact state for each contact body |

`knee_angle_{l,r}_beta` are always undefined and are removed, so the CSV has
49 DOFs. The column schema is the same for every sequence, including flight
phases.

## Repository structure

```
scripts/run_amass_to_bsm_csv.py     single-sequence entry point
scripts/run_amass_batch_*.py        batch runners (local pool, SLURM)
scripts/*groundlink*                GroundLink validation and torque targets
scripts/metrics_tau_claude/         GroundLink evaluator and LOSO calibration study
src/opensim_batch_dynamics/         library code (loading, markers, IK, GRF, ID, export)
model/bsm/                          BSM OpenSim model and geometry
model/groundlink_calibrated/        coefficients of the calibrated variant
assets/smpl2ab/                     SMPL-X → BSM virtual-marker map
docs/                               detailed usage and dataset notes
tests/                              unit tests
```

Converters from NTU RGB+D, HumanML3D and CARE-PD to AMASS-like SMPL-X files,
a PDF report generator for the output CSVs, and a real-time prototype (`RT/`)
are also included. See
[`docs/usage_reference.md`](docs/usage_reference.md).

Run the tests with

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

## License and acknowledgements

The code is released under the [MIT License](LICENSE). SMPL-X, AMASS, SKEL
and GroundLink are distributed under their own licenses and must be obtained
from their authors. The pipeline builds on
[SMPL-X](https://smpl-x.is.tue.mpg.de/),
[AddBiomechanics](https://github.com/keenon/AddBiomechanics),
[SKEL](https://skel.is.tue.mpg.de/),
[OpenSim](https://opensim.stanford.edu/) and
[Nimble Physics](https://nimblephysics.org/).

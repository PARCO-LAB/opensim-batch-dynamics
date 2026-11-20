# Pipeline Description

This document explains, step by step, the pipeline currently implemented in this repository to convert an AMASS `SMPL-X` motion file into:

- a fitted OpenSim model (`.osim`) in BSM format,
- a kinematic motion file (`.mot`),
- a CSV with BSM joint positions, velocities, and accelerations,
- a CSV with estimated generalized torques,
- and the intermediate GRF/contact-wrench files used for inverse dynamics.

The main entrypoint is:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/A3-_Swing_arms_stageii.npz \
  --trial my_trial \
  --output-dir outputs/bsm \
  --bsm-model model/bsm/bsm.osim \
  --smplx-model-dir model/smpl \
  --addbio-root /path/to/AddBiomechanics
```

## 1. Input AMASS Parsing

The input file is an AMASS `.npz` sample with `surface_model_type='smplx'`.

The loader lives in [amass_loader.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/amass_loader.py).

It reads and validates:

- `trans`: global translation, shape `(T, 3)`
- `root_orient`: root orientation, shape `(T, 3)`
- `pose_body`: body pose, shape `(T, 63)`
- `pose_hand`: left/right hand pose, split into two `(T, 45)` tensors
- `pose_jaw`: jaw pose, shape `(T, 3)`
- `pose_eye`: split into left/right eye pose, each `(T, 3)`
- `betas`: body shape parameters
- `gender`
- `mocap_frame_rate`

The loader converts everything to `float32`, checks frame-count consistency, and stores the result in an `AMASSSequence` object.

## 2. SMPL-X Forward Pass

The forward pass is implemented in [smplx_forward.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/smplx_forward.py).

For each frame, the code builds the SMPL-X mesh and joints by feeding:

- global orientation,
- body pose,
- hand pose,
- jaw pose,
- eye pose,
- translation,
- repeated shape coefficients `betas`.

Important implementation details:

- If a sex-specific SMPL-X model is not available, the pipeline falls back to `SMPLX_NEUTRAL.npz`.
- The output is a dense surface mesh `vertices` with shape `(T, V, 3)` and `joints` with shape `(T, J, 3)`.
- These vertices are the geometric basis for the virtual-marker extraction.

## 3. Virtual Marker Extraction

The virtual markers are generated in [bsm_markers.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/bsm_markers.py).

### Marker map

The repository uses a BSM-specific map from marker name to SMPL-X vertex index:

- preferred source: `assets/smpl2ab/bsm_markers_smplx.yaml`
- fallback: an embedded default text mapping inside the code

Examples:

- `BLTI -> 3762`
- `BRTI -> 6520`
- `C7 -> 3832`

### Extraction algorithm

The implemented algorithm is intentionally simple and deterministic:

1. Load the ordered marker map `{marker_name: vertex_index}`.
2. For each marker, select the corresponding SMPL-X vertex.
3. Copy that vertex position for every frame.

In formula form:

```text
marker_position[t, m, :] = vertices[t, vertex_index(m), :]
```

This means:

- there is no interpolation between vertices,
- there is no local rigid-cluster fitting,
- there is no soft-tissue artifact model,
- each marker is a direct sample of one SMPL-X surface vertex.

The result is a marker tensor of shape `(T, M, 3)`.

## 4. TRC Export

The marker trajectories are exported to OpenSim TRC format in [trc_export.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/trc_export.py).

The writer:

- preserves the original AMASS frame rate,
- writes coordinates in meters,
- writes one row per frame,
- keeps the marker order defined by the marker map.

At this stage the pipeline has transformed:

```text
AMASS SMPL-X motion -> virtual marker trajectories -> markers.trc
```

## 5. Subject Metadata Estimation

Before running AddBiomechanics, the pipeline estimates subject-level metadata in [bsm_subject_json.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/bsm_subject_json.py).

### Height

Height is estimated from a neutral SMPL-X mesh as:

```text
height = max_y(vertices) - min_y(vertices)
```

### Mass

Mass is estimated in two possible ways:

1. Preferred method:
   compute SMPL-X mesh volume with `trimesh` and convert volume to mass using density `985 kg/m^3`.
2. Fallback method:
   use a height-based heuristic similar to a BMI prior:

```text
mass = clip(22 * height^2, 40, 140)
```

### Subject JSON

The generated `_subject.json` includes:

- `sex`
- `massKg`
- `heightM`
- `skeletonPreset="custom"`
- `disableDynamics=true`
- `runMoco=false`
- `segmentTrials=false`

The important point is that AddBiomechanics is used here primarily as a kinematic fitting engine, not as a full dynamics engine.

## 6. AddBiomechanics Subject Folder Construction

The AddBiomechanics folder layout is built in [addbio_subject_folder.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/addbio_subject_folder.py).

The subject folder contains:

- `_subject.json`
- `unscaled_generic.osim`
- `Geometry/`
- `trials/<trial>/markers.trc`

### Marker parent-frame canonicalization

One implementation detail is important here.

Some markers in `bsm.osim` use body names such as:

- `tibia_l`
- `tibia_r`

During AddBiomechanics/OpenSim scaling, bare names can fail to resolve correctly.  
To make the model robust, the pipeline rewrites marker parent sockets from:

```text
tibia_l
```

to:

```text
/bodyset/tibia_l
```

This is done automatically when copying the unscaled model into the subject folder.

## 7. Kinematic Fitting with AddBiomechanics

The local AddBiomechanics engine is launched by [addbio_runner.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/addbio_runner.py).

The runner:

- resolves the local AddBiomechanics checkout,
- executes `server/engine/src/engine.py`,
- forces the current Python environment into `PATH` so that subprocesses such as `opensim-cmd` are found,
- removes stale `_errors.json` files before each run,
- returns the fitted model and kinematic outputs.

### What AddBiomechanics is doing in this pipeline

Inside this repository, AddBiomechanics is treated as an external optimizer that takes:

- the unscaled BSM model,
- the TRC of virtual markers,
- subject metadata,

and returns:

- a fitted/scaled BSM model,
- an IK motion file,
- additional visualization and summary artifacts.

From the observed engine output, the kinematic stage includes:

1. trial loading,
2. marker-quality repair,
3. initialization and body-scale estimation,
4. marker-fitting / kinematic optimization,
5. final smoothing IK,
6. an acceleration-minimizing pass,
7. OpenSim result export.

The exact internal optimization mathematics live inside AddBiomechanics, not in this repo.  
This repo uses AddBiomechanics as the kinematic solver.

## 8. Export of BSM Positions, Velocities, and Accelerations

Once AddBiomechanics writes the final `.mot`, this repo converts it to a model-ordered CSV in [addbio_csv_export.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/addbio_csv_export.py).

### Position extraction

The code:

1. reads the final `.mot`,
2. extracts the DOF names directly from the final fitted `.osim`,
3. reorders the motion columns to match the exact DOF order of the model.

### Velocity and acceleration

Velocities and accelerations are currently computed numerically inside this repo with `np.gradient`.

For each DOF:

```text
qdot(t)  = d q(t) / dt
qddot(t) = d qdot(t) / dt
```

Important practical note:

- AddBiomechanics already performs an acceleration-minimizing smoothing pass before exporting the final motion.
- This means the `.mot` used here is already smoother than the raw fit.
- The repo then differentiates that smoothed motion numerically to obtain `vel` and `acc`.

The CSV therefore contains:

- `dof`
- `dof_vel`
- `dof_acc`

for every BSM coordinate.

## 9. Inverse Dynamics Overview

Inverse dynamics is implemented in [inverse_dynamics_no_grf.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/inverse_dynamics_no_grf.py).

There are two modes:

- `grf_mode="none"`: inverse dynamics with no external loads
- `grf_mode="estimated"`: inverse dynamics with explicit estimated contact wrench / GRF

The default in the current pipeline is:

```text
grf_mode = estimated
```

This is the mode used to produce the final torque CSV.

## 10. Preprocessing for Inverse Dynamics

Before inverse dynamics:

1. the fitted model is copied,
2. markers are stripped from the ID model,
3. the IK time window is inferred from the `.mot`,
4. the filtering cutoff is selected.

### Cutoff selection

The cutoff policy comes from `infer_cutoff_hz()`:

- `walking -> 12 Hz`
- `dynamic -> 30 Hz`
- `auto -> 12 Hz` for gait-like trial names, otherwise `30 Hz`
- `none -> no filtering`

For `A3-_Swing_arms_stageii`, the default selected cutoff is `30 Hz`.

## 11. Explicit GRF / Contact-Wrench Estimation

This is the core custom dynamics algorithm of the repo.

The implementation is in `_estimate_contact_wrenches_from_kinematics()` in [inverse_dynamics_no_grf.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/inverse_dynamics_no_grf.py#L187).

### 11.1 Model and motion loading

The code uses `nimblephysics` to load:

- the fitted OpenSim model,
- the IK motion.

It then reconstructs, frame by frame:

- whole-body center of mass,
- selected contact-body world positions.

By default the two contact bodies are:

- `calcn_l`
- `calcn_r`

If these are not available, the code tries fallback names such as `toes_l`, `toes_r`, `foot_l`, `foot_r`.

### 11.2 Signal smoothing

Before differentiating, the code applies a zero-phase 4th-order Butterworth low-pass filter to:

- COM trajectory,
- contact-body trajectories,
- later also the target external force.

This is done to reduce differentiation noise and keep the reconstructed forces physically smooth.

### 11.3 Floor estimate

The ground plane is estimated from the contact-body trajectories.

The code projects the filtered contact-body positions onto the gravity-up axis and uses:

```text
ground_height = percentile_1(heights_all)
```

So the floor is a robust low percentile of the observed foot-body heights.

### 11.4 Target total external force from COM dynamics

The global external force required by the motion is estimated from COM acceleration:

```text
F_target = m * (a_com - g)
```

where:

- `m` is the skeleton mass,
- `a_com` is the COM acceleration,
- `g` is gravity.

This is the total force that contact should provide at each frame.

### 11.5 Contact activation

For each contact body, the algorithm computes:

- height above the estimated floor,
- normal speed relative to the floor.

A contact is considered active when:

```text
height <= contact_height_threshold_m
and
abs(normal_speed) <= contact_speed_threshold_mps
```

Default thresholds:

- `contact_height_threshold_m = 0.06`
- `contact_speed_threshold_mps = 0.6`

Then the boolean contact signal is temporally expanded with a short convolution window to avoid frame-to-frame flicker.

### 11.6 Center of pressure proxy

For each active foot body, the code defines a COP proxy by projecting the body origin onto the estimated ground plane.

This is a simplified COP model:

- the COP is not estimated from pressure distribution,
- it is a geometric projection of the contact body position onto the floor.

### 11.7 Force sharing between left and right foot

For each frame:

1. compute the floor projection of COM,
2. compute the planar distance from COM projection to each active contact point,
3. assign weights inversely proportional to that distance:

```text
w_i = 1 / (distance_i + 1e-3)
```

4. normalize the weights so they sum to one.

The intuition is:

- if the COM projection is closer to one foot, that foot should carry more load.

### 11.8 Friction-cone projection

The initial force assigned to each foot is:

```text
F_i = w_i * F_target
```

This force is then projected onto a linearized Coulomb friction cone:

1. split into normal and tangential parts,
2. clamp the normal component to be non-negative,
3. enforce:

```text
||F_tangential|| <= mu * F_normal
```

where `mu` is the friction coefficient.

Default:

```text
mu = 0.8
```

### 11.9 Iterative force-balance correction

After the initial split, the sum of the two projected foot forces may not exactly equal `F_target`.

To fix this, the algorithm performs a few redistribution iterations:

1. compute the residual:

```text
delta = F_target - sum(F_i)
```

2. distribute `delta` to active contacts using the same weights,
3. re-project each updated foot force into the friction cone.

This keeps the forces physically admissible while recovering the desired net external load.

### 11.10 Contact moments

In the current implementation:

- explicit free moments are set to zero,
- the exported wrench uses force plus COP,
- no torsional foot-ground moment is estimated.

So the exported wrench is a simplified contact wrench:

- force vector,
- application point (COP proxy),
- zero moment vector.

### 11.11 Exported GRF artifacts

The repo exports:

- `*_grf_estimated.mot`
- `*_external_forces_estimated.xml`
- `*_contact_wrenches_estimated.csv`

The debug CSV includes:

- target total force,
- achieved total force,
- force-balance residual norm,
- per-foot COP,
- per-foot force,
- per-foot moment.

This makes it easy to inspect physical consistency.

## 12. Inverse Dynamics with OpenSim

After estimating the contact wrench, the pipeline runs OpenSim inverse dynamics with:

- fitted model without markers,
- final IK motion,
- generated external loads XML,
- selected low-pass cutoff.

This is handled by `run_inverse_dynamics_with_estimated_grf()` in [inverse_dynamics_no_grf.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/inverse_dynamics_no_grf.py#L550).

The output is an OpenSim `.sto` containing generalized forces.

Interpretation:

- rotational coordinates -> moments in `N*m`
- translational coordinates -> generalized forces in `N`

## 13. Torque CSV Export

The final torque CSV is produced by `export_torque_csv_from_id_sto()` in [inverse_dynamics_no_grf.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/src/opensim_batch_dynamics/inverse_dynamics_no_grf.py#L634).

The algorithm:

1. read the ID `.sto`,
2. extract model DOF names from the fitted `.osim`,
3. map each DOF to the corresponding ID output column,
4. interpolate the ID output onto the IK timestamps,
5. export one torque column per DOF:

```text
<dof>_tau
```

This keeps the kinematic CSV and torque CSV aligned frame by frame.

## 14. End-to-End Data Flow

The implemented pipeline can be summarized as:

```text
AMASS npz
-> validated SMPL-X fields
-> SMPL-X forward pass
-> virtual markers from selected SMPL-X vertices
-> TRC
-> AddBiomechanics custom-skeleton fit
-> fitted BSM model + IK motion
-> DOF CSV (q, qdot, qddot)
-> explicit GRF/contact-wrench estimation from COM dynamics
-> OpenSim inverse dynamics with generated ExternalLoads
-> torque CSV
```

## 15. What Is Custom in This Repo vs Delegated

### Custom logic implemented in this repo

- AMASS parsing
- SMPL-X forward orchestration
- BSM virtual-marker extraction
- TRC generation
- subject metadata estimation
- AddBiomechanics subject folder preparation
- marker parent-frame normalization
- CSV export for BSM DOFs
- explicit GRF/contact-wrench estimation
- OpenSim inverse dynamics orchestration
- torque CSV export

### Delegated to external tools

- SMPL-X mesh generation: `smplx`
- kinematic fitting and scaling: AddBiomechanics
- inverse dynamics solve: OpenSim
- OpenSim model parsing / kinematic replay support: `nimblephysics`

## 16. Practical Limitations

The current implementation is robust and fully operational, but it is still a simplified physics pipeline in some respects:

- virtual markers are single-vertex markers, not clustered or averaged,
- GRFs are inferred from kinematics only,
- COP is approximated from projected foot-body origins,
- contact moments are set to zero,
- there is no explicit contact optimization against measured force plates,
- some model DOFs that are present in the `.osim` but absent in the motion may remain `NaN` in the kinematic CSV.

Despite these simplifications, the implemented pipeline is internally consistent and produces:

- smooth kinematics,
- explicit external loads,
- and physically interpretable generalized torques.

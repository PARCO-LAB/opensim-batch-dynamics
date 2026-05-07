# Codex Repository Context

This file is a persistent technical handoff for future Codex sessions working on
this repository. It summarizes the repository structure, the implemented offline
and real-time pipelines, important variables and dimensions, known limitations,
and the current scientific framing.

Repository root:

```text
/Users/enricomartini/Desktop/opensim-batch-dynamics
```

## High-Level Goal

This repository converts AMASS / SMPL-X human motion sequences into OpenSim BSM
dynamics datasets and provides a separate causal real-time inverse-dynamics
prototype.

The main downstream research question is:

```text
Does biomechanical dynamics improve self-supervised human motion representation
learning compared with kinematics alone?
```

The generated data are intended to support training a JEPA-style foundation
model over human motion, where one can compare representations trained with:

- kinematics only: generalized coordinates, velocities, accelerations;
- kinematics plus dynamics: torques, GRFs, contact states, and possibly residuals;
- dynamics-aware objectives or auxiliary prediction targets.

## Main Entrypoints

Offline AMASS-to-OpenSim/BSM CSV pipeline:

```text
scripts/run_amass_to_bsm_csv.py
```

Batch runners:

```text
scripts/run_amass_batch_parallel.py
scripts/run_amass_batch_slurm.py
```

Real-time causal estimator:

```text
RT/rt_library.py
RT/real_time_test.py
```

Report and visualization helpers:

```text
scripts/csv_explorer.py
scripts/realtime_vs_offline_pdf.py
```

Existing documentation:

```text
README.md
DESCRIPTION.md
RT/notion_21_04_realtime_algorithm.txt
RT/RESULTS.md
RT/NEXT_STEPS.md
RT/AMASS_GENERALIZATION.md
```

## Required Assets

Core assets:

```text
model/smpl/SMPLX_NEUTRAL.npz
model/bsm/bsm.osim
model/bsm/Geometry/
assets/smpl2ab/bsm_markers_smplx.yaml
```

External local dependency:

```text
AddBiomechanics checkout, passed via --addbio-root or ADDBIO_ENGINE_ROOT
```

Python / Conda environment:

```text
environment.yml
```

Important packages used by the code:

- `numpy`, `scipy`, `pandas`;
- `torch`, `smplx`;
- `trimesh`;
- `opensim`;
- `nimblephysics`;
- `cvxpy`, `OSQP`, optionally `SCS`;
- local AddBiomechanics engine.

On macOS, the pipeline sets:

```text
KMP_DUPLICATE_LIB_OK=TRUE
```

to avoid OpenMP runtime conflicts across Torch, OpenSim, and Nimble.

## Repository Structure

Core package:

```text
src/opensim_batch_dynamics/
```

Important modules:

- `amass_loader.py`: AMASS `.npz` parsing, validation, multi-trial support.
- `smplx_forward.py`: SMPL-X forward pass through external `smplx`.
- `bsm_markers.py`: BSM virtual marker extraction from SMPL-X vertices.
- `trc_export.py`: OpenSim TRC writer.
- `bsm_subject_json.py`: subject height/mass estimation and AddBiomechanics JSON.
- `addbio_subject_folder.py`: AddBiomechanics custom subject folder builder.
- `addbio_runner.py`: local AddBiomechanics engine launcher.
- `addbio_csv_export.py`: fitted `.mot` to model-ordered DOF CSV.
- `mot_to_csv.py`: OpenSim `.mot` parsing, coordinate extraction, filtering helpers.
- `inverse_dynamics_no_grf.py`: offline estimated-GRF and no-GRF inverse dynamics.
- `final_csv_export.py`: unified final CSV export.
- `bsm_assets.py`, `config.py`: asset path helpers.

Real-time subsystem:

```text
RT/
```

Important files:

- `RT/rt_library.py`: causal two-stage QP estimator, metrics helpers.
- `RT/real_time_test.py`: offline-vs-real-time driver and benchmark script.
- `RT/RESULTS.md`: latest AMASS batch results.
- `RT/NEXT_STEPS.md`: tuning history, rejected experiments, open failures.
- `RT/AMASS_GENERALIZATION.md`: older generalization report.

## Offline Pipeline Summary

The offline command usually looks like:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/A3-_Swing_arms_stageii.npz \
  --trial A3_swing_full \
  --output-dir outputs/bsm \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

The data flow is:

```text
AMASS npz
-> validated SMPL-X fields
-> SMPL-X forward pass
-> fixed-vertex BSM virtual markers
-> TRC marker file
-> AddBiomechanics custom-skeleton fit
-> fitted BSM OpenSim model + IK .mot
-> q, dq, ddq CSV
-> kinematics-derived contact wrench / GRF estimate
-> OpenSim inverse dynamics
-> torque CSV
-> unified final CSV
```

### AMASS Parsing

Implemented in:

```text
src/opensim_batch_dynamics/amass_loader.py
```

Supported layouts:

- Stage-II split SMPL-X fields;
- legacy AMASS `poses/trans` layout;
- multi-trial `.npz` with prefixed keys such as `<trial>/trans`;
- multi-trial object container under `trials`;
- optional sibling `shape.npz` fallback for `gender` and `betas`.

Key loaded fields:

- `trans`: shape `(T, 3)`;
- `root_orient`: shape `(T, 3)`;
- `pose_body`: shape `(T, 63)`;
- `pose_hand`: shape `(T, >=90)`, split into left/right `(T, 45)`;
- `pose_jaw`: shape `(T, 3)`;
- `pose_eye`: shape `(T, >=6)`, split into left/right `(T, 3)`;
- `betas`: at least 10 shape coefficients;
- `gender`;
- `mocap_frame_rate` or `mocap_framerate`.

The loader normalizes arrays to `float32`, validates frame counts, and returns an
`AMASSSequence`.

### SMPL-X Forward Pass

Implemented in:

```text
src/opensim_batch_dynamics/smplx_forward.py
```

For each frame:

```text
(vertices, joints) = SMPL-X(root_orient, body_pose, hands, jaw, eyes, transl, betas)
```

Output:

- `vertices`: shape `(T, V, 3)`;
- `joints`: shape `(T, J, 3)`.

The code uses up to 16 betas. If sex-specific SMPL-X files are unavailable, it
falls back to `SMPLX_NEUTRAL.npz`.

### Virtual BSM Marker Extraction

Implemented in:

```text
src/opensim_batch_dynamics/bsm_markers.py
assets/smpl2ab/bsm_markers_smplx.yaml
```

The current marker map has:

```text
M = 105 virtual markers
```

Each marker is a direct sample of one SMPL-X vertex:

```latex
Y_{k,m} = V_{k,\pi(m)}
```

There is no interpolation, local rigid cluster fitting, learned marker regressor,
or soft-tissue artifact model.

### TRC Export

Implemented in:

```text
src/opensim_batch_dynamics/trc_export.py
```

The TRC writer:

- preserves the AMASS frame rate;
- writes coordinates in meters;
- writes one row per frame;
- keeps marker order from the YAML map;
- supports optional axis rotations and vertical offset, though the main BSM path
  writes raw marker positions.

### Subject Metadata Estimation

Implemented in:

```text
src/opensim_batch_dynamics/bsm_subject_json.py
```

Height is estimated from a neutral SMPL-X mesh:

```latex
h = \max_v V^0_{v,y} - \min_v V^0_{v,y}
```

Mass uses mesh volume when possible:

```latex
m = 985 \, |\operatorname{Vol}(V^0, F)|
```

Fallback mass:

```latex
m = \operatorname{clip}(22 h^2, 40, 140)
```

The generated `_subject.json` includes:

- `sex`;
- `massKg`;
- `heightM`;
- `subjectTags = ["AMASS", "SMPL-X", "BSM"]`;
- `skeletonPreset = "custom"`;
- `disableDynamics = true`;
- `runMoco = false`;
- `segmentTrials = false`.

This means AddBiomechanics is used here mainly for scaling and kinematic
fitting, not for the final custom dynamics export.

### AddBiomechanics Subject Folder

Implemented in:

```text
src/opensim_batch_dynamics/addbio_subject_folder.py
```

Generated layout:

```text
<trial_root>/
  _subject.json
  unscaled_generic.osim
  Geometry/
  trials/<trial>/markers.trc
```

The code canonicalizes marker parent frame sockets in the copied `.osim`, e.g.:

```text
tibia_l -> /bodyset/tibia_l
```

This avoids body socket resolution failures during AddBiomechanics/OpenSim
processing.

### AddBiomechanics Run

Implemented in:

```text
src/opensim_batch_dynamics/addbio_runner.py
```

The launcher resolves:

```text
<AddBiomechanics>/server/engine/src/engine.py
```

and runs it with the active Python environment on `PATH`, so subprocesses such
as `opensim-cmd` resolve correctly.

Expected key outputs:

```text
results/Models/match_markers_but_ignore_physics.osim
results/IK/<subject>_ik.mot
```

The exact AddBiomechanics optimization is delegated to AddBiomechanics. In paper
notation, it can be described abstractly as:

```latex
(\hat{\phi}, \hat{q}_{0:T-1}) =
\arg\min_{\phi, q_{0:T-1}}
\sum_{k=0}^{T-1}\sum_{m=1}^{105}
\rho(\|P_m(q_k;\phi) - Y_{k,m}\|_2^2)
+ R_{AB}(\phi, q_{0:T-1})
```

where `phi` includes subject/model fit parameters and `P_m` is an OpenSim marker
position function. Do not claim this is the exact internal AddBiomechanics
objective unless verified from AddBiomechanics itself.

### Kinematic CSV Export

Implemented in:

```text
src/opensim_batch_dynamics/addbio_csv_export.py
src/opensim_batch_dynamics/mot_to_csv.py
```

The fitted `.osim` defines model coordinate order. The fitted `.mot` provides
positions. The exporter computes:

```latex
\dot{q}_{k,i} = D_t(q_{0:T-1,i})_k
\qquad
\ddot{q}_{k,i} = D_t(\dot{q}_{0:T-1,i})_k
```

using `numpy.gradient`.

Important detail:

- The BSM `.osim` has 51 unlocked coordinates.
- The final CSV removes `knee_angle_r_beta` and `knee_angle_l_beta` because they
  are always NaN.
- Therefore the final CSV has 49 effective DOFs.

Current full coordinate list in `model/bsm/bsm.osim`:

```text
0  pelvis_tilt
1  pelvis_list
2  pelvis_rotation
3  pelvis_tx
4  pelvis_ty
5  pelvis_tz
6  hip_flexion_r
7  hip_adduction_r
8  hip_rotation_r
9  knee_angle_r
10 knee_angle_r_beta
11 ankle_angle_r
12 subtalar_angle_r
13 mtp_angle_r
14 hip_flexion_l
15 hip_adduction_l
16 hip_rotation_l
17 knee_angle_l
18 knee_angle_l_beta
19 ankle_angle_l
20 subtalar_angle_l
21 mtp_angle_l
22 lumbar_bending
23 lumbar_extension
24 lumbar_twist
25 thorax_bending
26 thorax_extension
27 thorax_twist
28 head_bending
29 head_extension
30 head_twist
31 scapula_abduction_r
32 scapula_elevation_r
33 scapula_upward_rot_r
34 scapula_abduction_l
35 scapula_elevation_l
36 scapula_upward_rot_l
37 shoulder_r_x
38 shoulder_r_y
39 shoulder_r_z
40 shoulder_l_x
41 shoulder_l_y
42 shoulder_l_z
43 elbow_flexion_r
44 elbow_flexion_l
45 pro_sup_r
46 pro_sup_l
47 wrist_flexion_r
48 wrist_deviation_r
49 wrist_flexion_l
50 wrist_deviation_l
```

Final effective DOFs are all of the above except indices 10 and 18.

### Offline Contact / GRF Estimation

Implemented in:

```text
src/opensim_batch_dynamics/inverse_dynamics_no_grf.py
```

Core function:

```text
_estimate_contact_wrenches_from_kinematics()
```

The code uses NimblePhysics to load the fitted OpenSim model and IK motion.

Definitions:

- `q_k`: fitted model positions at frame `k`;
- `c_k`: whole-body center of mass;
- `b_{i,k}`: world position of contact-body `i`;
- `g`: model gravity;
- `u = -g / ||g||`: upward normal;
- `m`: skeleton mass;
- `F*_k`: target support force.

COM and body trajectories are low-pass filtered with a 4th-order zero-phase
Butterworth filter when a cutoff is active.

Ground height:

```latex
h_g = \operatorname{percentile}_{1}(\{u^\top b_{i,k}\}_{i,k})
```

COM-derived target support force:

```latex
F^*_k = m(\ddot{c}_k - g)
```

Contact activation:

```latex
a_{i,k} =
\mathbb{1}[
(u^\top b_{i,k} - h_g) \le 0.06
\land
|D_t(u^\top b_i)_k| \le 0.6]
```

Then the binary active mask is temporally expanded by approximately 40 ms to
reduce frame-to-frame flicker.

COP proxy:

```latex
p_{i,k} = b_{i,k} - (u^\top b_{i,k} - h_g)u
```

Force-sharing weights:

```latex
d_{i,k} = \|(I - uu^\top)(p_{i,k} - p_{com,k})\|_2
```

```latex
\alpha_{i,k} =
\frac{(d_{i,k}+10^{-3})^{-1}}
{\sum_{j\in A_k}(d_{j,k}+10^{-3})^{-1}}
```

Initial force:

```latex
f^0_{i,k} = \alpha_{i,k} F^*_k
```

Coulomb cone:

```latex
\mathcal{K}_{\mu} =
\{f \in \mathbb{R}^3 :
f_n = u^\top f \ge 0,\;
\|f - f_n u\|_2 \le \mu f_n\}
```

Default friction:

```text
mu = 0.8
```

The implementation projects each force into the cone and then performs three
project-distribute correction iterations:

```latex
\Delta F_k = F^*_k - \sum_i f_{i,k}
```

```latex
f_{i,k} \leftarrow
\Pi_{\mathcal{K}_{\mu}}(f_{i,k} + \alpha_{i,k}\Delta F_k)
```

Offline free contact moments are currently zero:

```latex
w_{i,k} = [f_{i,k}; 0_3]
```

Generated GRF artifacts:

```text
*_grf_estimated.mot
*_external_forces_estimated.xml
*_contact_wrenches_estimated.csv
```

The debug CSV includes:

- target force;
- achieved force;
- force-balance residual norm;
- per-contact COP;
- per-contact force;
- per-contact moment, currently zero.

### Offline OpenSim Inverse Dynamics

Implemented in:

```text
src/opensim_batch_dynamics/inverse_dynamics_no_grf.py
```

Modes:

```text
grf_mode = "estimated"  # default
grf_mode = "none"
```

The ID model is a marker-free copy of the fitted model to avoid marker socket
issues during inverse dynamics.

Filtering cutoff policy:

- `walking`: 12 Hz;
- `dynamic`: 30 Hz;
- `auto`: 12 Hz for gait-like trial names, 30 Hz otherwise;
- `none`: no filtering.

OpenSim inverse dynamics can be summarized as:

```latex
M(q_k)\ddot{q}_k + h(q_k,\dot{q}_k)
= \tau_k + J_{ext}(q_k)^\top w_k
```

or:

```latex
\tau_k =
M(q_k)\ddot{q}_k + h(q_k,\dot{q}_k)
- J_{ext}(q_k)^\top w_k
```

Rotational coordinates produce moments in N m. Translational coordinates produce
generalized forces in N.

Torque export maps ID `.sto` labels back to model coordinate names and
interpolates the result onto IK timestamps.

### Final Offline CSV

Implemented in:

```text
src/opensim_batch_dynamics/final_csv_export.py
```

Final columns:

- `frame`, `time`;
- `subject_mass_kg`, `subject_height_m`;
- for each of 26 bodies: `<body>_scale_x`, `<body>_scale_y`, `<body>_scale_z`;
- for each of 49 effective DOFs: `<dof>`, `<dof>_vel`, `<dof>_acc`, `<dof>_tau`;
- for each exported contact body: `<body>_grf_x`, `<body>_grf_y`, `<body>_grf_z`, `<body>_contact`;
- `grf_total_x`, `grf_total_y`, `grf_total_z`.

The 26 BSM bodies are:

```text
pelvis, femur_r, tibia_r, patella_r, talus_r, calcn_r, toes_r,
femur_l, tibia_l, patella_l, talus_l, calcn_l, toes_l, head,
lumbar_body, thorax, scapula_r, scapula_l, humerus_r, humerus_l,
ulna_r, ulna_l, radius_r, radius_l, hand_r, hand_l
```

Current unified AMASS CSVs show 24 contact bodies:

```text
calcn_l, calcn_r, femur_l, femur_r, hand_l, hand_r, head,
humerus_l, humerus_r, lumbar_body, pelvis, radius_l, radius_r,
scapula_l, scapula_r, talus_l, talus_r, thorax, tibia_l, tibia_r,
toes_l, toes_r, ulna_l, ulna_r
```

Patella bodies are scaled but not present as exported contact bodies in current
CSV examples.

Contact code:

```latex
contact_{i,k} = \mathbb{1}[\|f_{i,k}\|_2 > 10^{-6}]
```

If GRF/contact artifacts are missing, the exporter falls back to safe zero GRF
columns for fallback contact bodies, usually `calcn_l` and `calcn_r`.

## Real-Time Pipeline Summary

The real-time estimator is in:

```text
RT/rt_library.py
```

Main function:

```text
qpid()
```

The evaluation driver is:

```text
RT/real_time_test.py
```

The solver is causal and weakly coupled:

```text
Stage 1: robust kinematic estimation and causal derivative filtering
Stage 2: dynamic/contact QP over ddq, foot wrenches, and root residual
```

Important weak-coupling detail:

```text
q_t  = q_kin,t
dq_t = dq_kin,t
ddq_t comes from Stage 2 QP
```

Stage 2 explains the Stage 1 pose dynamically; it does not replace the pose
with an independently integrated dynamic pose.

### Observed Joints

`RT/real_time_test.py` uses 12 observed joints:

```text
walker_knee_r
wrist_l
hip_r
GlenoHumeral_r
elbow_l
hip_l
elbow_r
wrist_r
walker_knee_l
GlenoHumeral_l
ankle_r
ankle_l
```

The measured input is:

```text
x_t: shape (12, 3)
```

It may contain NaN rows under dropout. Optional measurement weights/confidences
can be supplied.

### Real-Time State Variables

Full BSM dimension:

```text
n = 51
```

Actuated dimension:

```text
n_a = n - 6 = 45
```

The first six coordinates are the floating pelvis/root coordinates.

Persistent state from `initialize_rt_state()`:

- `q`: shape `(51,)`;
- `dq`: shape `(51,)`;
- `ddq`: shape `(51,)`;
- `q_kin`: shape `(51,)`;
- `dq_kin`: shape `(51,)`;
- `ddq_kin`: shape `(51,)`;
- `tau`: shape `(45,)`;
- `tau_full`: shape `(51,)`;
- `root_residual`: shape `(6,)`;
- `foot_forces["left"]`, `foot_forces["right"]`: each shape `(3,)`;
- `foot_wrenches["left"]`, `foot_wrenches["right"]`: each shape `(6,)`;
- `contact_state["left"]`, `contact_state["right"]`: booleans;
- `contact_prob["left"]`, `contact_prob["right"]`: floats in `[0, 1]`;
- `floor_height`: scalar;
- `step_index`: integer.

### Stage 1: Causal Robust IK

Stage 1 starts from a constant-acceleration prior:

```latex
q_{pred,t} =
q_{t-1} + \Delta t \dot{q}_{t-1}
+ \frac{1}{2}\Delta t^2 \ddot{q}_{t-1}
```

```latex
\dot{q}_{pred,t} =
\dot{q}_{t-1} + \Delta t \ddot{q}_{t-1}
```

The code clips `q_pred` to model position limits and uses per-DOF velocity caps:

- most DOFs: 25;
- first 3 coordinates: 12;
- coordinates 3 to 5: 8.

Root recentering:

- It computes the mean task-space residual over valid observed joints.
- It directly shifts root translation coordinates `q[3:6]`.
- This is important to prevent pelvis/root drift with only 12 global keypoints.

Stage 1 solves a QP over incremental position correction `Delta q`:

```latex
\min_{\Delta q}
\|W_x(J_x\Delta q - e_x)\|_2^2
+ \|W_g(J_g\Delta q - e_g)\|_2^2
+ \|W_s\Delta q\|_2^2
+ \|W_p(\Delta q - (q_{pred}-q))\|_2^2
+ \|W_m(\Delta q - (q_{meas}-q))\|_2^2
```

The optional `q_meas` term is only included if direct coordinate measurements
are provided.

Constraints:

```latex
q_{min} \le q + \Delta q \le q_{max}
```

```latex
-\Delta t \dot{q}_{cap}
\le \Delta q_{cum} + \Delta q
\le \Delta t \dot{q}_{cap}
```

Stage 1 QP details:

- maximum linearization iterations: `STAGE1_MAX_ITERS = 5`;
- primary solver: OSQP;
- fallback solver: SCS;
- geometry row budget: `STAGE1_GEOM_ROWS = 36`.

Robust measurement weights:

```latex
r_j = \frac{1}{1 + (\|e_j\|_2/0.05)^2}
```

The current task-space measurement weight is approximately:

```text
30 * joint_weight * robust_weight
```

Stage 1 geometric priors are built from observed segment structure:

- hip span;
- shoulder span;
- trunk axis from hip center to shoulder center;
- knee span;
- upper-arm vectors;
- forearm vectors;
- thigh vectors;
- shank vectors.

Weakly observed DOF masks:

- pelvis orientation: `pelvis_tilt`, `pelvis_list`, `pelvis_rotation`;
- trunk: `lumbar_*`, `thorax_*`;
- scapula: `scapula_*`;
- shoulder: `shoulder_*`.

The code applies stronger priors and smoothing to weakly observed DOFs.

DOF reliability:

```latex
\rho_i =
\frac{\sum_l (w_l J_{l,i})^2}
{\max_i \sum_l (w_l J_{l,i})^2}
```

Velocity blend:

```latex
\dot{q}_{hat,t} =
\rho_t \odot \frac{q_{hat,t}-q_{t-1}}{\Delta t}
+ (1-\rho_t)\odot \dot{q}_{pred,t}
```

Raw acceleration:

```latex
\ddot{q}_{hat,t} =
\frac{\dot{q}_{hat,t} - \dot{q}_{t-1}}{\Delta t}
```

### Stage 1 Causal Kinematic Filter

Enabled by default through `use_stage1_kin_filter=True`.

Prediction:

```latex
q_{kin,pred} =
q_{kin,t-1} + \Delta t \dot{q}_{kin,t-1}
+ \frac{1}{2}\Delta t^2 \ddot{q}_{kin,t-1}
```

```latex
\dot{q}_{kin,pred} =
\dot{q}_{kin,t-1} + \Delta t \ddot{q}_{kin,t-1}
```

Residual:

```latex
\epsilon_t = q_{hat,t} - q_{kin,pred}
```

Update:

```latex
q_{kin,t} = q_{hat,t}
```

```latex
\dot{q}_{kin,t} =
\dot{q}_{kin,pred} + \frac{\beta_t}{\Delta t}\epsilon_t
```

```latex
\ddot{q}_{kin,t} =
\ddot{q}_{kin,t-1}
+ \frac{2\gamma_t}{\Delta t^2}\epsilon_t
```

The beta/gamma gains are DOF-wise and reliability-aware. They are reduced for
weakly observed pelvis orientation, trunk, scapula, and shoulder DOFs to avoid
noisy accelerations and torque spikes.

Important constants in `RT/rt_library.py`:

```text
KIN_BETA_POS = 0.18
KIN_BETA_ROOT_ROT = 0.10
KIN_BETA_JOINT = 0.06
KIN_GAMMA_POS = 0.020
KIN_GAMMA_ROOT_ROT = 0.010
KIN_GAMMA_JOINT = 0.006
```

### Stage 1 Contact Cues

Foot bodies:

```text
left:  calcn_l, toes_l
right: calcn_r, toes_r
```

For each side, the code computes:

- heel and toe world positions;
- heel and toe velocities;
- heel/toe Jacobians and bias terms;
- foot anchor point;
- foot basis: forward, lateral, up;
- COP half-length and half-width;
- torsion radius;
- ankle joint confidence;
- previous load ratio;
- height above causal floor;
- vertical foot velocity;
- contact score;
- contact probability.

Gravity defines up:

```latex
u = -g/\|g\|
```

COM-derived support force:

```latex
F^s_t = m(a_{COM,t} - g)
```

The support prior is transformed into the ground basis, step-limited from
previous foot forces, clamped to nonnegative vertical support, capped at 3.5
body weights, and projected to the friction cone.

Support ratio:

```latex
\sigma_t = \frac{u^\top F^s_t}{m\|g\|}
```

Causal floor estimate:

```latex
h_{floor,t} =
\begin{cases}
0.7h_{floor,t-1}+0.3h_{cand,t}, & h_{cand,t}<h_{floor,t-1} \\
h_{floor,t-1}+0.02(h_{cand,t}-h_{floor,t-1}), & h_{cand,t}\ge h_{floor,t-1}
\end{cases}
```

Contact score:

```latex
s =
0.40s_{joint}
+0.30s_{height}
+0.15s_{vel}
+0.10c_{prev}
+0.05s_{support}
```

Contact probability:

```latex
p =
0.50s
+0.20s_{height}
+0.10s_{vel}
+0.12p_{prev}
+0.08s_{support}
```

Hysteresis thresholds:

```text
CONTACT_ON_SCORE = 0.52
CONTACT_OFF_SCORE = 0.34
```

If a foot was already active, it remains active when:

```text
score >= 0.34 and height <= 0.18 m
```

If it was not active, it enters contact when:

```text
score >= 0.52 and height <= 0.14 m
```

There are fallback rules:

- if support is clearly nonzero and no foot is active, activate the best foot;
- if support is near zero, weak contacts are turned off;
- quasi-static support has additional pruning heuristics based on previous load
  ratio, height, vertical velocity, and support dominance.

### Stage 2: Dynamic Contact QP

Stage 2 decision variables:

```text
ddq_var: active generalized acceleration, shape (n_active,)
wrench_var: left/right foot wrench, shape (12,)
root_var: root residual wrench, shape (6,)
```

The 12D foot wrench is:

```text
[Fx_L, Fy_L, Fz_L, Mx_L, My_L, Mz_L,
 Fx_R, Fy_R, Fz_R, Mx_R, My_R, Mz_R]
```

The code builds the multibody dynamics:

```latex
M(q_{kin})\ddot{q} + h(q_{kin},\dot{q}_{kin})
= \tau + J_w(q_{kin})^\top w
```

Root rows are constrained with a residual:

```latex
M_r\ddot{q} + h_r = J_r w + Rr
```

Actuated torque is reconstructed after solving:

```latex
\tau_a(\ddot{q}, w) =
M_a\ddot{q} + h_a - J_a w
```

The QP does not directly optimize `tau` as a primary variable anymore. This was
an intentional design change because direct `tau` optimization was too
permissive and noisy.

Stage 2 objective includes:

- discrete pose consistency with Stage 1 target;
- discrete velocity consistency with Stage 1 target;
- acceleration prior toward Stage 1 filtered acceleration;
- explicit `ddq_t - ddq_{t-1}` smoothing;
- torque smoothness and torque norm penalties via affine `tau_a`;
- wrench smoothness and wrench norm penalties;
- root residual smoothness and norm penalties;
- net support force prior from COM dynamics;
- left/right support split prior;
- task-space measurement consistency term on integrated pose;
- foot anchor acceleration/velocity and angular acceleration/velocity soft
  constraints scaled by contact probability.

Compact objective:

```latex
\min_{\ddot{q},w,r}
\|A_q\ddot{q}-b_q\|^2
+\|A_{\dot{q}}\ddot{q}-b_{\dot{q}}\|^2
+\|W_{\ddot{q}}(\ddot{q}-\ddot{q}_{target})\|^2
+\|W_s(\ddot{q}-\ddot{q}_{prev})\|^2
```

```latex
+\|W_{\tau}(\tau_a-\tau_{a,t-1})\|^2
+\|W_{\tau0}\tau_a\|^2
+\|W_w(w-w_{t-1})\|^2
+\|W_{w0}w\|^2
```

```latex
+\|W_{net}(Ew-F^s_t)\|^2
+\|W_{split}(w-w^s_t)\|^2
+\|W_r(r-r_{t-1})\|^2
+\|W_{r0}r\|^2
+\|W_x(J_x\ddot{q}-b_x)\|^2
```

Acceleration bounds combine:

- position-limit feasibility;
- velocity caps;
- absolute acceleration caps.

Typical absolute caps:

```text
most DOFs: 90
first 3 coordinates: 50
coordinates 3 to 5: 40
```

Foot wrench constraints use a linearized friction and COP model:

```latex
f_z \ge 0
```

```latex
|f_x| \le \mu f_z,\qquad |f_y| \le \mu f_z
```

```latex
|M_x| \le b f_z,\qquad |M_y| \le l f_z,\qquad |M_z| \le r_{torsion} f_z
```

Constants:

```text
FOOT_COP_HALF_LENGTH = 0.11
FOOT_COP_HALF_WIDTH = 0.045
FOOT_TORSION_RADIUS = 0.07
```

Stage 2 solver:

- primary solver: OSQP;
- fallback solver: SCS;
- max OSQP iterations: `STAGE2_MAX_OSQP_ITERS = 5000`;
- cached fixed-shape QPs in `DYN_QP_CACHE`.

After solving:

```latex
\tau_{full,t} = [r_t; \tau_{a,t}]
```

The final real-time state stores:

```text
q = q_kin
dq = dq_kin
ddq = ddq_full from Stage 2
```

The code also computes diagnostic `q_dyn` and `dq_dyn`, but these are not the
main returned state.

Dynamic residual diagnostic:

```latex
e_{dyn,t} =
M(q_t)\ddot{q}_t + h(q_t,\dot{q}_t)
- \tau_{full,t}
- J_w(q_t)^\top w_t
```

```latex
\eta_t = \|e_{dyn,t}\|_2
```

This is saved as:

```text
dynamics_residual_norm
```

### Real-Time CSV Output

When `RT/real_time_test.py --output-csv` is used, the output includes:

- `frame`, `time`;
- subject metadata and body scales copied from offline CSV;
- all 51 model DOFs with `<dof>`, `<dof>_vel`, `<dof>_acc`, `<dof>_tau`;
- left/right GRFs;
- total GRF;
- left/right contact flags;
- left/right 6D wrenches;
- `mpjpe_m`;
- `dynamics_residual_norm`;
- `solve_time_ms`;
- stress-test metadata: noise, dropout, friction, Stage 1 filter flag.

### Real-Time Metrics

`RT/real_time_test.py` reports:

- MPJPE over the 12 observed joints;
- q RMSE / MAE;
- dq RMSE / MAE;
- ddq RMSE / MAE;
- tau full RMSE / MAE;
- actuated tau RMSE / MAE;
- actuated tau jerk RMSE;
- actuated tau jerk L2 mean;
- left/right GRF RMSE;
- left/right contact accuracy, precision, recall, F1;
- worst q DOFs;
- worst tau DOFs;
- solve time mean and p95.

Metric mask excludes prefixes:

```text
ankle_angle_
subtalar_angle_
head_
wrist_
pro_sup_
```

This mask is applied to precision metrics, not necessarily to all plots.

## Latest Real-Time Results Context

From `RT/RESULTS.md`, full AMASS clean evaluation over 25 original offline CSV
files:

```text
Total sequence frames: 24786
Weighted MPJPE mean: 0.004694 m
Weighted q RMSE: 0.536112
Weighted actuated tau RMSE: 265.279520
Weighted tau jerk RMSE: 94.123808
Left contact F1: 0.9557
Right contact F1: 0.9431
Mean solve time: 20.16 ms/frame
```

Noise stress test:

```text
--noise-std 0.01
Weighted MPJPE mean: 0.018019 m
Weighted q RMSE: 0.629619
Weighted actuated tau RMSE: 284.066170
Weighted tau jerk RMSE: 152.177417
Left contact F1: 0.9508
Right contact F1: 0.9405
Mean solve time: 26.22 ms/frame
```

Dropout stress test:

```text
--drop-joint-prob 0.2
Weighted MPJPE mean: 0.006430 m
Weighted q RMSE: 0.591666
Weighted actuated tau RMSE: 270.101608
Weighted tau jerk RMSE: 100.442736
Left contact F1: 0.9411
Right contact F1: 0.9344
Mean solve time: 19.14 ms/frame
```

Persistent difficult cases:

- `CNRS/SW_B_3_stageii.csv`: pelvis/trunk/shoulder ambiguity, high MPJPE,
  torque explosion, contact ambiguity.
- `220926_yogi...stageii.csv`: non-locomotion / static asymmetric support
  ambiguity.

Current diagnosis from `RT/NEXT_STEPS.md`:

- raw derivative noise as the main torque-noise source is mostly addressed;
- direct tau optimization as a primary decision variable was replaced;
- fully binary contact representation was softened;
- the remaining bottleneck is weak observability of pelvis/trunk/shoulder
  kinematics from only 12 keypoints;
- Stage 2 can still produce large torques when Stage 1 leaves too much freedom
  in weakly observed coordinates.

Rejected experiments to avoid reintroducing blindly:

- hard torque bounds and torque-rate bounds inside the QP: caused infeasibility;
- over-aggressive Stage 1 filtering as the only state: helped some outliers but
  caused lag and baseline regressions;
- broad shoulder/scapula priors and broad contact heuristics: worsened CNRS or
  yoga/static support cases;
- simple soft torque plausibility priors: worsened nominal and dynamic-outlier
  cases in first attempt.

Accepted improvements:

- Stage 1 weak-observation priors;
- geometric segment priors;
- filtered support-force prior;
- soft contact probability;
- Stage 2 over `ddq`, foot wrench, and root residual rather than primary `tau`;
- explicit `ddq` smoothing;
- fixed-shape cached QPs for dropout runtime.

## Offline vs Real-Time Methodological Differences

Offline estimator:

- non-causal;
- uses the full AMASS/SMPL-X sequence;
- uses dense 105 virtual markers;
- delegates batch scaling and IK to AddBiomechanics;
- uses numerical derivatives from fitted trajectories;
- estimates external loads from full-sequence kinematics and COM dynamics;
- runs OpenSim inverse dynamics after external loads are generated;
- final data are best for dataset generation and offline training.

Real-time estimator:

- strictly causal;
- uses only current 12 keypoints plus persistent previous state;
- estimates kinematics through custom robust IK QP;
- filters derivatives causally;
- represents contacts as soft left/right contact probabilities;
- solves a QP over acceleration, foot wrenches, and root residual;
- reconstructs actuated torque from rigid-body dynamics;
- intended for online sparse-pose dynamics inference.

Key conceptual distinction:

```text
Offline:
  fit q over sequence -> estimate GRF -> OpenSim ID returns tau

Real-time:
  estimate q,dq causally -> QP solves ddq,wrench,residual -> reconstruct tau
```

## Known Limitations and Caveats

Offline:

- Virtual markers are fixed single SMPL-X vertices.
- No learned marker correction or soft-tissue artifact model.
- GRFs are inferred from kinematics alone.
- No force plates, pressure distribution, or learned contact model.
- COP is a projected body-origin proxy.
- Offline free contact moments are zero.
- COM-derived support force may be physically plausible but not uniquely
  identifiable from kinematics.
- AddBiomechanics internals are external and should not be overclaimed.

Real-time:

- Only 12 keypoints are observed.
- Pelvis, trunk, scapula, shoulder, head, wrist, and forearm DOFs can be weakly
  constrained.
- Contact remains ambiguous in unusual support geometries.
- Root residual is intentionally retained as a safety variable; report it as a
  diagnostic.
- Torque fidelity is sensitive to causal acceleration quality.
- Current real-time output uses 51 DOFs, while final offline CSV uses 49
  effective DOFs after excluding knee beta coordinates.

## Research Framing

Suggested paper-style title used in recent discussion:

```text
Does Dynamics Improve Self-Supervised Human Motion Representation Learning?
```

Alternative title examples:

```text
Dynamics-Aware JEPA for Foundation Models of Human Movement
Beyond Kinematics: Dynamics-Aware Foundation Models for Human Motion
Biomechanical Dynamics Improve JEPA-Based Human Motion Foundation Models
Physics-Informed JEPA for Human Movement Foundation Models
Joint Kinematic-Dynamic Pretraining for Human Motion Understanding
```

The cleanest scientific claim is:

```text
We test whether physically grounded dynamic variables provide additional
predictive and transferable information for self-supervised human motion
representation learning beyond kinematics alone.
```

Important ablations for the foundation-model paper:

- kinematics only: `q`, `dq`, `ddq`;
- kinematics plus torques: add `tau`;
- kinematics plus contact/GRF: add GRFs and contact states;
- dynamics only auxiliary loss vs dynamics as input;
- offline dynamics labels vs real-time estimated dynamics;
- contact-aware masked prediction;
- downstream tasks with and without dynamic targets;
- robustness under noise/dropout if using sparse-pose observations.

Potential Q1 journal targets discussed:

- `IEEE Transactions on Neural Networks and Learning Systems`: strong fit for
  JEPA, self-supervised learning, representation learning.
- `IEEE Transactions on Pattern Analysis and Machine Intelligence`: high bar,
  best if there is strong ML/CV novelty and broad benchmark evidence.
- `Nature Machine Intelligence`: very ambitious, best if result is general and
  conceptually strong.
- `Neural Networks`: good ML/representation-learning venue.
- `Pattern Recognition`: good if positioned as motion pattern recognition and
  representation learning.
- `PLOS Computational Biology`: good if framed as a scientific question about
  whether dynamics provide better representations of human movement.
- `IEEE Transactions on Neural Systems and Rehabilitation Engineering`: good if
  framed around human movement, neuroengineering, rehab, motor assessment.
- `IEEE Journal of Biomedical and Health Informatics`: good for health/digital
  biomarker framing.
- `Journal of NeuroEngineering and Rehabilitation`: good movement/rehab fit.
- `Journal of Biomechanics`: suitable if the main contribution becomes
  biomechanical rather than ML-focused.

## Useful References

SMPL-X:

```text
Pavlakos et al., 2019.
Expressive Body Capture: 3D Hands, Face, and Body from a Single Image.
CVPR.
```

AMASS:

```text
Mahmood et al., 2019.
AMASS: Archive of Motion Capture as Surface Shapes.
ICCV.
DOI: 10.1109/ICCV.2019.00554
```

OpenSim:

```text
Delp et al., 2007.
OpenSim: Open-source software to create and analyze dynamic simulations of movement.
IEEE Transactions on Biomedical Engineering.
DOI: 10.1109/TBME.2007.901024
```

```text
Seth et al., 2018.
OpenSim: Simulating musculoskeletal dynamics and neuromuscular control to study
human and animal movement.
PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1006223
```

AddBiomechanics:

```text
Werling et al., 2023.
AddBiomechanics: Automating model scaling, inverse kinematics, and inverse
dynamics from human motion data through sequential optimization.
PLOS ONE.
DOI: 10.1371/journal.pone.0295152
```

NimblePhysics:

```text
Werling et al., 2021.
Fast and Feature-Complete Differentiable Physics for Articulated Rigid Bodies
with Contact.
arXiv:2103.16021
```

OpenCap:

```text
Uhlrich et al., 2023.
OpenCap: Human movement dynamics from smartphone videos.
PLOS Computational Biology.
DOI: 10.1371/journal.pcbi.1011462
```

## Practical Commands

Run a single offline trial:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/A3-_Swing_arms_stageii.npz \
  --trial A3_swing_full \
  --output-dir outputs/bsm \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

Run all trials inside a multi-trial `.npz`:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/your_multitrial_file.npz \
  --trial all \
  --output-dir outputs/bsm \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

Run AMASS folder in parallel:

```bash
python scripts/run_amass_batch_parallel.py \
  --input-root /path/to/AMASS \
  --output-dir outputs/bsm_batch \
  --workers 8 \
  --smplx-model-dir model/smpl \
  --bsm-model model/bsm/bsm.osim \
  --addbio-root "$HOME/AddBiomechanics" \
  --id-grf-mode estimated \
  --cleanup-intermediate
```

Run real-time benchmark on one offline CSV:

```bash
python RT/real_time_test.py \
  --csv data/AMASS/BMLhandball/Trial_upper_left_012_poses.csv \
  --model model/bsm/bsm.osim \
  --output-csv data/AMASS/BMLhandball/Trial_upper_left_012_poses_realtime.csv
```

Run real-time benchmark with noise:

```bash
python RT/real_time_test.py \
  --csv data/AMASS/BMLhandball/Trial_upper_left_012_poses.csv \
  --noise-std 0.01
```

Run real-time benchmark with dropout:

```bash
python RT/real_time_test.py \
  --csv data/AMASS/BMLhandball/Trial_upper_left_012_poses.csv \
  --drop-joint-prob 0.2
```

Generate CSV report PDF:

```bash
python scripts/csv_explorer.py \
  --input-csv outputs/bsm/A3-_Swing_arms_stageii.csv \
  --output-pdf outputs/bsm/A3-_Swing_arms_stageii_report.pdf
```

## Files to Read First in Future Sessions

If a future agent needs to quickly regain context, read in this order:

1. `CODEX.md`
2. `README.md`
3. `DESCRIPTION.md`
4. `scripts/run_amass_to_bsm_csv.py`
5. `src/opensim_batch_dynamics/inverse_dynamics_no_grf.py`
6. `RT/rt_library.py`
7. `RT/real_time_test.py`
8. `RT/RESULTS.md`
9. `RT/NEXT_STEPS.md`

For exact implementation details, trust the code over this summary.

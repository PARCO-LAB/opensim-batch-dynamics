# Next Steps for Real-Time Dynamics

## Goal

The main goal is no longer just to track pose well. The current system already tracks the 12 global keypoints well on many sequences. The remaining goal is to make:

- `tau`
- `ddq`
- GRF
- contact transitions

stable enough to be biomechanically usable, not only numerically feasible.

In practice, the bottleneck is now:

`Stage 1 kinematics -> Stage 2 dynamic explanation -> usable tau`

## Current Status

The implementation has already moved well beyond the original monolithic formulation.

### Implemented

In [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py), the following changes are now implemented:

1. Stage 1 causal kinematic filter
   - persistent `q_kin`, `dq_kin`, `ddq_kin`
   - optional `alpha-beta-gamma` style filtering of kinematic derivatives
   - this is now the default path in `qpid()`

2. Stage 2 without `tau` as a primary decision variable
   - the QP solves for:
     - `ddq`
     - foot wrench
     - root residual
   - `tau` is reconstructed afterward from the actuated rows of rigid-body dynamics

3. Explicit `ddq` smoothing
   - Stage 2 now penalizes `ddq_t - ddq_{t-1}` directly
   - this was introduced because raw acceleration noise was still the main driver of dirty torque

4. Soft contact support
   - there is now a `contact_prob` in addition to binary `contact_state`
   - contact priors, wrench split, and contact penalties are scaled continuously

5. Filtered support-force prior
   - the support prior is now driven by the filtered kinematic state rather than only by the raw one-step derivative path

6. Stronger dynamic regularization
   - stronger torque smoothness penalties
   - stronger `ddq` smoothness penalties
   - stronger wrench smoothness and moment regularization
   - stronger priors on weakly observed DOFs

7. Tau diagnostics
   - [RT/real_time_test.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/real_time_test.py) now reports:
     - `tau_actuated_jerk_rmse`
     - `tau_actuated_jerk_l2_mean`

8. Reliability-aware priors for weakly observed DOFs
   - stronger Stage 1 priors and smoothing for:
     - `pelvis_*` orientation
     - `lumbar_*`, `thorax_*`
     - `scapula_*`
     - `shoulder_*`
   - weakly observed DOFs are now pulled more strongly toward prediction / previous filtered state

9. Geometric Stage 1 priors from observed segment structure
   - geometric residuals are now added for:
     - hip span
     - shoulder span
     - trunk axis from hip center to shoulder center
     - knee span
     - upper-arm, forearm, thigh, and shank segment vectors
   - these are used as additional soft geometric constraints inside the Stage 1 QP

### Tried But Rejected

These were also tried and should not be reintroduced blindly:

1. Hard torque bounds and hard torque-rate bounds inside the QP
   - result: the QP became infeasible on multiple AMASS sequences
   - especially bad on `CNRS/SW_B_3_stageii.csv`
   - conclusion: hard bounds were too brittle at the current conditioning level

2. Over-aggressive Stage 1 filtering used directly as the only state
   - result: it helped on outliers but introduced lag and could hurt the baseline
   - conclusion: filtering is useful, but only as part of a balanced weakly-coupled design

## What Improved

On the tuning subset, the current implementation improved the difficult dynamic cases substantially.

### Representative improvements

Using the current default configuration:

- `BMLhandball`, `159` frames:
  - `tau_actuated_rmse = 299.90`
  - `tau_actuated_jerk_rmse = 127.12`

- `TotalCapture/rom1`, `159` frames:
  - `tau_actuated_rmse = 228.77`
  - `tau_actuated_jerk_rmse = 88.21`

This is materially better than the old behavior on the strong outlier case.

## Benchmark Tracking

The table below tracks the current tuning subset and compares the state before the latest Stage 1 weak-observation priors against the current implementation.

All runs below use:

- current metric mask excluding ankle/head/wrist-related angles
- `--max-frames 160` for the longer sequences
- `--max-frames 80` for `CNRS`

| File | Frames | Previous q RMSE | Current q RMSE | Previous tau actuated RMSE | Current tau actuated RMSE | Previous tau jerk RMSE | Current tau jerk RMSE | Previous MPJPE mean | Current MPJPE mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BMLhandball/Trial_upper_left_012_poses.csv` | `159` | `0.195393` | `0.138785` | `299.897926` | `136.909908` | `127.120888` | `1.906794` | `0.000022` | `0.000752` |
| `TotalCapture/rom1_stageii.csv` | `159` | `0.261543` | `0.140915` | `228.770875` | `127.066740` | `88.209739` | `5.825987` | `0.000064` | `0.001045` |
| `HUMAN4D/INF_Running_S2_01_stageii.csv` | `159` | `0.041421` | `0.041441` | `143.639930` | `143.640501` | `1.156806` | `1.157043` | `0.000034` | `0.000127` |
| `CNRS/SW_B_3_stageii.csv` | `79` | `0.938780` | `0.688289` | `4080.297783` | `3793.441515` | `3953.117671` | `3715.624975` | `0.396519` | `0.412225` |

### Improvement summary

This second table is the one to keep updating after each meaningful algorithmic change. It makes the direction of the change explicit instead of requiring manual subtraction.

Negative deltas are better for:

- `q RMSE`
- `tau actuated RMSE`
- `tau jerk RMSE`
- `MPJPE`

| File | q RMSE delta | tau actuated RMSE delta | tau jerk RMSE delta | MPJPE delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `BMLhandball/Trial_upper_left_012_poses.csv` | `-0.056608` | `-162.988018` | `-125.214094` | `+0.000730` | strong improvement; tiny MPJPE cost acceptable |
| `TotalCapture/rom1_stageii.csv` | `-0.120628` | `-101.704135` | `-82.383752` | `+0.000981` | strong improvement; large torque stabilization |
| `HUMAN4D/INF_Running_S2_01_stageii.csv` | `+0.000020` | `+0.000571` | `+0.000237` | `+0.000093` | neutral; essentially unchanged |
| `CNRS/SW_B_3_stageii.csv` | `-0.250491` | `-286.856268` | `-237.492696` | `+0.015706` | partial improvement only; still failing |

## Metric Mask

Precision metrics in [RT/real_time_test.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/real_time_test.py) and [scripts/realtime_vs_offline_pdf.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/scripts/realtime_vs_offline_pdf.py) already exclude ankle/head/wrist-related angles.

Current excluded DOF prefixes:

| Category | Excluded prefixes |
| --- | --- |
| ankle | `ankle_angle_`, `subtalar_angle_` |
| head | `head_` |
| wrist / forearm | `wrist_`, `pro_sup_` |

Current behavior:

- metric mask built by `include_in_precision_metrics(...)`
- applied to `q`, `dq`, `ddq`, `tau`, worst-DOF ranking, aggregate precision metrics
- not applied to full signal plots, which still show all DOFs

### Test tracking protocol

Every time a new change is tested, this section should be updated in two places:

1. overwrite the `Current ...` columns in the first table with the latest accepted implementation
2. append one line to the experiment log below with:
   - date
   - short name of the change
   - which files were tested
   - whether the change was accepted or rejected

This avoids losing the history of what was tried and why.

### Experiment log

| Date | Change | Files checked | Outcome | Notes |
| --- | --- | --- | --- | --- |
| `2026-04-22` | Stage 1 weak-observation priors + geometric segment priors | `BMLhandball`, `TotalCapture`, `HUMAN4D`, `CNRS` | accepted | strong gain on torque stability for nominal and dynamic-outlier sequences; `CNRS` still poor |
| `2026-04-22` | Fixed-shape DPP cache for Stage 1/Stage 2 under dropout | `BMLhandball`, `TotalCapture`, `HUMAN4D` | accepted | dropout runtime dropped from roughly `170-215 ms/frame` to roughly `10-20 ms/frame` with no material accuracy regression |
| `2026-04-22` | Weak pelvis/trunk orientation priors from hip/shoulder/trunk geometry | `BMLhandball`, `TotalCapture`, `CNRS` + `BMLhandball` noisy/dropout | accepted | small but real gain on `TotalCapture`, neutral-to-small regression on `BMLhandball`, almost neutral on `CNRS`; acceptable as a weak prior |
| `2026-04-22` | Soft torque plausibility priors in Stage 2 | `BMLhandball`, `TotalCapture`, `CNRS` | rejected | increased `tau_actuated_rmse` too much on nominal and dynamic-outlier cases |
| `2026-04-22` | Shoulder-girdle orientation priors + quasi-static asymmetric contact heuristics | `BMLhandball`, `TotalCapture`, `CNRS`, `220926_yogi...` | rejected | `CNRS` got worse, and the yoga case produced a severe right-contact precision/recall regression |
| `2026-04-22` | Ambiguity-gated stabilization on pelvis/trunk/shoulder/scapula subspace | `BMLhandball`, `TotalCapture`, `CNRS`, `220926_yogi...` | rejected | too broad; no gain on yoga, no gain on `CNRS`, slight nominal regression |
| `2026-04-22` | Shoulder/scapula multi-frame gating with side-specific prior modulation | `BMLhandball`, `TotalCapture`, `CNRS`, `220926_yogi...` | rejected | clean gate failed; `BMLhandball` stayed neutral, but `TotalCapture` slightly worsened, `CNRS` worsened, and yoga right-contact F1 stayed poor; noisy/dropout not run |
| `2026-04-22` | Shoulder confidence used only to gate shoulder-line contribution into pelvis/trunk priors | `BMLhandball`, `TotalCapture`, `CNRS`, `220926_yogi...` | rejected | clean gate failed; `BMLhandball` stayed neutral and `TotalCapture` slightly improved, but `CNRS` q RMSE worsened materially and yoga still did not improve; noisy/dropout not run |
| `2026-04-22` | Contact pruning from previous wrench consistency, narrow weak-foot suppression | `BMLhandball`, `TotalCapture`, `CNRS`, `220926_yogi...` | rejected | clean gate failed because yoga did not improve at all; nominal and outlier cases stayed effectively unchanged, so noisy/dropout not run |
| `2026-04-22` | Contact pruning from previous wrench consistency with multi-frame support-dominance memory | `BMLhandball`, `TotalCapture`, `220926_yogi...` | rejected | still no change on yoga right-contact F1, while nominal cases stayed neutral; branch reverted because it added state complexity without measurable gain |

## Robustness Validation

From now on, every accepted change must be checked in three conditions on the tuning subset:

1. `clean`
2. `case 1`: noisy keypoints
   - current default stress test: `--noise-std 0.01`
3. `case 2`: missing keypoints / intermittent detector failures
   - current default stress test: `--drop-joint-prob 0.2`

This is required because the intended runtime regime is closer to a human pose estimator than to perfect marker data.

### Acceptance rule

A change should not be accepted only because it improves the clean case. It should also remain usable under:

- moderate Gaussian noise on the 12 observed keypoints
- random keypoint dropout for some frames

The practical bar is:

- `BMLhandball`, `TotalCapture`, and `HUMAN4D` should remain stable under both stress tests
- `CNRS` is still the stress-failure case, so improvements there are a bonus but not yet the main accept/reject gate
- if a change improves clean metrics but clearly destabilizes the two robustness cases, it should be rejected or kept behind a flag

### Robustness benchmark table

The table below captures the current default implementation under the two synthetic pose-estimator stress tests.

| File | Scenario | MPJPE mean | q RMSE | tau actuated RMSE | tau jerk RMSE | Left F1 | Right F1 | Solve time mean ms | Robustness read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `BMLhandball/Trial_upper_left_012_poses.csv` | `case 1: noise_std=0.01` | `0.014900` | `0.218011` | `211.589420` | `93.247408` | `1.0000` | `1.0000` | `24.68` | acceptable; tracking degrades but stays usable |
| `TotalCapture/rom1_stageii.csv` | `case 1: noise_std=0.01` | `0.020591` | `0.450714` | `193.474702` | `51.482843` | `1.0000` | `1.0000` | `24.80` | acceptable; dynamic outlier still controlled |
| `HUMAN4D/INF_Running_S2_01_stageii.csv` | `case 1: noise_std=0.01` | `0.015089` | `0.235776` | `256.940682` | `99.813873` | `1.0000` | `1.0000` | `28.32` | acceptable; mostly graceful degradation |
| `CNRS/SW_B_3_stageii.csv` | `case 1: noise_std=0.01` | `0.414377` | `0.820250` | `3789.005401` | `3711.958922` | `0.7049` | `0.7158` | `42.84` | failing; confirms base-case weakness |
| `BMLhandball/Trial_upper_left_012_poses.csv` | `case 2: drop_joint_prob=0.2` | `0.000936` | `0.138238` | `135.660528` | `2.276023` | `1.0000` | `1.0000` | `170.96` | robust numerically, but solve time too high |
| `TotalCapture/rom1_stageii.csv` | `case 2: drop_joint_prob=0.2` | `0.005547` | `0.573572` | `179.374171` | `46.654508` | `0.9937` | `0.9840` | `175.49` | robust enough, but with visible kinematic degradation |
| `HUMAN4D/INF_Running_S2_01_stageii.csv` | `case 2: drop_joint_prob=0.2` | `0.000201` | `0.043519` | `142.400083` | `1.302509` | `1.0000` | `1.0000` | `171.54` | strong robustness; dropout handled well |
| `CNRS/SW_B_3_stageii.csv` | `case 2: drop_joint_prob=0.2` | `0.447994` | `0.784520` | `3761.212315` | `3695.549642` | `0.8633` | `0.6889` | `213.98` | still failing; contact and torque remain unusable |

### Reading the robustness table

- The current implementation is reasonably robust to moderate noise on three out of four benchmark files.
- The current implementation is also robust to random 20% keypoint dropout on three out of four benchmark files.
- The main regression under dropout is computational:
  - solve time jumps from roughly `25-30 ms` to roughly `170-215 ms`
  - this is acceptable for robustness debugging, but not for the target realtime budget
- `CNRS` still fails in both stress tests, which is consistent with the clean-case diagnosis:
  - the root issue is weak observability / ambiguity in pelvis-trunk-shoulder kinematics
  - noise and dropout only expose that weakness more clearly

Update:

- the dropout runtime issue has now been fixed by moving to fixed-shape cached QPs
- the current practical dropout timings are now roughly:
  - `BMLhandball`: `11.64 ms/frame`
  - `TotalCapture`: `20.17 ms/frame`
  - `HUMAN4D`: `9.88 ms/frame`
- the remaining open problem is no longer computational under dropout; it is still the `CNRS`-style kinematic ambiguity

## Full AMASS Baseline

Full AMASS batch now evaluated on all `25` original offline CSV files in three modes:

- `clean`
- `noise --noise-std 0.01`
- `dropout --drop-joint-prob 0.2`

Source report:

- [RT/RESULTS.md](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/RESULTS.md)
- [RT/amass_batch_results.json](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/amass_batch_results.json)

Weighted aggregate results:

| Mode | MPJPE mean | q RMSE | tau actuated RMSE | tau jerk RMSE | Left F1 | Right F1 | Solve ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clean` | `0.004694` | `0.536112` | `265.279520` | `94.123808` | `0.9557` | `0.9431` | `20.16` |
| `noise` | `0.018019` | `0.629619` | `284.066170` | `152.177417` | `0.9508` | `0.9405` | `26.22` |
| `dropout` | `0.006430` | `0.591666` | `270.101608` | `100.442736` | `0.9411` | `0.9344` | `19.14` |

Interpretation:

- `noise` hurts more than `dropout`
- `dropout` no longer main compute problem
- persistent outliers across modes:
  - `CNRS/SW_B_3_stageii.csv`
  - `220926_yogi...stageii.csv`

### New rule for future experiments

Every future entry in the experiment log should explicitly state:

- clean result on the tuning subset
- `case 1` noisy result on the tuning subset
- `case 2` dropout result on the tuning subset
- whether runtime under dropout stayed acceptable or not

### Reading the table

- `BMLhandball` improved strongly.
- `TotalCapture` improved strongly.
- `HUMAN4D` stayed essentially unchanged, which is acceptable because it was already good.
- `CNRS` improved, but is still far from acceptable.

This is the key practical conclusion:

- the recent Stage 1 weak-observation priors and geometric priors are good and should stay
- but they are not enough to solve the remaining `CNRS`-type failures

### What is now clearly better

- torque is less dominated by raw one-step acceleration noise
- `ddq` is much better behaved on difficult sequences
- Stage 2 behaves more like real inverse dynamics and less like a monolithic residual absorber
- `TotalCapture`-like dynamic outliers improved significantly

## What Is Still Not Good Enough

The current implementation is not yet globally “solved”.

### Remaining hard case

`CNRS/SW_B_3_stageii.csv` is still very poor.

Observed behavior on the current default:

- high MPJPE
- very poor contact recall
- extremely large `tau_actuated_rmse`
- torque explosion concentrated in pelvis/lumbar/hip channels

This means the remaining failure mode is no longer mainly Stage 2 alone. On these sequences the issue is:

- poorly observed or weakly constrained DOFs
- kinematic ambiguity in pelvis / trunk / shoulder complex
- contact ambiguity
- then large dynamic compensation downstream

After the latest Stage 1 geometric priors, this diagnosis still holds. The improvement on `CNRS` is real but modest, which means:

- segment-structure priors help
- but the missing ingredient is still explicit orientation reasoning for pelvis / trunk / shoulder complex

### Important conclusion

The current bottleneck is now:

1. weakly observed DOFs in Stage 1
2. poor kinematic conditioning on those DOFs
3. Stage 2 then tries to explain that with large torque

So the next work should not only keep tuning Stage 2. The next priority is to constrain the unobserved or weakly observed kinematic subspace better.

## Updated Diagnosis

The original diagnosis in this file was directionally correct:

- raw `ddq_hat` was too noisy
- solving `tau` directly was too permissive
- binary contact switching was too sharp

After implementation, the revised diagnosis is:

### Solved enough to move on

- raw derivative noise as the main source of torque noise
- monolithic `tau` decision variable
- fully binary contact as the main contact representation

### Still open

- underconstrained kinematics for pelvis / trunk / upper limb couplings
- poor observability of some DOFs from only 12 global points
- dataset-specific contact and support ambiguity
- torque explosions when Stage 1 leaves too much freedom in weakly observed coordinates

## Next Priorities

The next changes should focus on constraining the hidden kinematic subspace better rather than only adding more penalties downstream.

## Priority 1: Better priors for weakly observed DOFs

This is now the most important next step.

Status:

- partially implemented
- reliability-aware smoothing and segment-geometry priors are now in place
- remaining work is to move from segment-vector consistency to stronger orientation-specific priors

### What to change

Add explicit kinematic priors for DOFs that are not directly constrained well by the 12 keypoints:

- scapula DOFs
- some shoulder rotations
- lumbar twist / bending / extension
- pelvis orientation
- possibly some elbow / forearm couplings when keypoints are ambiguous

### Ways to do it

1. stronger rest-pose or low-acceleration priors only on low-reliability DOFs
2. joint-group coupling priors:
   - scapula-shoulder
   - thorax-lumbar-pelvis
3. reliability-aware smoothing:
   - if a DOF has low observability, prioritize temporal smoothness over measurement fitting

### Why this is next

This directly attacks the current root cause of the worst remaining failures.

### What remains inside this priority

The next layer to add is:

1. explicit pelvis orientation priors
2. explicit thorax / lumbar orientation priors
3. explicit shoulder-girdle priors tied to scapula and humerus geometry

## Priority 2: Stage 1 should output reliability-weighted anisotropic derivative filters

The current filter is already useful, but it is still fairly generic.

### What to change

Make the Stage 1 derivative filter:

- stronger on low-reliability DOFs
- weaker on highly observed DOFs
- possibly joint-group specific

Examples:

- very strong filtering on scapula and lumbar twist
- moderate filtering on pelvis orientation
- light filtering on well-observed limbs

### Why this matters

Right now some weakly observed DOFs can still generate torque noise even after the structural Stage 2 fixes.

## Priority 3: Add model-based priors for pelvis and trunk orientation

This is likely necessary for `CNRS`-type failures.

This is now the next concrete priority.

### What to change

Introduce explicit geometric priors from:

- pelvis-hip-left/right configuration
- thorax-shoulder-left/right configuration
- global uprightness / gravity consistency

Examples:

- weak upright prior when support is high
- temporal prior on pelvis orientation acceleration
- thorax-pelvis relative orientation prior

### Why this matters

Large torque explosions still cluster in:

- pelvis
- lumbar
- hip

which is exactly where a poor root/trunk estimate propagates into bad inverse dynamics.

### Immediate implementation direction

The next code change should add orientation-style residuals in Stage 1 using observed geometry, for example:

- pelvis left-right axis from `hip_l` and `hip_r`
- trunk vertical / forward axis from hip center to shoulder center
- shoulder-line axis from `GlenoHumeral_l` and `GlenoHumeral_r`

These should be turned into soft residuals that specifically constrain:

- `pelvis_tilt`, `pelvis_list`, `pelvis_rotation`
- `lumbar_extension`, `lumbar_bending`, `lumbar_twist`
- possibly `thorax_*`

Status:

- partially implemented
- weak orientation-style priors from pelvis left-right axis, trunk axis, and shoulder-line geometry are now in place
- they help slightly, but are still not strong enough to solve `CNRS`
- next improvement should be better orientation priors that are:
  - reliability-gated
  - support-aware
  - less dependent on a fixed world-up heuristic alone
- a first direct attempt at shoulder-girdle priors was rejected because it worsened:
  - `CNRS`
  - static / yoga-like asymmetric support

## Priority 4: Improve contact for non-locomotion poses

The current soft contact logic works better than before, but yoga / unusual support geometries still remain difficult.

### What to change

Extend the contact cue logic to better support:

- quasi-static asymmetric poses
- edge-of-support cases
- low-motion but highly loaded support phases

Potential additions:

- CoP plausibility from the previous wrench
- stance persistence prior when COM is clearly above one foot
- better handling of one-foot support with low vertical velocity

### Why this matters

Some remaining torque spikes are still caused by support ambiguity rather than by pure dynamic inconsistency.

Status:

- a first heuristic attempt was rejected
- the failure mode was clear on the yoga benchmark:
  - `right_contact f1` collapsed to `0.2326`
- this means the next contact update must be:
  - much more selective
  - explicitly benchmarked on `220926_yogi...`
  - preferably tied to previous wrench consistency, not only COM proximity heuristics

## Priority 5: Add soft torque plausibility priors, not hard bounds

Hard bounds were too brittle, but the idea itself was not wrong.

### What to change

Instead of hard constraints, use soft per-DOF torque plausibility priors:

- larger penalties when `|tau|` exceeds a heuristic threshold
- larger penalties when `|tau_t - tau_{t-1}|` exceeds a heuristic threshold

Possible implementation:

- iterative reweighting
- Huber-style or piecewise quadratic penalties

### Why this matters

This could keep the solver stable while still discouraging implausible bursts.

Status:

- attempted once
- rejected in the current simple form because it worsened:
  - `BMLhandball`
  - `TotalCapture`
- if retried, it should be done with a more selective design, likely:
  - only on a subset of problematic DOFs
  - or only above an outlier detector / gating condition

## Priority 6: Add batch evaluation on the tuning subset as a standard regression target

This should be formalized now.

### Recommended tuning subset

Keep using this representative subset:

- `BMLhandball/Trial_upper_left_012_poses.csv`
- `TotalCapture/rom1_stageii.csv`
- `CNRS/SW_B_3_stageii.csv`
- `HUMAN4D/INF_Running_S2_01_stageii.csv`
- `220926_yogi_body_hands_..._stageii.csv`

### Why

These cover:

- nominal locomotion
- strong dynamic outlier
- strong kinematic-conditioning failure
- running
- unusual support geometry

## Recommended Metrics Going Forward

The regression target should not be only RMSE.

For each sequence track at least:

- `mpjpe_m`
- `q_rmse`
- `ddq_rmse`
- `tau_actuated_rmse`
- `tau_actuated_jerk_rmse`
- `tau_actuated_jerk_l2_mean`
- left/right contact F1
- dynamics residual mean

## Practical Acceptance Criteria

The next version should only be considered better if it satisfies both:

### 1. It improves nominal performance

On `BMLhandball`:

- `tau_actuated_rmse` lower than current default
- `tau_actuated_jerk_rmse` not worse, ideally lower
- no material regression in MPJPE or contact F1

### 2. It improves at least one hard outlier without destabilizing the rest

On `TotalCapture/rom1` and `HUMAN4D`:

- lower `tau_actuated_rmse`
- lower `tau_actuated_jerk_rmse`
- no solve failures

### 3. It must not collapse on `CNRS`

Even if `CNRS` is not solved yet, the next version should:

- avoid infeasibility
- reduce the worst torque explosions
- improve contact or pelvis/trunk stability

## Immediate Next Work

The next concrete modifications to try are:

1. pelvis / trunk orientation priors from observed keypoint geometry
2. shoulder-girdle orientation priors
3. contact improvements for quasi-static asymmetric support
4. more selective torque plausibility penalties, only if reliability-gated

If only one change is done next, it should be:

1. add explicit pelvis and trunk orientation residuals in Stage 1
2. use them to stabilize `pelvis_*`, `lumbar_*`, and `thorax_*`

That is now the most likely next step to reduce the remaining `CNRS`-style failures without undoing the improvements already obtained on the nominal and dynamic-outlier cases.

Status update:

- all four branches above have now been explicitly tried in at least one concrete form
- only item `1` has produced an accepted change so far
- items `2`, `3`, and `4` have all been tried and rejected in their first broad implementation
- conclusion:
  - remaining work is no longer "try obvious version of listed idea"
  - remaining work is "design narrower, reliability-gated versions"

Current measured snapshot after the latest accepted changes:

- `BMLhandball`, clean:
  - `q_rmse = 0.135278`
  - `tau_actuated_rmse = 136.855040`
  - `tau_actuated_jerk_rmse = 1.915582`
- `TotalCapture`, clean:
  - `q_rmse = 0.143382`
  - `tau_actuated_rmse = 125.239115`
  - `tau_actuated_jerk_rmse = 5.687225`
- `CNRS`, clean:
  - `q_rmse = 0.678330`
  - `tau_actuated_rmse = 3784.076182`
  - `tau_actuated_jerk_rmse = 3712.989375`
- `BMLhandball`, noise `0.01 m`:
  - `tau_actuated_rmse = 209.723666`
- `BMLhandball`, dropout `20%`:
  - `tau_actuated_rmse = 135.343884`
  - `solve_time_ms = 11.64`

Latest attempted but rejected branch:

- shoulder-girdle priors + quasi-static asymmetric contact heuristics

Observed regressions:

- `BMLhandball`, clean:
  - `tau_actuated_rmse` worsened from `136.855040` to `137.183311`
- `TotalCapture`, clean:
  - `tau_actuated_rmse` worsened from `125.239115` to `125.320900`
- `CNRS`, clean:
  - `q_rmse` worsened from `0.678330` to `0.988272`
  - `left/right contact f1` worsened to `0.8382 / 0.6742`
- `220926_yogi...`, clean:
  - `right_contact f1 = 0.2326`

Conclusion:

- the next version of `Immediate Next Work` should not use broad shoulder/contact heuristics
- it should instead use:
  - reliability-gated shoulder priors only when upper-arm geometry is strong and symmetric
  - contact persistence tied to previous support wrench consistency
  - explicit yoga/static benchmark gating before acceptance

Update after the latest two narrower attempts:

- even reliability-gated shoulder/scapula priors are still too risky if they directly change Stage 1 priors or trunk gains
- using shoulder confidence only to gate shoulder-derived trunk cues also did not pass, because:
  - `CNRS` remained unstable or regressed
  - yoga/static support did not improve
- practical conclusion:
  - the next accepted gain is unlikely to come from "more shoulder prior"
  - the next step should move to contact/wrench consistency or to explicit diagnostics of how shoulder asymmetry contaminates the trunk estimate before applying any new prior

Latest rejected branch after that:

- ambiguity-gated stabilization of pelvis/trunk/shoulder/scapula

Observed outcome:

- `BMLhandball`, clean:
  - `tau_actuated_rmse` changed from `136.855040` to `136.761653`
  - essentially neutral
- `TotalCapture`, clean:
  - `tau_actuated_rmse` changed from `125.239115` to `125.252986`
  - essentially neutral
- `CNRS`, clean:
  - `tau_actuated_rmse` changed from `3784.076182` to `3785.748064`
  - `left/right contact f1` changed to `0.7656 / 0.7174`
  - not a meaningful gain
- `220926_yogi...`, clean:
  - `right_contact f1 = 0.2381`
  - still failing badly

Conclusion:

- generic "ambiguity smoothing" also not enough
- next useful work must be even more targeted:
  - DOF-local gating
  - support/wrench-consistency gating
  - perhaps sequence-class-dependent heuristics

## Revised Next Work

The original immediate list has now been fully exercised in first-pass form. The next useful steps are narrower:

1. dof-local reliability gating on `shoulder_*` and `scapula_*`
   - activate only if both shoulder and elbow geometry are consistent for multiple consecutive frames
2. contact pruning from previous wrench consistency
   - use previous vertical load + current height/velocity
   - do not use broad COM-proximity heuristics
3. special handling for `CNRS`-like shoulder-dominated failures
   - inspect whether one-sided shoulder observations are poisoning pelvis/trunk through shared priors
4. optional sequence-mode classifier
   - locomotion-like vs quasi-static asymmetric support
   - switch only contact heuristics, not whole solver

## Concrete Next Steps

What I would do next, in order:

| Priority | Change | Why | Acceptance gate |
| --- | --- | --- | --- |
| `1` | shoulder/scapula priors with multi-frame gating | broad shoulder priors failed; need activation only when upper-arm geometry stable | `CNRS` shoulder DOFs improve, `BMLhandball` and yoga do not regress |
| `2` | contact pruning from previous wrench consistency | yoga/static failures look like false persistent support on wrong foot | `220926_yogi...` right-contact F1 improves without hurting locomotion F1 |
| `3` | isolate pelvis/trunk priors from contaminated shoulder cues | `CNRS` suggests bad shoulder cues leak into trunk/root stabilization | lower `CNRS` `tau_actuated_rmse` and better shoulder-side worst DOFs |
| `4` | add full-batch acceptance gate for every accepted patch | tuning subset not enough now | no accepted patch without `clean/noise/dropout` AMASS summary |
| `5` | reduce remaining cvxpy parameter count | runtime OK now, but warning remains | warning reduced or removed without metric regression |

Practical next experiment:

1. build per-side shoulder confidence using:
   - shoulder-elbow segment consistency
   - 3-5 frame temporal stability
2. use that confidence only to gate `shoulder_*` / `scapula_*` priors
3. do **not** feed low-confidence shoulder cues into pelvis/trunk stabilization
4. benchmark on:
   - `BMLhandball`
   - `TotalCapture`
   - `CNRS`
   - `220926_yogi...`
   - then full AMASS if accepted

Status update after running exactly that experiment:

- two concrete variants were tried:
  1. side-specific multi-frame shoulder/scapula prior modulation
  2. shoulder-confidence gating only on shoulder-derived pelvis/trunk cues
- both were rejected

Measured outcomes against the accepted baseline:

| Variant | BMLhandball | TotalCapture | CNRS | 220926_yogi... | Decision |
| --- | --- | --- | --- | --- | --- |
| side-specific multi-frame shoulder/scapula prior modulation | `tau_actuated_rmse 136.839130` vs `136.855040` | `125.258114` vs `125.239115` | `tau_actuated_rmse 3788.277315` vs `3784.076182` | `right_contact_f1 0.2367` | rejected |
| shoulder-confidence gating only on shoulder-derived pelvis/trunk cues | `tau_actuated_rmse 136.855040` vs `136.855040` | `125.009504` vs `125.239115` | `q_rmse 0.938297` vs `0.678330`; `tau_actuated_rmse 3783.260191` vs `3784.076182` | `right_contact_f1 0.2367` | rejected |

Practical conclusion:

- `Priority 1` has now been exercised in broad, medium, and narrow forms
- no version passed the acceptance gate cleanly
- the next implementation should now shift to `Priority 2` and `Priority 3` in a more diagnostic-first way

Revised immediate order:

1. CNRS diagnostic split
   - log or inspect whether one shoulder side dominates the Stage 1 residuals
   - only then decide if a new trunk-isolation prior is justified
2. contact diagnostics for yoga/static cases
   - inspect whether the false-positive foot is already low in height but kept alive by score hysteresis, support split, or post-threshold clamping
   - only then decide whether pruning should act on:
     - contact score
     - contact probability
     - support-force split
3. only after that, revisit shoulder priors
   - but they must be passive diagnostics first, not active stabilizers

Latest contact-pruning conclusion:

- two narrow `Priority 2` variants have now been tried
- neither changed yoga at all:
  - `right_contact_f1` stayed at `0.2367`
- both also left `BMLhandball` and `TotalCapture` essentially unchanged
- practical reading:
  - the yoga failure is probably not caused by the final pruning stage alone
  - the false-positive foot is likely being re-activated upstream by:
    - the raw contact score
    - the `support_ratio >= 0.25` re-activation branch
    - or the support-force split itself
- therefore the next useful move is diagnostic-first, not another blind pruning tweak

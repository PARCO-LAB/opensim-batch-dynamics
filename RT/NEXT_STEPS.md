# Next Steps for Real-Time Dynamics

## Goal

The current formulation is already good as a causal kinematic tracker, but `tau` is still too noisy to be used directly. The next work should focus on making torque estimation stable across the full AMASS batch, especially on hard sequences such as:

- `TotalCapture/rom1_stageii.csv`
- `CNRS/SW_B_3_stageii.csv`
- `220926_yogi_body_hands_..._stageii.csv`
- `Transitions/airkick_stand_stageii.csv`

The main objective is not just reducing kinematic error. It is to make `tau`, `ddq`, GRF, and contact transitions physically smoother and more usable frame-to-frame.

## Current Situation

From [RT/RESULTS.md](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/RESULTS.md):

- baseline sequence has very small MPJPE
- contact classification is now strong
- `tau_actuated_rmse` is still around `300+` on the baseline
- `ddq_rmse` is still very large

From [RT/AMASS_GENERALIZATION.md](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/AMASS_GENERALIZATION.md):

- weighted mean `tau_actuated_rmse` improved a lot, but it is still `566.99`
- some sequences remain badly conditioned dynamically even when MPJPE is good
- worst outlier is still `TotalCapture/rom1_stageii.csv`, where kinematics are acceptable but dynamic quantities blow up

This means the main remaining problem is no longer pose tracking. It is the dynamic consistency path from `q_hat, dq_hat` to `ddq, wrench, tau`.

## Why `tau` Is Still Noisy

### 1. `ddq_hat` is still obtained from a one-step velocity difference

In [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:440), `dq_hat` is computed from a one-step blend between finite-difference velocity and prediction. Then in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:445), `ddq_hat` is just:

```python
ddq_hat = (dq_hat - dq_prev) / dt
```

This is still a very high-gain operation. Even when `q_hat` is excellent, small frame-to-frame changes produce large acceleration spikes. Those spikes immediately contaminate:

- the Stage 2 `ddq_target` in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:666)
- the COM acceleration prior in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:467)
- the final inverse dynamics balance

This is probably the single biggest remaining source of dirty `tau`.

### 2. Stage 2 solves for `tau` directly, instead of deriving it from a stabilized dynamic state

The current QP optimizes `ddq`, `tau`, wrench, and root residual at the same time in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:727). The torque terms are only lightly regularized:

- `tau_smooth_const = 0.05 * I` in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:749)
- `tau_reg_const = 0.004 * I` in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:750)

And the smoothness target is just the previous torque in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:842):

```python
b_tau_s = 0.05 * tau_prev
```

So the solver still has a lot of freedom to move `tau` frame-by-frame in order to absorb modeling error, contact mismatch, or acceleration noise.

### 3. `q` and `dq` are taken from Stage 1, but `ddq` comes from Stage 2

At the end of `qpid()`, the exported state is:

- `q = q_hat`
- `dq = dq_hat`
- `ddq = ddq_full`

see [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:939).

This is a reasonable weakly-coupled design, but it creates a structural mismatch:

- pose and velocity come from the kinematic stage
- acceleration comes from the dynamic stage

If Stage 2 needs to move `ddq` to satisfy contact and dynamics, `tau` may become the variable that absorbs inconsistency between the Stage 1 kinematics and the Stage 2 dynamic explanation.

### 4. Contact is still binary

Even though contact classification is much better, the final activation is still boolean in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:605). That means the QP still switches between:

- wrench free
- wrench exactly zero

This creates mode-switch discontinuities. Even if the switch is physically correct, the effect on `ddq`, wrench, and `tau` can still be abrupt.

### 5. The support-force prior is driven by raw COM acceleration

The support-force prior is built from:

```python
support_force_target = mass * (com_acceleration - gravity)
```

in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:470).

That is physically meaningful, but only if `com_acceleration` is already stable. Right now it still comes from the same `ddq_hat` chain that is noisy, so the force prior can inject avoidable jitter into:

- vertical GRF
- tangential GRF
- wrench split
- final `tau`

### 6. There are no explicit torque bounds or torque-rate bounds

The QP only regularizes torque norm and torque change. It does not constrain:

- `|tau|`
- `|tau_t - tau_{t-1}|`

That is why some difficult sequences still show implausible torque bursts even when the solve remains feasible.

## Priority Order

The next experiments should be done in this order:

1. Stabilize `ddq` before it reaches Stage 2
2. Stop solving raw `tau` directly if possible
3. Make contact and wrench transitions softer
4. Add hard physical bounds on torque and torque rate
5. Rebalance root residual vs contact wrench vs torque

## Recommended Modifications

## Step 1: Add a causal kinematic derivative filter

This should be the next change to implement first.

### What to change

Replace the current one-step derivative logic with a causal filtered state for:

- `q_filt`
- `dq_filt`
- `ddq_filt`

Options:

- `alpha-beta-gamma` filter
- causal Savitzky-Golay over a short window
- fixed-lag causal least-squares derivative fit

The simplest good option is an `alpha-beta-gamma` filter per DOF, driven by `q_hat`.

### Why it matters

Right now, even tiny pose jitter becomes large acceleration noise. If `ddq_target` becomes smoother, then:

- torque becomes smoother
- COM acceleration prior becomes smoother
- contact wrench no longer needs to chase high-frequency artifacts

### Implementation direction

Add to the persistent state:

- `q_kin`
- `dq_kin`
- `ddq_kin`

Then Stage 1 should output:

- raw `q_hat`
- filtered `q_kin, dq_kin, ddq_kin`

Stage 2 should use `q_kin, dq_kin, ddq_kin` as priors instead of the current raw `q_hat, dq_hat, ddq_hat`.

### Expected impact

This is the most likely single change to reduce `tau` noise without damaging tracking.

## Step 2: Remove `tau` as a primary optimization variable

This is the most important structural experiment after Step 1.

### What to change

Instead of optimizing `tau` directly inside the Stage 2 QP, solve Stage 2 for:

- `ddq`
- foot wrench
- root residual

Then recover torque algebraically from rigid-body dynamics:

`tau = S ( M ddq + h - J_w^T w - r )`

where `S` extracts the actuated coordinates.

### Why it matters

Right now `tau` is a free variable, so it can absorb dynamic inconsistency and solver noise. If `tau` is instead computed from a stabilized dynamic solution, it becomes a consequence of the solve rather than an additional degree of freedom.

This is much closer to the offline pipeline:

- estimate kinematics
- estimate contact
- run inverse dynamics

### Variants to try

Variant A:
- QP solves only `ddq`, wrench, `r`
- `tau` computed afterward

Variant B:
- QP still has `tau`, but add a hard equality tying it to dynamics projection
- this is less attractive than Variant A

### Expected impact

This should reduce frame-to-frame torque chatter and make the result easier to interpret.

## Step 3: Add explicit jerk regularization

### What to change

Add a state variable for previous acceleration and previous wrench, then penalize:

- `ddq_t - ddq_{t-1}`
- `w_t - w_{t-1}`
- `r_t - r_{t-1}`

Some of this already exists for wrench and root residual, but not strongly enough and not with a clear focus on acceleration smoothness.

### Why it matters

Torque is highly sensitive to acceleration. If `ddq` is smooth, `tau` becomes much smoother.

### Implementation direction

Add a term stronger than the current `ddq` target penalty, for example:

`|| W_jerk (ddq_t - ddq_{t-1}) ||^2`

This should likely matter more than increasing the direct `tau` regularization alone.

## Step 4: Use soft contact activation instead of binary on/off contact

### What to change

Keep the current contact score, but do not convert it immediately to a boolean that hard-switches wrench constraints. Instead build a continuous `contact_prob` in `[0, 1]` and use it to scale:

- contact acceleration penalties
- contact velocity penalties
- wrench priors
- force normal lower bound activation
- torsion / CoP penalties

### Why it matters

A foot near lift-off or near touchdown should not jump between:

- full contact model
- no contact model

That jump still creates impulses in dynamic quantities even if pose tracking is stable.

### Expected impact

This should reduce spikes in:

- `ddq`
- GRF
- `tau`

especially on fast transitions and asymmetrical support.

## Step 5: Filter the COM support-force prior

### What to change

Do not compute the support-force prior from raw instantaneous COM acceleration only. Instead compute:

- filtered COM velocity
- filtered COM acceleration

or derive the support prior from filtered `ddq_kin`.

### Why it matters

The current support prior is physically meaningful but still too sensitive. This is likely one reason some sequences still show excessive vertical GRF and torque bursts.

### Expected impact

Cleaner:

- vertical GRF
- force split left/right
- hip and pelvis torque

## Step 6: Add torque and torque-rate bounds

### What to change

Add either:

- heuristic per-DOF torque bounds
- or bounds derived from model / actuator metadata if available

Also add explicit rate bounds:

- `tau_min <= tau_t <= tau_max`
- `|tau_t - tau_{t-1}| <= delta_tau_max`

### Why it matters

Regularization alone does not prevent occasional large bursts.

### Caveat

This should not be the first change. If done too early, it may just hide the real problem by clipping output. It becomes useful after `ddq` and contact are cleaner.

## Step 7: Rebalance root residual vs torque

### What to check

The root residual is currently regularized in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:755). It may still be cheaper in some situations for the solver to push error into `tau`, and in others to push it into `r`.

### What to try

Run an ablation:

- stronger root residual penalty
- weaker root residual penalty
- anisotropic root residual penalty by axis

Then inspect how that changes:

- pelvis torque
- hip torque
- `tau_actuated_rmse`
- dynamics residual

### Why it matters

Some current outliers look like incorrect load sharing between:

- contact wrench
- root residual
- actuated torque

## Step 8: Improve the wrench prior on foot moments

### What to change

The current split prior initializes foot moments to zero in [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py:650). That is too weak for motions where support moment is important.

Try:

- CoP continuity prior
- moment-rate regularization stronger than force-rate regularization
- stance-phase prior on `Mx, My, Mz`

### Why it matters

If support moments jump, `tau` will jump too, especially in pelvis and hips.

## Step 9: Add diagnostics targeted at `tau`

Before changing too much code, extend evaluation so we can separate the sources of torque noise.

### Add these metrics

- frame-to-frame `tau` jerk:
  `mean ||tau_t - tau_{t-1}||`
- per-DOF torque jerk RMSE
- spectral energy ratio above a cutoff frequency
- correlation between `|tau_t - tau_{t-1}|` and contact state changes
- correlation between `|tau_t - tau_{t-1}|` and `ddq` spikes

### Why it matters

Right now the reports focus on RMSE. That is necessary but not sufficient. A signal can have acceptable RMSE and still be unusably spiky.

## Concrete Plan

The recommended implementation order is:

1. Add causal filter for `q/dq/ddq`
2. Feed filtered kinematics into Stage 2 and filtered COM prior
3. Remove `tau` from the Stage 2 decision vector and recover it after the solve
4. Add `ddq` jerk penalty
5. Move contact from binary to soft activation
6. Add torque-rate bounds
7. Tune root residual weighting
8. Tune wrench-moment priors

## Acceptance Criteria

For a change to be considered successful, it should improve not only MPJPE but especially torque usability.

Minimum acceptance targets:

- no regression in contact F1 on the baseline sequence
- no major regression in AMASS weighted mean MPJPE
- lower weighted mean `tau_actuated_rmse`
- lower `tau` jerk on baseline and AMASS batch
- better behavior on `TotalCapture/rom1_stageii.csv`

Recommended torque-specific targets:

- reduce baseline `tau_actuated_rmse` below `250`
- reduce AMASS weighted mean `tau_actuated_rmse` below `450`
- reduce worst-case outlier torque bursts, even if RMSE improvement is only moderate

## Immediate Next Experiment

If only one modification is implemented next, it should be:

1. add a causal filter for `q/dq/ddq`
2. feed the filtered `ddq` into Stage 2
3. recompute `support_force_target` from filtered COM acceleration

If that works, the next structural change should be:

4. remove `tau` from the QP and compute it after the dynamic solve

That combination is the most likely path to turn `tau` from a mathematically feasible signal into a usable biomechanical output.

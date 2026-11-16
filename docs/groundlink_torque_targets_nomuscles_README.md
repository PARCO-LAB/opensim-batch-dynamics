# GroundLink `torque_targets_12hz_nomuscles`

## Current delivery status (2026-11-15)

The complete 206-trial cohort (861,064 frames) is copied to
`~/nas/MAEVE/GroundLink_torque/torque_targets_12hz_nomuscles`. The HPC source is
the path recorded in `logs/acceptance_checks.json`. The user authorized this
copy with the criterion-3 discrepancies documented below; the acceptance
report still records criterion 3 as failed and has not been relabeled as a
pass.

Criterion 3 remains outside its `1e-6` pelvis-residual limit in three segments:
`s005/sidestretch_3/segment_1` (max difference `0.00925238`),
`s005/sidestretch_4/segment_1` (`0.00627283`), and
`s005/sidestretch_4/segment_2` (`0.00262747`).

This dataset reruns inverse dynamics for the same 206 GroundLink trials selected
by the H1 index. It preserves the source trial folders, segment boundaries,
frame/time grid, models, IK coordinates, external-load files, and the coordinate
filter cutoff saved in each setup. The only inverse-dynamics setup change is
`forces_to_exclude = Actuators`, which removes muscle actuator contributions
from the reported joint moments while retaining the external loads.

For each segment, `target` moments use the measured force-plate loads and `hat`
moments use the saved estimated-GRF loads. Both use the same saved scaled BSM,
IK coordinates, setup cutoff, and time range. Setup and input hashes are
recorded in each trial's `run_metadata.json`.

The output `torque_targets.csv` keeps the source `frame`, `time`, `segment`, and
validity labels, adds `cop_invalid`, and masks a frame from `valid` if either
measured foot has vertical force above 20 N with a zero CoP. Per-trial metadata
records the number of affected frames and foot-frames.

`logs/acceptance_checks.json` records the acceptance checks. The revised ankle
comparison is reported per trial and side for both target and estimated moments:
the maximum absolute change must be below 2 N·m and below 2% of that signal's
source absolute-moment p99. It also records the number of frames above
`1e-6` N·m; any threshold violations are listed without stopping the cohort.
The pelvis residual comparison retains its strict `1e-6` limit and is blocking.
`logs/ankle_drift_per_trial.json` contains the full per-trial ankle report.

The acceptance report also checks the reference target values, force-free swing
frames, moment magnitudes normalized by each model's body mass, cohort/frame
integrity, and exact `Actuators` versus `Muscles` equivalence on
`s001/hopping_1/segment_0`.

The NAS copy dereferences the HPC `Geometry` links so the output does not
depend on HPC paths. The same geometry assets occur under multiple segment
paths; `rsync -L` copied those paths independently. This is redundant storage,
not a difference in the torque outputs.

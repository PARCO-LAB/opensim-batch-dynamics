# CVXPY Full-Sequence Evaluation (Current Build)

## Setup

- Sequence: `data/AMASS/BMLhandball/Trial_upper_left_012_poses.csv`
- Model: `model/bsm/bsm.osim`
- Runtime environment: `conda run -n opensim-torque`
- Solver stack:
  - `cvxpy 1.6.5`
  - `osqp 1.0.4`
  - `numpy 1.25.2`
- Evaluated code:
  - [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py)
  - [RT/real_time_test.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/real_time_test.py)
- Main benchmark input: clean 12 global joint targets from the offline CSV, no injected noise, no dropout

## What Changed

- Stage 2 now uses a less gait-biased contact classifier with hysteresis and causal fallback support activation.
- A soft support-force prior from COM dynamics is used only as guidance, with physical clipping and causal rate limiting.
- `ddq` is more strongly regularized toward Stage 1 and previous-frame values to reduce torque/GRF blow-ups.

## Full-Sequence Coverage

- Total frames in CSV: `1005`
- Inference steps: `1004`
- Sequence duration: `8.35834 s`

## Inference Timing

These timings are end-to-end wall-clock measurements of the full script run on the clean sequence.

| Metric | Value |
| --- | ---: |
| Total wall time | `21.1611 s` |
| Mean wall time / frame | `21.0768 ms` |
| Mean throughput | `47.45 FPS` |
| Real-time factor vs 50 Hz | `0.95x` |

## Accuracy vs Offline Ground Truth

### Aggregate Metrics

| Metric | Value |
| --- | ---: |
| MPJPE mean | `0.000394 m` |
| MPJPE max | `0.010616 m` |
| q RMSE | `0.562347` |
| q MAE | `0.322635` |
| dq RMSE | `13.208406` |
| dq MAE | `9.700251` |
| ddq RMSE | `624.678090` |
| ddq MAE | `216.962632` |
| tau full RMSE | `369.664399` |
| tau full MAE | `184.272461` |
| tau actuated RMSE | `349.688294` |
| tau actuated MAE | `170.039290` |
| Left foot GRF RMSE | `955.458510` |
| Right foot GRF RMSE | `567.823985` |
| Dynamics residual mean | `0.000970` |
| Dynamics residual max | `0.158497` |

### Contact Classification

| Foot | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Left | `0.9373` | `1.0000` | `0.9373` | `0.9676` |
| Right | `0.8984` | `0.9442` | `0.9022` | `0.9227` |

## Additional Regression Runs

### Clean Short Run (`--max-frames 40`)

| Metric | Value |
| --- | ---: |
| MPJPE mean | `0.000021 m` |
| MPJPE max | `0.000050 m` |
| Dynamics residual mean | `0.000915` |
| q RMSE | `0.104120` |
| tau actuated RMSE | `122.025904` |
| Left contact F1 | `1.0000` |
| Right contact F1 | `1.0000` |

### Noisy + Dropout Run (`--max-frames 15 --noise-std 0.01 --drop-joint-prob 0.2`)

| Metric | Value |
| --- | ---: |
| MPJPE mean | `0.016861 m` |
| MPJPE max | `0.026813 m` |
| Dynamics residual mean | `0.000594` |
| q RMSE | `0.146881` |
| tau actuated RMSE | `201.461284` |
| Left contact F1 | `1.0000` |
| Right contact F1 | `1.0000` |

## Interpretation

- On the main clean sequence, kinematic tracking is still very strong and contact classification is now much better than in the earlier `cvxpy` build.
- Runtime is now close to strict `50 Hz`, though still slightly under in end-to-end wall time.
- The main remaining weakness is not baseline target following but torque/GRF quality on hard out-of-distribution motions.
- This matches the AMASS-wide report in [RT/AMASS_GENERALIZATION.md](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/AMASS_GENERALIZATION.md): the new Stage 2 generalizes much better overall, but `TotalCapture/rom1_stageii.csv` is still the hardest dynamic outlier.

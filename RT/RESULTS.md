# CVXPY Full-Sequence Evaluation (Foot Wrench Build)

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

## What Changed

- Stage 2 now uses a **6D foot wrench per foot** as primary contact variable.
- Rigid-body dynamics now include both force and moment contributions at each foot.
- The wrench QP includes:
  - friction bounds on foot forces
  - CoP-like moment bounds
  - torsional-moment bounds
  - angular contact stabilization for feet in support
- The non-DPP warning was removed from the wrench formulation. The remaining `cvxpy` warning is only about the number of parameters.

## Full-Sequence Coverage

- Total frames in CSV: `1005`
- Inference steps: `1004`
- Sequence duration: `8.35834 s`

## Inference Timing

These timings are end-to-end wall-clock measurements of the full script run on the clean sequence.

| Metric | Value |
| --- | ---: |
| Total wall time | `21.7802 s` |
| Mean wall time / frame | `21.6934 ms` |
| Mean throughput | `46.10 FPS` |
| Real-time factor vs 50 Hz | `0.92x` |

## Accuracy vs Offline Ground Truth

### Aggregate Metrics

| Metric | Value |
| --- | ---: |
| MPJPE mean | `0.000440 m` |
| MPJPE max | `0.013644 m` |
| q RMSE | `0.581901` |
| q MAE | `0.319369` |
| dq RMSE | `13.017680` |
| dq MAE | `9.310759` |
| ddq RMSE | `618.039382` |
| ddq MAE | `212.919118` |
| tau full RMSE | `331.366343` |
| tau full MAE | `161.994808` |
| tau actuated RMSE | `304.681881` |
| tau actuated MAE | `145.721461` |
| Left foot GRF RMSE | `777.528046` |
| Right foot GRF RMSE | `570.249351` |
| Dynamics residual mean | `0.000991` |
| Dynamics residual max | `0.105309` |

### Contact Classification

| Foot | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Left | `0.9900` | `1.0000` | `0.9900` | `0.9950` |
| Right | `0.9353` | `0.9408` | `0.9644` | `0.9525` |

## Additional Regression Runs

### Clean Short Run (`--max-frames 40`)

| Metric | Value |
| --- | ---: |
| MPJPE mean | `0.000019 m` |
| MPJPE max | `0.000049 m` |
| Dynamics residual mean | `0.000384` |
| q RMSE | `0.113013` |
| tau actuated RMSE | `118.123727` |
| Left contact F1 | `1.0000` |
| Right contact F1 | `1.0000` |

### Noisy + Dropout Run (`--max-frames 15 --noise-std 0.01 --drop-joint-prob 0.2`)

| Metric | Value |
| --- | ---: |
| MPJPE mean | `0.016834 m` |
| MPJPE max | `0.026951 m` |
| Dynamics residual mean | `0.000546` |
| q RMSE | `0.146898` |
| tau actuated RMSE | `192.094299` |
| Left contact F1 | `1.0000` |
| Right contact F1 | `1.0000` |

## Interpretation

- On the baseline clean sequence, the foot-wrench build improves torque estimation relative to the previous force-only build while keeping sub-millimetric mean MPJPE.
- Contact classification remains strong and robust under noise/dropout.
- The remaining difficult case is still out-of-distribution high-dynamics motion such as `TotalCapture/rom1_stageii.csv`, where the wrench formulation is now competitive with the previous 3D-force build but not yet clearly superior.
- The next useful step is to rerun the full AMASS batch with this stabilized wrench build and decide whether it should replace the force-only version as the default.

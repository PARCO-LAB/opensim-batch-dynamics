# CVXPY Full-Sequence Evaluation (Problem Reuse)

## Setup

- Sequence: `data/AMASS/BMLhandball/Trial_upper_left_012_poses.csv`
- Model: `model/bsm/bsm.osim`
- Runtime environment: `conda run -n opensim-torque`
- Solver stack:
  - `cvxpy 1.6.5`
  - `osqp 1.0.4`
  - `numpy 1.25.2`
- Main benchmark input: clean 12 global joint targets from the offline CSV, no injected noise, no dropout
- Evaluated code:
  - [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py)
  - [RT/real_time_test.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/real_time_test.py)

## What Changed

- Stage 1 and Stage 2 now cache and reuse the same `cvxpy.Problem` objects across frames.
- The cache key is no longer just the number of valid measurements; it includes the actual valid-measurement mask, plus contact mode for Stage 2.
- The objectives were kept in least-squares form so the reused problem is structurally stable.
- This reduced average full-sequence runtime from about `41.9 ms/frame` to about `36.4 ms/frame`.

## Full-Sequence Coverage

- Total frames in CSV: `1005`
- Inference steps: `1004`
- Sequence duration: `8.35834 s`

## Inference Timing

| Metric | Value |
| --- | ---: |
| Total inference time | `36.6266 s` |
| Mean inference time / frame | `36.4273 ms` |
| Median inference time / frame | `23.3457 ms` |
| P95 inference time / frame | `142.1043 ms` |
| Mean throughput | `27.45 FPS` |
| Real-time factor vs 50 Hz | `0.55x` |

## Accuracy vs Offline Ground Truth

### Aggregate Metrics

| Metric | Value |
| --- | ---: |
| MPJPE mean | `0.000725 m` |
| MPJPE max | `0.010738 m` |
| q RMSE | `0.534296` |
| q MAE | `0.312611` |
| dq RMSE | `14.310799` |
| dq MAE | `11.133559` |
| ddq RMSE | `630.277354` |
| ddq MAE | `243.616183` |
| tau full RMSE | `625.601633` |
| tau full MAE | `272.636667` |
| tau actuated RMSE | `615.857299` |
| tau actuated MAE | `258.678415` |
| Left foot GRF RMSE | `456.670441` |
| Right foot GRF RMSE | `241.255613` |
| Dynamics residual mean | `0.006866` |
| Dynamics residual max | `0.309360` |

### Contact Classification

#### Left Foot

| Metric | Value |
| --- | ---: |
| Accuracy | `0.1643` |
| Precision | `1.0000` |
| Recall | `0.1643` |
| F1 | `0.2823` |

#### Right Foot

| Metric | Value |
| --- | ---: |
| Accuracy | `0.4114` |
| Precision | `1.0000` |
| Recall | `0.1244` |
| F1 | `0.2213` |

### Worst DOFs by q RMSE

| DOF | RMSE |
| --- | ---: |
| `pelvis_tilt` | `1.4835` |
| `pelvis_rotation` | `1.4759` |
| `pro_sup_r` | `1.0900` |
| `pro_sup_l` | `1.0314` |
| `head_twist` | `0.9694` |
| `wrist_flexion_l` | `0.9077` |
| `wrist_flexion_r` | `0.8476` |
| `head_extension` | `0.8236` |

### Worst DOFs by tau RMSE

| DOF | RMSE |
| --- | ---: |
| `scapula_upward_rot_r` | `1623.7830` |
| `shoulder_r_x` | `1601.7425` |
| `pelvis_tz` | `1249.2415` |
| `lumbar_extension` | `1196.5971` |
| `thorax_extension` | `1183.2593` |
| `hip_flexion_r` | `1024.5079` |
| `thorax_bending` | `956.3173` |
| `lumbar_bending` | `920.4536` |

### GRF Axis RMSE

#### Left Foot

| Axis | RMSE |
| --- | ---: |
| `fx` | `244.0911` |
| `fy` | `251.9037` |
| `fz` | `708.9483` |

#### Right Foot

| Axis | RMSE |
| --- | ---: |
| `fx` | `159.9037` |
| `fy` | `99.8547` |
| `fz` | `372.9244` |

## Additional Regression Runs

### Clean Short Run (`--max-frames 40`)

| Metric | Value |
| --- | ---: |
| MPJPE mean | `0.000014 m` |
| MPJPE max | `0.000052 m` |
| Dynamics residual mean | `0.010887` |
| q RMSE | `0.235939` |
| tau actuated RMSE | `221.820760` |
| Left contact F1 | `0.8696` |
| Right contact F1 | `0.8529` |

### Noisy + Dropout Run (`--max-frames 15 --noise-std 0.01 --drop-joint-prob 0.2`)

| Metric | Value |
| --- | ---: |
| MPJPE mean | `0.016836 m` |
| MPJPE max | `0.027094 m` |
| Dynamics residual mean | `0.000473` |
| q RMSE | `0.146309` |
| tau actuated RMSE | `191.693522` |
| Left contact F1 | `0.8333` |
| Right contact F1 | `0.8333` |

## Interpretation

- Reusing the same `cvxpy.Problem` instances does help.
- Full-sequence speed improved from about `23.9 FPS` to about `27.5 FPS`.
- Clean target tracking remains extremely accurate:
  - `MPJPE mean = 0.73 mm`
  - `MPJPE max = 10.7 mm`
- The implementation is still not at `50 Hz` real time, but it is closer than before.
- The remaining main bottlenecks are:
  - too many `cvxpy` parameters for ideal DPP compilation
  - contact / GRF estimation quality
  - occasional solver inaccuracy on some dynamic variables

## Overall Conclusion

Current state of the reused-problem `cvxpy` version:

- **Target following with clean 12-point global observations:** excellent
- **Long-horizon kinematic stability:** excellent
- **Runtime speed:** improved again
- **Contact / GRF quality:** still weak
- **Main next optimization target:** reduce parameter count in the reused QPs or drop one more level and reuse the low-level solver workspace directly

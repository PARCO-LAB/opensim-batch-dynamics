# AMASS Generalization Report (Updated Stage 2)

## Scope

- Dataset root: [data/AMASS](/Users/enricomartini/Desktop/opensim-batch-dynamics/data/AMASS)
- Evaluated implementation:
  - [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py)
  - [RT/real_time_test.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/real_time_test.py)
- Number of CSV files tested: `25`
- Successful runs: `25/25`
- Total batch wall time: `568.21 s`

## Aggregate View

| Metric | Previous | Current | Delta |
| --- | ---: | ---: | ---: |
| Weighted mean MPJPE (m) | `0.009188` | `0.006185` | `-0.003003` |
| Weighted mean q RMSE | `0.687344` | `0.666880` | `-0.020464` |
| Weighted mean tau actuated RMSE | `1151.795910` | `566.991273` | `-584.804637` |
| Weighted mean left-contact F1 | `0.309954` | `0.938433` | `+0.628479` |
| Weighted mean right-contact F1 | `0.325967` | `0.923520` | `+0.597553` |
| Weighted mean dyn residual | `-` | `0.003678` | `-` |
| Median file MPJPE | `0.002867` | `0.001364` | `-0.001503` |
| Mean wall time per file | `28.45 s` | `22.73 s` | `-5.72 s` |

## Interpretation

- The updated Stage 2 is materially better as a generalizable kinematic tracker and contact classifier.
- The biggest gains come from:
  - less conservative foot contact detection
  - causal support-force priors from COM dynamics
  - stronger `ddq` regularization to stop dynamic blow-ups
- The method is still not fully reliable as a torque/GRF estimator on every AMASS sequence.
- The remaining hard outlier is still `TotalCapture/rom1_stageii.csv`, though it is much less catastrophic than before.

## Best Cases

| File | MPJPE mean (m) | Tau actuated RMSE | Left F1 | Right F1 |
| --- | ---: | ---: | ---: | ---: |
| `BMLrub/0021_jumping2_stageii.csv` | `0.000135` | `305.23` | `0.945` | `0.940` |
| `GRAB/elephant_pass_1_stageii.csv` | `0.000139` | `291.97` | `0.993` | `1.000` |
| `BMLmovi/Subject_4_F_4_stageii.csv` | `0.000171` | `410.13` | `1.000` | `1.000` |
| `PosePrior/uar5_stageii.csv` | `0.000225` | `371.23` | `0.982` | `0.982` |
| `CMU/10_05_stageii.csv` | `0.000255` | `321.17` | `0.491` | `0.811` |

## Worst Cases

Worst kinematic tracking:

| File | MPJPE mean (m) | MPJPE max (m) | Tau actuated RMSE |
| --- | ---: | ---: | ---: |
| `CNRS/SW_B_3_stageii.csv` | `0.056047` | `1.065013` | `1413.06` |
| `WEIZMANN/Fast_SShapeLR(10)_stageii.csv` | `0.027213` | `0.167769` | `555.43` |
| `SFU/0012_JumpAndRoll001_stageii.csv` | `0.017066` | `0.191288` | `409.81` |
| `HUMAN4D/INF_Running_S2_01_stageii.csv` | `0.015441` | `0.190542` | `432.27` |
| `HumanEva/Box_1_stageii.csv` | `0.008610` | `0.088888` | `500.70` |

Worst dynamics:

| File | Tau actuated RMSE | MPJPE mean (m) | Left F1 | Right F1 |
| --- | ---: | ---: | ---: | ---: |
| `TotalCapture/rom1_stageii.csv` | `1930.82` | `0.002858` | `0.999` | `0.999` |
| `CNRS/SW_B_3_stageii.csv` | `1413.06` | `0.056047` | `0.774` | `0.764` |
| `220926_yogi_body_hands_03596_Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-c_stageii.csv` | `1016.86` | `0.004335` | `0.953` | `0.755` |
| `WEIZMANN/Fast_SShapeLR(10)_stageii.csv` | `555.43` | `0.027213` | `0.913` | `0.929` |
| `HumanEva/Box_1_stageii.csv` | `500.70` | `0.008610` | `0.954` | `0.958` |

## Per-File Results

| File | Frames | MPJPE mean (m) | Tau actuated RMSE | Left F1 | Right F1 | Dyn residual mean | Wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `220926_yogi_body_hands_03596_Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-c_stageii.csv` | `921` | `0.004335` | `1016.86` | `0.953` | `0.755` | `0.007434` | `31.73` |
| `ACCAD/C14_-__run_turn_right__(90)_stageii.csv` | `510` | `0.006060` | `330.35` | `0.915` | `0.904` | `0.001013` | `11.71` |
| `BMLhandball/Trial_upper_left_012_poses.csv` | `1004` | `0.000394` | `349.69` | `0.968` | `0.923` | `0.000970` | `20.32` |
| `BMLmovi/Subject_4_F_4_stageii.csv` | `236` | `0.000171` | `410.13` | `1.000` | `1.000` | `0.005301` | `6.09` |
| `BMLrub/0021_jumping2_stageii.csv` | `377` | `0.000135` | `305.23` | `0.945` | `0.940` | `0.003563` | `9.09` |
| `CMU/10_05_stageii.csv` | `435` | `0.000255` | `321.17` | `0.491` | `0.811` | `0.002035` | `10.61` |
| `CNRS/SW_B_3_stageii.csv` | `652` | `0.056047` | `1413.06` | `0.774` | `0.764` | `0.002164` | `18.47` |
| `DFAUST/50009_shake_hips_stageii.csv` | `250` | `0.001005` | `371.04` | `0.996` | `0.994` | `0.005266` | `8.55` |
| `EKUT/BEAMN02_stageii.csv` | `464` | `0.001364` | `328.83` | `0.828` | `0.881` | `0.003188` | `11.13` |
| `Eyes_Japan_Dataset/pose-08-pray_buddhismintone-kanno_stageii.csv` | `1999` | `0.003223` | `418.07` | `0.974` | `0.954` | `0.001962` | `42.20` |
| `GRAB/elephant_pass_1_stageii.csv` | `728` | `0.000139` | `291.97` | `0.993` | `1.000` | `0.001662` | `13.94` |
| `HDM05/HDM_dg_01-02_03_120_stageii.csv` | `1999` | `0.005273` | `378.62` | `0.948` | `0.947` | `0.001686` | `42.97` |
| `HUMAN4D/INF_Running_S2_01_stageii.csv` | `1935` | `0.015441` | `432.27` | `0.831` | `0.877` | `0.002564` | `39.22` |
| `HumanEva/Box_1_stageii.csv` | `1477` | `0.008610` | `500.70` | `0.954` | `0.958` | `0.002030` | `32.30` |
| `KIT/bend_left09_stageii.csv` | `498` | `0.001106` | `469.48` | `0.964` | `0.957` | `0.001750` | `11.73` |
| `LARa/L01_S03_R10.raw_stageii.csv` | `1999` | `0.000705` | `302.73` | `0.989` | `0.983` | `0.002671` | `41.86` |
| `MOSh/misc_stageii.csv` | `1999` | `0.005995` | `407.18` | `0.968` | `0.971` | `0.001426` | `38.48` |
| `PosePrior/uar5_stageii.csv` | `1999` | `0.000225` | `371.23` | `0.982` | `0.982` | `0.002046` | `39.12` |
| `SFU/0012_JumpAndRoll001_stageii.csv` | `428` | `0.017066` | `409.81` | `0.805` | `0.754` | `0.002121` | `11.91` |
| `SOMA/clap_002_stageii.csv` | `699` | `0.000490` | `387.24` | `1.000` | `1.000` | `0.001692` | `16.27` |
| `SSM/punch_kick_sync_stageii.csv` | `276` | `0.000401` | `338.00` | `0.980` | `0.696` | `0.005256` | `8.65` |
| `TCDHands/graspLumbricalB_cl_stageii.csv` | `600` | `0.000255` | `374.95` | `1.000` | `1.000` | `0.001835` | `13.78` |
| `TotalCapture/rom1_stageii.csv` | `1999` | `0.002858` | `1930.82` | `0.999` | `0.999` | `0.018457` | `55.35` |
| `Transitions/airkick_stand_stageii.csv` | `799` | `0.008506` | `500.10` | `0.872` | `0.561` | `0.003857` | `18.24` |
| `WEIZMANN/Fast_SShapeLR(10)_stageii.csv` | `503` | `0.027213` | `555.43` | `0.913` | `0.929` | `0.001817` | `14.50` |

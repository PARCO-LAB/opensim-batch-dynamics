# AMASS Full-Batch Evaluation

## Setup

- Dataset scope: `25` original AMASS CSV files under `data/AMASS`
- Total sequence frames evaluated per mode: `24786`
- Runtime environment: `conda run -n opensim-torque`
- Evaluated code:
  - [RT/rt_library.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/rt_library.py)
  - [RT/real_time_test.py](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/real_time_test.py)

## Aggregate Results

| Mode | Weighted MPJPE mean (m) | Weighted q RMSE | Weighted tau actuated RMSE | Weighted tau jerk RMSE | Weighted left F1 | Weighted right F1 | Weighted solve ms | Batch wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | `0.004694` | `0.536112` | `265.279520` | `94.123808` | `0.9557` | `0.9431` | `20.16` | `582.76 s` |
| Noise (`--noise-std 0.01`) | `0.018019` | `0.629619` | `284.066170` | `152.177417` | `0.9508` | `0.9405` | `26.22` | `732.95 s` |
| Dropout (`--drop-joint-prob 0.2`) | `0.006430` | `0.591666` | `270.101608` | `100.442736` | `0.9411` | `0.9344` | `19.14` | `557.02 s` |

## Readout

- `clean` remains best overall configuration.
- `noise` hurts both kinematics and torque more than `dropout` does.
- `dropout` is now computationally cheap enough because fixed-shape cached QPs stay reused across frames.
- Main failure case across all three modes remains `CNRS/SW_B_3_stageii.csv`.
- Main non-locomotion contact failure case remains `220926_yogi...stageii.csv`.

## Clean

| Metric | Weighted value |
| --- | ---: |
| MPJPE mean | `0.004694` |
| q RMSE | `0.536112` |
| tau actuated RMSE | `265.279520` |
| tau actuated jerk RMSE | `94.123808` |
| Dynamics residual mean | `0.000357` |
| Left contact F1 | `0.9557` |
| Right contact F1 | `0.9431` |
| Mean solve time / frame | `20.16 ms` |

Best `tau_actuated_rmse` files:
- `data/AMASS/SOMA/clap_002_stageii.csv`: `143.004`
- `data/AMASS/GRAB/elephant_pass_1_stageii.csv`: `156.975`
- `data/AMASS/TotalCapture/rom1_stageii.csv`: `166.630`
- `data/AMASS/BMLmovi/Subject_4_F_4_stageii.csv`: `170.439`
- `data/AMASS/LARa/L01_S03_R10.raw_stageii.csv`: `171.914`

Worst `tau_actuated_rmse` files:
- `data/AMASS/HUMAN4D/INF_Running_S2_01_stageii.csv`: `316.933`
- `data/AMASS/HDM05/HDM_dg_01-02_03_120_stageii.csv`: `322.006`
- `data/AMASS/SFU/0012_JumpAndRoll001_stageii.csv`: `354.272`
- `data/AMASS/220926_yogi_body_hands_03596_Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-c_stageii.csv`: `472.661`
- `data/AMASS/CNRS/SW_B_3_stageii.csv`: `1339.810`

Best `q_rmse` files:
- `data/AMASS/SOMA/clap_002_stageii.csv`: `0.152`
- `data/AMASS/BMLrub/0021_jumping2_stageii.csv`: `0.228`
- `data/AMASS/EKUT/BEAMN02_stageii.csv`: `0.296`
- `data/AMASS/TotalCapture/rom1_stageii.csv`: `0.333`
- `data/AMASS/BMLmovi/Subject_4_F_4_stageii.csv`: `0.344`

Worst `q_rmse` files:
- `data/AMASS/Eyes_Japan_Dataset/pose-08-pray_buddhismintone-kanno_stageii.csv`: `0.736`
- `data/AMASS/DFAUST/50009_shake_hips_stageii.csv`: `0.821`
- `data/AMASS/SFU/0012_JumpAndRoll001_stageii.csv`: `0.849`
- `data/AMASS/MOSh/misc_stageii.csv`: `1.084`
- `data/AMASS/KIT/bend_left09_stageii.csv`: `1.367`

## Noise (`--noise-std 0.01`)

| Metric | Weighted value |
| --- | ---: |
| MPJPE mean | `0.018019` |
| q RMSE | `0.629619` |
| tau actuated RMSE | `284.066170` |
| tau actuated jerk RMSE | `152.177417` |
| Dynamics residual mean | `0.000451` |
| Left contact F1 | `0.9508` |
| Right contact F1 | `0.9405` |
| Mean solve time / frame | `26.22 ms` |

Best `tau_actuated_rmse` files:
- `data/AMASS/SOMA/clap_002_stageii.csv`: `171.146`
- `data/AMASS/TotalCapture/rom1_stageii.csv`: `171.842`
- `data/AMASS/GRAB/elephant_pass_1_stageii.csv`: `189.047`
- `data/AMASS/WEIZMANN/Fast_SShapeLR(10)_stageii.csv`: `201.771`
- `data/AMASS/BMLrub/0021_jumping2_stageii.csv`: `204.786`

Worst `tau_actuated_rmse` files:
- `data/AMASS/HDM05/HDM_dg_01-02_03_120_stageii.csv`: `331.658`
- `data/AMASS/HUMAN4D/INF_Running_S2_01_stageii.csv`: `333.983`
- `data/AMASS/SFU/0012_JumpAndRoll001_stageii.csv`: `362.942`
- `data/AMASS/220926_yogi_body_hands_03596_Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-c_stageii.csv`: `472.270`
- `data/AMASS/CNRS/SW_B_3_stageii.csv`: `1341.454`

Best `q_rmse` files:
- `data/AMASS/PosePrior/uar5_stageii.csv`: `0.260`
- `data/AMASS/TCDHands/graspLumbricalB_cl_stageii.csv`: `0.269`
- `data/AMASS/HumanEva/Box_1_stageii.csv`: `0.323`
- `data/AMASS/BMLrub/0021_jumping2_stageii.csv`: `0.355`
- `data/AMASS/BMLmovi/Subject_4_F_4_stageii.csv`: `0.400`

Worst `q_rmse` files:
- `data/AMASS/SFU/0012_JumpAndRoll001_stageii.csv`: `0.932`
- `data/AMASS/Eyes_Japan_Dataset/pose-08-pray_buddhismintone-kanno_stageii.csv`: `1.030`
- `data/AMASS/GRAB/elephant_pass_1_stageii.csv`: `1.091`
- `data/AMASS/MOSh/misc_stageii.csv`: `1.350`
- `data/AMASS/KIT/bend_left09_stageii.csv`: `1.403`

## Dropout (`--drop-joint-prob 0.2`)

| Metric | Weighted value |
| --- | ---: |
| MPJPE mean | `0.006430` |
| q RMSE | `0.591666` |
| tau actuated RMSE | `270.101608` |
| tau actuated jerk RMSE | `100.442736` |
| Dynamics residual mean | `0.000380` |
| Left contact F1 | `0.9411` |
| Right contact F1 | `0.9344` |
| Mean solve time / frame | `19.14 ms` |

Best `tau_actuated_rmse` files:
- `data/AMASS/SOMA/clap_002_stageii.csv`: `142.968`
- `data/AMASS/GRAB/elephant_pass_1_stageii.csv`: `157.431`
- `data/AMASS/BMLmovi/Subject_4_F_4_stageii.csv`: `170.089`
- `data/AMASS/LARa/L01_S03_R10.raw_stageii.csv`: `177.158`
- `data/AMASS/Eyes_Japan_Dataset/pose-08-pray_buddhismintone-kanno_stageii.csv`: `179.668`

Worst `tau_actuated_rmse` files:
- `data/AMASS/HDM05/HDM_dg_01-02_03_120_stageii.csv`: `320.422`
- `data/AMASS/SSM/punch_kick_sync_stageii.csv`: `331.716`
- `data/AMASS/SFU/0012_JumpAndRoll001_stageii.csv`: `370.693`
- `data/AMASS/220926_yogi_body_hands_03596_Extended_Revolved_Side_Angle_Pose_or_Utthita_Parsvakonasana_-c_stageii.csv`: `472.937`
- `data/AMASS/CNRS/SW_B_3_stageii.csv`: `1329.547`

Best `q_rmse` files:
- `data/AMASS/SOMA/clap_002_stageii.csv`: `0.156`
- `data/AMASS/BMLmovi/Subject_4_F_4_stageii.csv`: `0.307`
- `data/AMASS/HumanEva/Box_1_stageii.csv`: `0.308`
- `data/AMASS/PosePrior/uar5_stageii.csv`: `0.349`
- `data/AMASS/TCDHands/graspLumbricalB_cl_stageii.csv`: `0.382`

Worst `q_rmse` files:
- `data/AMASS/DFAUST/50009_shake_hips_stageii.csv`: `0.824`
- `data/AMASS/SFU/0012_JumpAndRoll001_stageii.csv`: `0.875`
- `data/AMASS/ACCAD/C14_-__run_turn_right__(90)_stageii.csv`: `0.964`
- `data/AMASS/MOSh/misc_stageii.csv`: `1.201`
- `data/AMASS/KIT/bend_left09_stageii.csv`: `1.370`

## Conclusions

- System generalizes reasonably on most AMASS files in `clean`, with weighted `tau_actuated_rmse = 265.279520`.
- Moderate noise degrades weighted `tau_actuated_rmse` to `284.066170` and weighted `q_rmse` to `0.629619`.
- Random 20% keypoint dropout degrades weighted `tau_actuated_rmse` only to `270.101608`, which is notably milder than additive noise.
- Remaining work should target two persistent outlier families:
  - `CNRS/SW_B_3_stageii.csv` for shoulder/pelvis/trunk ambiguity
  - `220926_yogi...stageii.csv` for non-locomotion contact disambiguation

Raw batch data written to:
- [RT/amass_batch_results.json](/Users/enricomartini/Desktop/opensim-batch-dynamics/RT/amass_batch_results.json)

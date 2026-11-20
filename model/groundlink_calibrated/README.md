# Calibrated GroundLink GRF models

These coefficient files are least-squares fits using all six GroundLink subjects. They are the full-data deployment fits, not LOSO fold models. The evaluation used leave-one-subject-out folds; its analysis is under `pipeline_tuning/2026-11-17_improve` and `pipeline_tuning/2026-11-18_table1_hsymp` on the NAS.

Pass these paths to `--id-horizontal-split-model` and `--id-cop-progress-model` respectively when running `scripts/run_amass_to_bsm_csv.py`.

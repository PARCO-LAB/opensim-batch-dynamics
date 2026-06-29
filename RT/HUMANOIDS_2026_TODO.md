# Humanoids 2026 TODO

## Claim

- [ ] Formulare il contributo come metodo causale two-stage QP per ricostruzione dinamica full-body da keypoint 3D sparsi, rumorosi e incompleti.
- [ ] Evitare claim troppo forti tipo "ground-truth torques from video/AMASS"; usare "dynamic explanation" o "offline-reference reconstruction".
- [ ] Esplicitare che `real_time_test.py` e' il driver sperimentale; il metodo principale e' `qpid()` in `rt_library.py`.
- [ ] Definire chiaramente input e output: 12 keypoint 3D, `q`, `dq`, `ddq`, `tau`, foot wrenches, contacts, root residual.

## Criticita' Da Risolvere

- [ ] Rimuovere o giustificare il warm start da ground truth offline: testare inizializzazione realistica dai primi frame.
- [ ] Dichiarare il limite di identificabilita': da 12 keypoint non tutti i DOF, contatti e torques sono univoci.
- [ ] Motivare la metric mask che esclude ankle/head/wrist/pro_sup, oppure aggiungere risultati full-DOF in appendice.
- [ ] Rendere visibili i failure cases principali: `CNRS/SW_B_3_stageii.csv` e yoga/non-locomotion contact.
- [ ] Normalizzare le metriche dinamiche: torque in `Nm/kg`, GRF in body-weight, residual in body-weight o body-weight * height.

## Esperimenti AMASS

- [ ] Baseline clean su tutti i 25 CSV AMASS gia' usati.
- [ ] Noise sweep: `0`, `0.005`, `0.01`, `0.02`, `0.03` m.
- [ ] Dropout random sweep: `0`, `0.1`, `0.2`, `0.4`.
- [ ] Gap contigui: `100 ms`, `250 ms`, `500 ms` senza osservazioni su joint selezionati.
- [ ] Noise + gaps combinati: almeno `0.01 m + 250 ms gaps`.
- [ ] Dropout strutturato: piede mancante, braccio mancante, lato intero mancante.
- [ ] Ripetere stress test con piu' seed, almeno `5`, e riportare media + intervallo.
- [ ] Misurare runtime `mean`, `p95`, `p99`, e fallimenti solver.

## Ablation Minime

- [ ] Full method.
- [ ] Senza Stage 1 kinematic filter.
- [ ] Senza robust measurement weights.
- [ ] Senza geometric segment priors.
- [ ] Senza contact/support prior.
- [ ] Senza Stage 2 dynamics QP, usando solo inverse dynamics offline-style da cinematica filtrata.
- [ ] Riportare ablation su subset corto ma rappresentativo: nominale, running, yoga, CNRS outlier.

## CARE-PD

- [ ] Usare CARE-PD come applicazione/generalizzazione, non come validazione dinamica assoluta se non ci sono GRF/force plates.
- [ ] Estrarre metriche clinicamente leggibili: stance time, double support, left/right asymmetry, support force proxy, torque smoothness.
- [ ] Se disponibili score MDS-UPDRS, testare correlazioni semplici e dichiararle esplorative.
- [ ] Mostrare 2-3 sequenze qualitative: mild, moderate, severe.

## Figure E Tabelle

- [ ] Diagramma metodo: input keypoints -> Stage 1 robust IK -> Stage 2 dynamics/contact QP -> outputs.
- [ ] Tabella principale AMASS: clean/noise/dropout/gaps/runtime.
- [ ] Tabella ablation.
- [ ] Plot stress test: errore vs noise/gap length.
- [ ] Plot qualitativo: offline vs realtime per `q`, `tau`, GRF/contact su 2 sequenze.
- [ ] Figura failure case: dove e perche' fallisce CNRS/yoga.

## Paper

- [ ] Abstract: claim causale, sparse keypoints, robustezza a noise/gaps, runtime real-time.
- [ ] Introduction: problema realtime per humanoids, teleoperation/imitation e biomeccanica.
- [ ] Related work: offline IK/ID, QP whole-body dynamics, contact estimation, human-to-robot motion.
- [ ] Method: formulazione Stage 1, Stage 2, contact, residual root.
- [ ] Experiments: datasets, perturbazioni, baseline offline, ablation.
- [ ] Results: AMASS, CARE-PD case study, runtime, failure analysis.
- [ ] Limitations: pseudo-ground-truth offline, non-identificabilita', contatti senza force plates, tuning empirico.
- [ ] Conclusion: metodo realtime utilizzabile, ma non sostituto di misure dinamiche dirette.

## Prima Sottomissione

- [ ] Congelare branch/codice usato negli esperimenti.
- [ ] Salvare JSON/CSV grezzi per tutte le run.
- [ ] Generare script riproducibile per tutte le tabelle paper.
- [ ] Controllare licenze e citazioni per AMASS, CARE-PD, OpenSim, NimblePhysics.
- [ ] Preparare supplement con video e failure cases.

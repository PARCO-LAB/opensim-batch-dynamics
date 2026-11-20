# TODO - Pipeline automatizzata `AMASS (.npz) -> BSM (.osim/.mot/.csv)`

## Obiettivo

Costruire una pipeline completamente automatizzata che, dato un file AMASS in formato SMPL-X come `data/A3-_Swing_arms_stageii.npz`, produca:

- un modello OpenSim BSM scalato e marker-fit
- un file `.mot` della cinematica finale
- un file `.csv` con una riga per frame e colonne:
  - `frame`
  - `time`
  - `<dof>`
  - `<dof>_vel`
  - `<dof>_acc`

L'ordine dei DOF deve seguire il modello finale `bsm.osim`.

## Decisione architetturale raccomandata

Non conviene riusare come soluzione finale la pipeline OpenCap/LaiUhlrich che c'e` ora nella repo, perche':

- il target finale non e` `LaiUhlrich2022_torque_only.osim`, ma `bsm.osim`
- la logica marker e il modello anatomico sono diversi
- per ottenere velocita` e accelerazioni "precise" e` meglio usare il pass di smoothing di AddBiomechanics invece di derivare tutto da un `.mot` OpenSim standard

La soluzione consigliata e` una pipeline ibrida:

1. usare dalla repo `SMPL2AddBiomechanics` la parte BSM-specifica:
   - marker set BSM su SMPL-X
   - stima di `massKg` e `heightM` dai `betas`
   - convenzioni di input per AddBiomechanics
2. usare da `AddBiomechanics` solo l'engine locale headless:
   - kinematics fitting del modello BSM
   - marker offset optimization
   - acceleration minimizing pass
   - scrittura di `.osim` e `.mot`
3. usare questa repo come orchestratore:
   - parsing AMASS
   - forward SMPL-X
   - generazione markers TRC
   - creazione subject folder AddBiomechanics
   - esecuzione engine locale
   - export CSV finale con `pos/vel/acc`

## Riferimenti da riusare

### Da `SMPL2AddBiomechanics`

- `smpl2ab/data/bsm_markers_smplx.yaml`
- `smpl2ab/measurements/measurements.py`
- `smpl2ab/smpl2addbio.py`
- `smpl2ab/utils/smpl_utils.py`

Uso previsto:

- riusare la logica di stima antropometrica da SMPL-X
- riusare il marker dictionary BSM per SMPL-X
- non copiare l'intera repo come dipendenza runtime se non serve

### Da `AddBiomechanics`

- `server/engine/src/engine.py`
- `server/engine/src/kinematics_pass/subject.py`
- `server/engine/src/dynamics_pass/acceleration_minimizing_pass.py`
- `server/engine/src/writers/opensim_writer.py`

Uso previsto:

- riusare l'engine locale senza frontend, AWS o web app
- sfruttare il pass di accelerazione minimizzata
- usare i risultati finali dell'ultimo processing pass come sorgente per il CSV

## Vincoli e punti importanti

- L'input `A3-_Swing_arms_stageii.npz` e` SMPL-X, quindi bisogna usare il marker set `bsm_markers_smplx.yaml`, non quello SMPL/SMPL+H.
- `SMPL2AddBiomechanics` genera TRC e `_subject.json`, ma non basta da solo a produrre automaticamente `.osim` e `.mot` finali in locale.
- Il layout di output di `SMPL2AddBiomechanics` non coincide con il layout che l'engine locale di `AddBiomechanics` si aspetta.
- L'engine locale di `AddBiomechanics` legge i trial da cartelle `trials/<trial_name>/markers.trc`, non da file TRC piatti dentro `trials/`.
- Per usare un modello custom come BSM con `AddBiomechanics`, bisogna impostare `skeletonPreset: "custom"` e fornire `unscaled_generic.osim` nel subject folder.
- Per il caso AMASS senza GRF, il pipeline target deve essere kinematics-only + acceleration minimizing pass. La dynamics pass va disabilitata.

## Asset necessari

### Obbligatori

- `model/bsm/bsm.osim`
- `model/bsm/Geometry/*`
- `model/smpl/SMPLX_NEUTRAL.npz` oppure asset SMPL-X maschio/femmina aggiuntivi
- `assets/smpl2ab/bsm_markers_smplx.yaml`
- codice locale dell'engine di `AddBiomechanics`

### Facoltativi

- `SMPL2AddBiomechanics` completo come checkout di riferimento
- OSSO: non necessario per la prima versione

### Nota su licenze

- `SMPL2AddBiomechanics` e asset SKEL/BSM hanno vincoli di licenza non commerciali
- questi vincoli vanno documentati nel README finale della pipeline

## Struttura target della pipeline

Input:

- `data/A3-_Swing_arms_stageii.npz`

Workspace temporaneo AddBiomechanics:

```text
work/addbio/<subject_name>/
  _subject.json
  unscaled_generic.osim
  trials/
    A3-_Swing_arms_stageii/
      markers.trc
```

Output finali:

```text
outputs/bsm/<trial>/
  Models/
    match_markers_but_ignore_physics.osim
  IK/
    <trial>_ik.mot
  CSV/
    <trial>_bsm_dofs.csv
```

## File da aggiungere o modificare in questa repo

- `scripts/run_amass_to_bsm_csv.py`
- `src/opensim_batch_dynamics/bsm_assets.py`
- `src/opensim_batch_dynamics/bsm_markers.py`
- `src/opensim_batch_dynamics/bsm_subject_json.py`
- `src/opensim_batch_dynamics/addbio_subject_folder.py`
- `src/opensim_batch_dynamics/addbio_runner.py`
- `src/opensim_batch_dynamics/addbio_csv_export.py`
- `README.md`
- `environment.yml`

Possibili riusi diretti del codice gia` esistente:

- `src/opensim_batch_dynamics/amass_loader.py`
- `src/opensim_batch_dynamics/smplx_forward.py`
- `src/opensim_batch_dynamics/trc_export.py`

## Fase 1 - Preparazione asset e ambiente

- [ ] Aggiungere `model/bsm/bsm.osim` e la cartella `model/bsm/Geometry`.
- [ ] Copiare `bsm_markers_smplx.yaml` da `SMPL2AddBiomechanics` in `assets/smpl2ab/`.
- [ ] Scegliere come integrare AddBiomechanics:
  - opzione consigliata: checkout esterno configurato via env var `ADDBIO_ENGINE_ROOT`
  - alternativa: vendorizzare solo `server/engine/src` e `server/engine/Geometry`
- [ ] Aggiornare `environment.yml` con dipendenze coerenti per:
  - `smplx`
  - `torch`
  - `nimblephysics`
  - `numpy`
  - `scipy`
  - `pyyaml`
  - `trimesh`
  - `rtree`
  - `pandas` e `matplotlib` solo se necessari all'engine
- [ ] Verificare la compatibilita` della versione di `nimblephysics`.

Nota importante:

- `SMPL2AddBiomechanics` usa `nimblephysics==0.10.32`
- `AddBiomechanics/server/engine/requirements.txt` usa `nimblephysics==0.10.52.1`
- bisogna scegliere una versione unica testata sull'engine locale e adattare il codice marker/TRC di conseguenza

## Fase 2 - Parsing robusto di AMASS SMPL-X

- [ ] Riutilizzare `amass_loader.py` come loader base.
- [ ] Verificare che il file AMASS contenga almeno:
  - `trans`
  - `root_orient`
  - `pose_body`
  - `pose_hand`
  - `pose_jaw`
  - `pose_eye`
  - `betas`
  - `mocap_frame_rate`
  - `gender`
- [ ] Se mancano modelli `SMPLX_MALE/FEMALE`, decidere fallback controllato a `SMPLX_NEUTRAL.npz`.

## Fase 3 - Forward SMPL-X e marker BSM

- [ ] Riutilizzare `smplx_forward.py` per ottenere `vertices` e `joints`.
- [ ] Implementare `bsm_markers.py` che:
  - carica `bsm_markers_smplx.yaml`
  - mappa nome marker -> indice vertice SMPL-X
  - costruisce il tensore marker `(T, M, 3)`
- [ ] Aggiungere validazione:
  - ogni marker nel YAML deve esistere come vertice valido
  - nessun indice fuori range
- [ ] Esportare `markers.trc` con `trc_export.py` o con writer Nimble equivalente.

Output atteso di questa fase:

- un TRC BSM-consistent, non OpenCap-consistent

## Fase 4 - Stima antropometrica del soggetto

- [ ] Portare o riscrivere in `bsm_subject_json.py` la logica di `BodyMeasurements` da `SMPL2AddBiomechanics`.
- [ ] Calcolare da `betas`:
  - `massKg`
  - `heightM`
- [ ] Generare `_subject.json` con almeno questi campi:
  - `sex`
  - `massKg`
  - `heightM`
  - `subjectTags`
  - `skeletonPreset: "custom"`
  - `disableDynamics: true`
  - `runMoco: false`
  - `segmentTrials: false`
- [ ] Verificare che il JSON sia compatibile con `AddBiomechanics` locale.

Motivazione:

- con AMASS non ci sono GRF
- per il CSV finale servono cinematica e smoothing, non inverse dynamics o Moco

## Fase 5 - Creazione del subject folder per AddBiomechanics

- [ ] Implementare `addbio_subject_folder.py`.
- [ ] Creare automaticamente una cartella subject temporanea con layout compatibile con l'engine locale:

```text
<subject_root>/
  _subject.json
  unscaled_generic.osim
  trials/
    <trial_name>/
      markers.trc
```

- [ ] Copiare `model/bsm/bsm.osim` come `unscaled_generic.osim`.
- [ ] Se necessario, creare un symlink o una copia della cartella `Geometry` dove l'engine la possa trovare.
- [ ] Non usare il layout piatto di `SMPL2AddBiomechanics`, perche' non e` il formato che `engine.py` carica in locale.

## Fase 6 - Wrapper locale di AddBiomechanics

- [ ] Implementare `addbio_runner.py`.
- [ ] Invocare l'engine locale in modalita` headless sul subject folder creato.
- [ ] Catturare stdout/stderr e generare errori leggibili.
- [ ] Far fallire subito se l'engine produce `_errors.json`.
- [ ] Rendere configurabili:
  - root della checkout AddBiomechanics
  - output name
  - directory di lavoro temporanea

Output atteso di questa fase:

- `Models/match_markers_but_ignore_physics.osim`
- `IK/<trial>_ik.mot`
- eventuale `MarkerData/<trial>.trc`

## Fase 7 - Scelta della sorgente per posizioni, velocita` e accelerazioni

Scelta raccomandata:

- non derivare `vel` e `acc` dal `.mot` come primo approccio
- leggere direttamente il pass finale dell'engine AddBiomechanics

Perche':

- `AddBiomechanics` aggiunge un `acceleration minimizing pass`
- il codice dell'engine usa esplicitamente `getPoses()`, `getVels()` e `getAccs()` sul pass finale
- questa sorgente e` piu` pulita di un semplice Butterworth + differenze finite

Task:

- [ ] Implementare `addbio_csv_export.py`.
- [ ] Recuperare dal pass finale:
  - `poses`
  - `vels`
  - `accs`
- [ ] Estrarre l'ordine dei DOF dal modello finale `.osim`.
- [ ] Scrivere il CSV nell'ordine esatto dei DOF del modello finale.
- [ ] Tenere `.mot` come artefatto secondario per interoperabilita`.

Fallback solo se necessario:

- usare il `.mot` come sorgente posizioni
- applicare smoothing controllato
- derivare numericamente `vel` e `acc`

## Fase 8 - Definizione esplicita delle unita`

- [ ] Verificare in quali unita` `AddBiomechanics` espone `poses/vels/accs`.
- [ ] Definire un contratto unico per il CSV.

Scelta consigliata:

- traslazioni in metri, m/s, m/s^2
- rotazioni in radianti, rad/s, rad/s^2

- [ ] Se serve compatibilita` OpenSim "human-readable", aggiungere opzionalmente un export secondario in gradi.
- [ ] Documentare chiaramente le unita` nel README e nell'header del CSV, se possibile.

## Fase 9 - CLI finale utente

- [ ] Creare `scripts/run_amass_to_bsm_csv.py`.
- [ ] Argomenti minimi:
  - `--input`
  - `--trial`
  - `--output-dir`
  - `--bsm-model`
  - `--smplx-model-dir`
  - `--addbio-root`
- [ ] Argomenti utili:
  - `--sex`
  - `--keep-workdir`
  - `--csv-units si|degrees`
  - `--skip-engine`
  - `--reuse-subject-folder`

Comando target:

```bash
python scripts/run_amass_to_bsm_csv.py \
  --input data/A3-_Swing_arms_stageii.npz \
  --trial A3-_Swing_arms_stageii \
  --output-dir outputs/bsm \
  --bsm-model model/bsm/bsm.osim \
  --smplx-model-dir model/smpl \
  --addbio-root /path/to/AddBiomechanics
```

## Fase 10 - Validazione tecnica

- [ ] Testare la pipeline completa su `data/A3-_Swing_arms_stageii.npz`.
- [ ] Verificare che il subject folder venga creato correttamente.
- [ ] Verificare che l'engine locale produca:
  - modello finale `.osim`
  - file `.mot`
  - nessun `_errors.json`
- [ ] Verificare che il CSV:
  - abbia una riga per frame
  - abbia tutte le colonne del modello BSM
  - non includa DOF bloccati
  - non abbia NaN inattesi
- [ ] Verificare che `pos/vel/acc` siano smooth.
- [ ] Verificare che le posizioni del CSV coincidano con il pass finale usato per scrivere il `.mot`.
- [ ] Fare un sanity check visivo con Nimble o equivalente.

## Fase 11 - Metriche di qualita`

- [ ] Salvare nel report finale:
  - numero frame
  - numero DOF esportati
  - marker RMSE finale
  - unit system del CSV
  - sorgente di `vel/acc`:
    - `addbio_final_pass`
    - oppure `numerical_derivative_fallback`
- [ ] Aggiungere un file JSON di summary accanto al CSV.

## Fase 12 - README finale

- [ ] Aggiornare `README.md` per spiegare solo la pipeline BSM.
- [ ] Rimuovere o declassare a "legacy" la pipeline OpenCap/LaiUhlrich se non serve piu` al target.
- [ ] Documentare:
  - asset richiesti
  - limiti di licenza
  - setup Ubuntu/macOS
  - comando end-to-end
  - struttura output
  - unita` del CSV

## Rischi da gestire

- [ ] Mismatch tra versioni di `nimblephysics`.
- [ ] Asset BSM mancanti o con licenza non redistribuibile.
- [ ] Marker names nel YAML non perfettamente coerenti col `bsm.osim`.
- [ ] Difficolta` a lanciare l'engine locale di AddBiomechanics senza tutta l'infrastruttura cloud.
- [ ] Ambiguita` sulle unita` angolari tra output interni Nimble e file `.mot`.

## Criterio di completamento

La pipeline e` considerata completata quando, lanciando un solo comando su `data/A3-_Swing_arms_stageii.npz`, vengono prodotti automaticamente:

- `match_markers_but_ignore_physics.osim`
- `<trial>_ik.mot`
- `<trial>_bsm_dofs.csv`

e il CSV:

- usa l'ordine DOF del modello BSM finale
- contiene posizioni, velocita` e accelerazioni
- e` derivato dal pass finale filtrato di AddBiomechanics oppure da un fallback esplicitamente documentato
- supera un controllo qualitativo di smoothness e coerenza cinematica

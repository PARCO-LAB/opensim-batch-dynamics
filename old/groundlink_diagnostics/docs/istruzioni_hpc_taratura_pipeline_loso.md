# Taratura LOSO della pipeline GRF su GroundLink

## §0 Contesto

GroundLink produce stime cinematiche di forza di reazione al suolo (GRF) e coppia articolare. Qui `τ` indica il target ottenuto con inverse dynamics (ID) e carichi misurati; `τ̂` indica l'uscita ID della pipeline con GRF stimate. I target usati sono senza muscoli e hanno il CoP corretto con T_code. La correzione B2 è stata rifiutata; non riannotare AMASS.

Radici note:

```text
Cluster: /storage-exp-large/shared-folders/gr_bombieri/emartini/
NAS:     ~/nas/MAEVE/
Scratch locale: ~/hdd/groundlink-grf-tuning/
```

Sotto `GroundLink_torque/` si trovano `pipeline/`, `h1_labels/`, `metrics_masked_claude/eval_groundlink.py`, `torque_targets_12hz_nomuscles_copfix_tcode/`, `metrics_tau_claude/` e `pipeline_tuning/`. Le forze grezze sono in `GroundLink/force/`. La struttura locale conserva la stessa forma dei percorsi HPC per poter rimappare i `source_paths` dei metadati.

Assi e file: Z è verticale verso l'alto; `*_grf_measured.mot` usa il sistema della pipeline e il CoP corretto; `*_grf_estimated.mot` contiene le forze applicate ai punti `calcn`/`toes`; `contact_wrenches_estimated.csv` può essere a livello segmento o trial. Usare `run_metadata.json` per trovare il run giusto: `sidestretch_1` può avere più run e alcuni recovery non hanno il CSV dei wrench. I 206 target sono ripartiti s001–s006 con conteggi 33, 33, 36, 27, 35, 42.

Numeri baseline forniti da riprodurre:

| Gruppo | Valore |
| --- | ---: |
| Fz RMSE sui frame di contatto | 0,137 BW |
| Correlazione Fz sui frame di contatto | 0,831 |
| F1 contatto | 0,957 |
| Residuo rho | 0,010 BW |
| RMSE coppie medio per articolazione | 0,344 N·m/kg |
| Pearson r medio per articolazione | 0,547 (caviglia circa 0,36) |
| Split BW per coppie | 0,401 / 0,299 |
| Split CoM per coppie | 0,431 / 0,441 |
| Errore punto di applicazione | 4,01 cm |
| RMSE forza orizzontale | 0,093 BW |
| Falsi contatti | 8–25% per soggetto, secondo la definizione della nota baseline |

Il modulo GRF nella copia locale da verificare è `_estimate_contact_wrenches_from_kinematics` in `src/opensim_batch_dynamics/inverse_dynamics_no_grf.py`. La descrizione da verificare è: pavimento al 1° percentile; target `m(a − g)`; contatto con altezza ≤4 cm e velocità ≤0,5 m/s; riempimento buchi 0,03 s; durata minima 0,08 s; transizione 0,08 s; normale minima 0,25·m·g; esponente orizzontale 3; pesi α/(d+1e-3); cono di attrito μ=0,8 per 3 iterazioni; punto all'origine dei corpi proiettata; momento libero nullo; nessun bilancio del momento angolare. Quasi tutti i parametri sono costanti: prima di modificarli controllare commit e implementazione effettivi.

### Versione dei dati e del codice

Il checkout HPC usato per i risultati della pipeline è `/storage-large/homedirs/mrtnrc33/opensim-batch-dynamics`, ma non contiene metadati Git. L'hash del file sorgente usato corrispondeva alla copia di lavoro locale sporca (`d663…`); la versione pulita principale riportata era `b0e2dc…`. Il clone Git annidato era fermo a `3b037…` e non è la sorgente dei risultati. Questo task usa il branch `codex/groundlink-grf-tuning`, creato dal commit `d5b29a1`; non presumere che corrisponda al checkout HPC. Registrare gli hash dei file quando si eseguono nuovi job.

## §1 Regole

1. Individuare repository, commit e copia effettivamente usati per `pipeline/`; annotare differenze dalla descrizione sopra.
2. Lavorare sul branch nuovo `codex/groundlink-grf-tuning`; non fare push senza autorizzazione.
3. Ogni modifica allo stimatore deve essere un'opzione con default invariato. Con i default, confrontare i `contact_wrenches` esistenti con tolleranza massima 1e-6.
4. Scrivere risultati solo in una nuova `GroundLink_torque/pipeline_tuning/<run>/`; copiare sul NAS e verificare SHA-256. Non toccare cache esistenti né AMASS.
5. L'inferenza usa solo cinematica. Le pedane possono essere usate per creare target e valutare, mai per stimare le GRF.
6. Nei fold LOSO non selezionare parametri guardando il soggetto escluso. Eseguire i job lunghi con SLURM e conservare log e manifest.

## §2 Fase 1 — valutatore e baseline (gate di arresto)

Comando locale sulla copia di lavoro:

```bash
python scripts/evaluate_groundlink_table1.py \
  --torque-root ~/hdd/groundlink-grf-tuning/GroundLink_torque \
  --force-root ~/hdd/groundlink-grf-tuning/GroundLink/force \
  --output-dir ~/hdd/groundlink-grf-tuning/GroundLink_torque/pipeline_tuning/<run>
```

Il valutatore usa contatto misurato `Fz > 0,05 BW`, filtra la forza misurata con Butterworth 4° ordine zero-phase a 12 Hz da 250 Hz e usa la maschera di coerenza tra forza misurata e target cinematico (`|ΣFz_meas − Fz_target| ≤ 0,25 BW`). GRF è normalizzata con `subject_mass_kg` dal CSV della pipeline; coppie con la massa del modello. Le coppie si valutano solo nei frame validi in cui il piede omolaterale è in appoggio misurato. L'aggregazione è piede/articolazione → trial → soggetto → media e IC 95%. Lag è solo riportato, mai corretto spostando le serie.

Il gate passa solo se tutte le metriche baseline rientrano nelle tolleranze: 0,002 per i valori adimensionali e 0,05 cm per il CoP. Se fallisce, fermarsi e descrivere la metrica, la maschera e il denominatore che divergono; non avviare le fasi successive.

La cache cinematica per segmento (CoM, punti di rilevamento, origini dei corpi, massa, gravità) va aggiunta solo dopo il gate e deve consentire di rivalutare lo stimatore in NumPy senza ripetere IK o ID.

## §3 Fase 2 — scomposizione di τ̂

Riutilizzare l'ID con carichi alternativi e sostituire un componente per volta, mantenendo gli altri carichi della pipeline:

1. stato di contatto: azzerare la forza del piede senza contatto misurato e ridistribuire il totale sull'altro;
2. Fz per piede;
3. forza orizzontale per piede;
4. punto di applicazione;
5. tutti i componenti insieme, che devono ricostruire τ come controllo.

Salvare variazione di RMSE e r per ciascuna delle sei coordinate. Usare l'ordine dei risultati per scegliere le migliorie successive.

## §4 Fase 3 — migliorie GRF

Ogni variante è opzionale e confrontata prima singolarmente e poi in combinazione:

- **Contatto:** esporre altezza, velocità, riempimento buchi, durata minima, transizione e percentile del pavimento; valutare contatto distinto per tallone e dita.
- **Forza/ripartizione:** B1 dirige la componente orizzontale di ciascun piede verso il CoM in proporzione a `(p_CoM − p_i)/h_CoM` e la riscalda per rispettare il totale; B2 usa `dL/dt` cinematico in un piccolo QP per frame, con cono d'attrito e CoP nel poligono del piede.
- **CoP:** far avanzare il punto tallone→dita in base a contatto e inclinazione del piede.

Per ogni modifica, verificare la regressione dei wrench con i default entro 1e-6. Non correggere il lag spostando le serie.

## §5 Fase 4 — LOSO

Eseguire sei fold, ciascuno con cinque soggetti di training e uno di test. Dichiarare prima il budget fisso di griglia/ricerca casuale. Ottimizzare solo misure dirette: rapporto rispetto alla baseline di Fz RMSE di contatto, `1−F1`, errore del punto, RMSE orizzontale e falsa frazione di contatto. Le coppie non entrano nell'obiettivo; l'RMSE coppie sul training non può peggiorare. Riportare media±IC95 sui sei test, parametri per fold, risultati per attività e confronto appaiato con baseline. Calcolare le coppie di test con ID completa e i parametri del fold. Tarare parametri AMASS finali su tutti i sei soggetti; dichiarare performance solo dai fold LOSO.

## §6 Fase 5 — trasferibilità ad AMASS

Non riannotare né sovrascrivere AMASS. Misurare l'altezza minima dei piedi per sequenza: il pavimento non è normalizzato per il bug `vertical_offset` sull'asse Y. Eseguire la candidata finale su circa 20 sequenze di sottoinsiemi diversi e riportare stabilità, tempi, residui del bacino e frazione di frame in contatto, confrontando la baseline.

## §7 Consegna

In una nuova cartella `pipeline_tuning/<run>/` creare `report.md` nell'ordine: sintesi; versione codice/dati; baseline; scomposizione; ablation singole; LOSO; AMASS; raccomandazione e diff/test. Allegare `baseline_metrics.csv`, `decomposition.csv`, `ablation.csv`, `loso_folds.csv`, `loso_params.csv`, `amass_sample.csv`. Se il gate baseline fallisce, indicare chiaramente quali fasi e CSV non sono stati eseguiti; non fabbricare risultati vuoti.

## Stato del tentativo del 2026-11-16

La copia lenta dal NAS è stata sostituita con uno staging locale in `~/hdd/groundlink-grf-tuning/`; il trasferimento di baseline è partito da `mrtnrc33@hpc-master`. I 206 trial sono stati caricati. GRF, torque, CoP e forza orizzontale riproducono i valori forniti quando le coppie sono limitate allo stance misurato del piede corrispondente. La falsa frazione di contatto non ha una definizione che riproduca il range 8–25% per tutti i soggetti: il rapporto letterale `Fz_est >20 N` durante volo misurato per piede è 27,7–54,4%; la frazione su tutti i frame validi è 2,8–5,4%. Il gate si ferma su questo disaccordo. Decomposizione, tuning, LOSO e AMASS non vanno avviati finché il denominatore baseline non è riconciliato.

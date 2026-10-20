# Motion-X pipeline fix

Checklist per la riesecuzione HPC:

1. Usare il loader condiviso `load_all_amass_npz`: per ogni `.npy` sotto un
   componente di percorso `aist`, divide `trans[309:312]` per 94 e inverte la
   terza componente prima della conversione Y-up→Z-up. Il file sorgente non va
   modificato.
2. Usare il cutoff effettivo `min(cutoff, 0.8*Nyquist)` calcolato dal `.mot`;
   lo stesso valore deve arrivare alla conversione CSV e a OpenSim ID.
3. Il default SLURM dei contatti resta `all`; restringere a
   `calcn_l,toes_l,calcn_r,toes_r` solo passando esplicitamente l'opzione.
4. Esportare prima in `.csv.partial`, eseguire il quality gate e sostituire
   atomicamente il CSV solo con esito `ok`; conservare `.quality.json`.
5. Riutilizzare un CSV esistente solo se il sidecar dichiara `status: ok`.
   Il default SLURM non salta più i vecchi marker `previously_ok`.
6. Verificare la provenance: il manifest registra il commit git (`git_commit`)
   e il worker deve essere lanciato dallo stesso checkout.
7. Dopo il batch, eseguire:

   ```bash
   python scripts/audit_motionx_outputs.py <output-root> --json audit.json
   ```

   Controllare separatamente `joint_torque_rate_max` e `root_force_rate_max`.

Canary obbligatori: AIST `Dance_Ballet_Jazz_Airplane_clip_1`, un trial
`game_motion`, uno `HAA500`, più un controllo CMU/SFU. Per forzare la
riesecuzione usare `--no-skip-existing-csv`.

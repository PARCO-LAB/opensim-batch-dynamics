# AMASS -> OpenSim Torque-Only Pipeline

Questa repo contiene solo il necessario per convertire un file AMASS `.npz` in:

- `.mot` (cinematica inversa OpenSim)
- `.osim` (modello scalato torque-only, senza muscoli)
- `.csv` (frame x DOF del modello torque-only)

Il modello usato di default e `model/LaiUhlrich2022_torque_only.osim`.

## Struttura minima

- `scripts/run_amass_to_opensim.py`: entrypoint unico della pipeline
- `src/opensim_batch_dynamics/`: logica Python della pipeline
- `assets/opencap/`: XML e mapping marker necessari a scaling + IK
- `model/LaiUhlrich2022_torque_only.osim`: modello torque-actuated (no muscles)
- `model/Geometry/`: mesh OpenSim richieste dal modello
- `model/smpl/SMPLX_NEUTRAL.npz`: modello SMPL-X

## Setup Ubuntu (consigliato con micromamba)

### 1) Installa micromamba

```bash
mkdir -p ~/.local/bin
cd /tmp
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mv -f /tmp/bin/micromamba ~/.local/bin/micromamba
```

### 2) Crea environment

```bash
MAMBA_ROOT_PREFIX=$HOME/.micromamba ~/.local/bin/micromamba create -y -n opensim-torque \
  -c opensim-org -c conda-forge python=3.11 opensim=4.5.2 pip
```

### 3) Installa dipendenze Python

```bash
MAMBA_ROOT_PREFIX=$HOME/.micromamba ~/.local/bin/micromamba run -n opensim-torque \
  python -m pip install numpy torch smplx
```

## Esecuzione pipeline

```bash
MAMBA_ROOT_PREFIX=$HOME/.micromamba ~/.local/bin/micromamba run -n opensim-torque \
python scripts/run_amass_to_opensim.py \
  --input data/A3-_Swing_arms_stageii.npz \
  --trial demo_torque \
  --output-dir outputs \
  --mass-kg 75 \
  --height-m 1.75 \
  --sex neutral
```

## Output generati

- `outputs/MarkerData/<trial>/<trial>.trc`
- `outputs/OpenSim/Model/<trial>/LaiUhlrich2022_torque_only_scaled.osim`
- `outputs/OpenSim/IK/<trial>/<trial>.mot`
- `outputs/OpenSim/IK/<trial>/<trial>_dofs.csv`

Lo script stampa un JSON finale con i path reali dei file creati.

## Opzioni utili

- `--skip-scale`: salta lo scaling e usa direttamente `--model-path`
- `--skip-ik`: non genera `.mot` (quindi niente `.csv`)
- `--trc-only`: genera solo i TRC
- `--csv-path /path/file.csv`: path custom del CSV
- `--csv-model-path /path/model.osim`: modello usato per ordinare le colonne DOF nel CSV
- `--missing-fill 0`: valore per DOF mancanti nel `.mot` (default `nan`)

## Note pratiche

- Questa repo include solo `SMPLX_NEUTRAL.npz`. Se il tuo AMASS e male/female puoi comunque usare `--sex neutral`.
- Se OpenSim segnala mesh mancanti, verifica che `model/Geometry` sia presente accanto al modello `.osim`.

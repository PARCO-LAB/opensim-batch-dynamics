# MoCha Challenge Compatibility Plan

This document plans the risky path: run the full preprocessing stack inside the
CodaBench submission, then run a Torch classifier and return
`predictions[subject_id][walk_id] = predicted_class`.

The target challenge API is:

```python
def predict(data: dict) -> dict:
    ...
```

with hidden input:

```python
data[subject_id][walk_id] = {
    "pose": np.ndarray,   # shape (T, 72), SMPL axis-angle pose
    "trans": np.ndarray,  # shape (T, 3), global translation
    "beta": np.ndarray,   # shape (1, 10), all zeros for privacy
    "fps": int,
}
```

The output must be:

```python
predictions[subject_id][walk_id] = predicted_class
```

where `predicted_class` is one of `0, 1, 2, 3`.

## Hard Constraints

- The zip root must contain `run.py`.
- Runtime must work inside the official Docker evaluation environment.
- No package installation at submission time.
- Hidden test data must not be exported, scraped, or persisted beyond temporary
  files needed during `predict`.
- The submission must be robust to batches with multiple subjects and walks.
- The current CodaBench phase execution limit is 3600 seconds.
- The visible challenge Docker image is `codalab/codalab-legacy:gpu310`.
- The challenge input uses canonicalized SMPL, not AMASS SMPL-X.
- `beta` is all zeros, so subject-specific shape/mass inference from beta is
  not useful on hidden data.

## End-to-End Runtime Flow

```text
predict(data)
-> iterate subject_id/walk_id
-> validate SMPL arrays
-> SMPL dict -> in-memory SMPL take
-> SMPL -> SMPL-X sequence
-> SMPL-X -> BSM CSV/features
-> Torch model inference
-> predicted class
-> return nested predictions dict
```

The target implementation should avoid writing source hidden data to stable
locations. If native tools require files, write under a per-call temporary
directory and delete it before returning.

## Submission Layout

Target zip layout:

```text
submission.zip
  run.py
  mocha_runtime/
    __init__.py
    challenge_adapter.py
    smpl_to_smplx_runtime.py
    smplx_to_bsm_runtime.py
    bsm_feature_loader.py
    model_runtime.py
    runtime_checks.py
    third_party_path.py
  opensim_batch_dynamics/
    ...
  assets/
    smpl/
      SMPL_NEUTRAL.pkl
      SMPL_MALE.pkl                 # optional, if license allows
      SMPL_FEMALE.pkl               # optional, if license allows
      SMPLX_NEUTRAL.npz
      SMPLX_MALE.npz                # optional, if license allows
      SMPLX_FEMALE.npz              # optional, if license allows
    transfer_data/
      smpl2smplx_deftrafo_setup.pkl
      smplx_mask_ids.npy
    smpl2ab/
      bsm_markers_smplx.yaml
    bsm/
      bsm.osim
      Geometry/
        ...
    addbiomechanics/
      server/engine/src/
      server/engine/Geometry/
      ...
    model/
      classifier.pt
      model_config.json
  vendor/
    python/
      ...
    lib/
      ...
```

Do not assume this layout will fit CodaBench upload limits. A packaging audit is
an explicit milestone below.

## Phase 0 - Docker Recon

Goal: know exactly what the official Docker already contains.

Tasks:

1. Run `codalab/codalab-legacy:gpu310` locally.
2. Inspect Python version, CUDA availability, PyTorch version, glibc, system
   libraries, writable directories, memory, CPU count, and GPU visibility.
3. Run a probe `run.py` submission that imports only `numpy`, `torch`, `scipy`,
   `sklearn` if available, and prints versions.
4. Confirm whether CodaBench hides stdout/stderr in failed submissions. The API
   says outputs are hidden, so local reproduction is required.
5. Freeze the runtime target as a compatibility matrix:

```text
Python:
Torch:
CUDA:
NumPy:
SciPy:
OpenSim:
NimblePhysics:
smplx:
Upload size limit:
Max memory:
Max wall clock:
```

Acceptance criteria:

- A local Docker command can execute a minimal `predict(data)` call.
- The matrix is filled in this file or in a linked runtime report.
- We know whether native dependencies must be vendored.

## Phase 1 - Extract a Challenge Adapter

Goal: convert the CodaBench nested dict into internal per-walk records.

Add:

```text
mocha_runtime/challenge_adapter.py
```

Responsibilities:

- Validate required keys: `pose`, `trans`, `beta`, `fps`.
- Normalize dtypes to `np.float32`.
- Validate shapes:
  - `pose`: `(T, 72)`
  - `trans`: `(T, 3)`
  - `beta`: `(1, 10)` or `(10,)`, then flatten
  - `fps`: positive number
- Build a stable trial name from `subject_id` and `walk_id`.
- Keep original `subject_id` and `walk_id` unchanged for output keys.
- Reject invalid data by returning a deterministic fallback class, not by
  crashing the whole submission.

Internal record:

```python
@dataclass
class MochaWalk:
    subject_id: str
    walk_id: str
    trial_name: str
    poses: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    fps: float
    gender: str = "neutral"
```

Acceptance criteria:

- Unit tests cover valid input, missing keys, wrong shapes, empty sequences, and
  mixed `subject_id`/`walk_id` types.
- Adapter returns output keys in exactly the same nested structure.

## Phase 2 - In-Memory SMPL -> SMPL-X

Goal: produce the same semantic output as
`scripts/convert_carepd_smpl_pkl_to_smplx_npz.py`, without relying on input
pickle files.

Existing reference:

```text
scripts/convert_carepd_smpl_pkl_to_smplx_npz.py
```

Target output fields:

```python
surface_model_type = "smplx"
gender
mocap_frame_rate
trans
root_orient
pose_body
pose_hand
pose_jaw
pose_eye
betas
```

Implementation plan:

1. Move the reusable conversion code out of the script into a library module:

```text
src/opensim_batch_dynamics/smpl_to_smplx_transfer.py
```

2. Add a function that accepts a `MochaWalk` directly:

```python
def convert_mocha_smpl_to_smplx(
    walk: MochaWalk,
    runtime: SmplToSmplxRuntime,
) -> AMASSSequence:
    ...
```

3. Preserve the challenge `trans` field. The current mesh transfer helper uses
   zero translation when generating source vertices; for MoCha, root translation
   must be restored into the final SMPL-X `trans`.
4. Use `gender="neutral"` by default because challenge data does not provide
   gender and `beta` is zeroed.
5. Use `beta=np.zeros(10)` for target SMPL-X unless experiments show that
   transferring input zero beta directly is numerically better.
6. Cache expensive runtime objects globally inside `run.py` process lifetime:
   - SMPL neutral model
   - SMPL-X neutral model
   - deformation transfer matrix
   - mask ids
   - fitting assets
7. Avoid per-frame subprocess calls. The `classic` path is too slow and too
   fragile. Use the existing in-memory path as the base.

Speed options to evaluate:

- Reduce `trust-ncg` `maxiters` from `100` to `10`, `5`, and `1`.
- Warm-start SMPL-X poses from SMPL body pose:
  - root: copy SMPL root orient
  - body: copy SMPL 21 body joints into SMPL-X body pose
  - hands/face/eyes/jaw: zeros
- Skip optimization entirely for first baseline and use direct pose expansion:
  - `trans = input trans`
  - `root_orient = pose[:, :3]`
  - `pose_body = pose[:, 3:66]`
  - `pose_hand = zeros(T, 90)`
  - `pose_jaw = zeros(T, 3)`
  - `pose_eye = zeros(T, 6)`
  - `betas = zeros(10)`
- Benchmark direct expansion against fitted transfer on a small CARE-PD subset.

Important decision:

The exact official transfer model may be impossible within 3600 seconds for a
large hidden batch. The runtime should expose two modes:

```text
SMPL_TO_SMPLX_MODE=direct
SMPL_TO_SMPLX_MODE=fitted_fast
```

Use `direct` as the emergency default unless `fitted_fast` is proven under the
time budget.

Acceptance criteria:

- A single MoCha walk converts to an `AMASSSequence` without writing a source
  pickle.
- Output passes `AMASSSequence` shape validation.
- Runtime cache is initialized once per process.
- Direct mode takes less than 1 second per 1000 frames on local CPU.
- Fitted mode has measured wall-clock time per frame and a timeout guard.

## Phase 3 - SMPL-X -> BSM Runtime

Goal: produce the same final CSV/features as `scripts/run_amass_to_bsm_csv.py`
from an in-memory `AMASSSequence`.

Existing reference:

```text
scripts/run_amass_to_bsm_csv.py
```

Current pipeline:

```text
AMASSSequence
-> SMPL-X forward
-> BSM virtual markers
-> TRC
-> AddBiomechanics subject folder
-> AddBiomechanics engine
-> fitted model + IK mot
-> BSM dof CSV
-> optional inverse dynamics/GRF/contact/final CSV
```

Submission-oriented implementation:

1. Extract `_run_single_trial_pipeline` into a library function that accepts an
   `AMASSSequence` and a temporary output root.
2. Add a challenge runtime wrapper:

```text
mocha_runtime/smplx_to_bsm_runtime.py
```

3. Add a minimal mode for classification:

```text
BSM_EXPORT_MODE=kinematics
```

This mode should skip inverse dynamics and GRF estimation unless the trained
model explicitly needs torque/contact columns. The first compatible submission
should use BSM kinematics only:

```text
q, dq, ddq
```

4. Keep full mode available for experiments:

```text
BSM_EXPORT_MODE=full
```

Full mode runs inverse dynamics and final CSV export, but it is the most likely
part to exceed runtime.

5. Make AddBiomechanics root submission-local:

```python
ADDBIO_ENGINE_ROOT = Path(__file__).parent / "assets" / "addbiomechanics"
```

6. Set runtime environment before importing native stacks:

```python
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
```

7. Ensure all intermediate files are under `tempfile.TemporaryDirectory()`.
8. Delete intermediates after feature extraction.

Acceptance criteria:

- BSM kinematics mode works on a synthetic `AMASSSequence`.
- It does not require absolute local paths from the developer machine.
- It can run with `ADDBIO_ENGINE_ROOT` inside the unpacked submission.
- It returns an in-memory `pd.DataFrame` or `np.ndarray` features, not only a
  final CSV path.
- It has a per-walk timeout and deterministic fallback.

## Phase 4 - Native Dependency Packaging

Goal: make the risky stack import and run inside the official Docker without
network access.

Dependencies from current `environment.yml`:

```text
opensim=4.5.2
numpy<2
scipy>=1.10,<1.14
pyyaml
cvxpy
trimesh
rtree
shapely
tqdm
pandas
matplotlib
pytorch
smplx
nimblephysics
meshio
open3d
loguru
omegaconf
pyrender
```

Packaging strategy:

1. First try to rely on packages already present in `codalab/codalab-legacy:gpu310`.
2. For pure Python packages, vendor under `vendor/python` and prepend to
   `sys.path`.
3. For native wheels, build/download wheels matching the official Docker
   Python ABI and Linux platform, then vendor them if importable without
   installation.
4. For packages that require shared libraries, vendor the `.so` tree under
   `vendor/lib` and set:

```python
os.environ["LD_LIBRARY_PATH"] = vendor_lib + ":" + old_ld_library_path
```

5. OpenSim is the highest-risk dependency. Validate one of these approaches:
   - official Docker already has compatible OpenSim;
   - ship a conda-pack style environment and call its Python/site-packages;
   - remove OpenSim from challenge runtime by using only AddBiomechanics output
     kinematics and skipping inverse dynamics;
   - replace OpenSim-dependent CSV export with a lightweight `.mot` parser if
     model DOF ordering can be loaded without OpenSim.
6. `nimblephysics` and AddBiomechanics are the second highest-risk dependency.
   Validate import and a minimal engine run inside Docker before any model work.

Acceptance criteria:

- `python -c "import torch, smplx, nimblephysics, opensim"` succeeds in the
  local official Docker or the runtime has documented fallbacks for missing
  modules.
- A zipped submission can import `run.py` without modifying the container.
- Upload size is below the CodaBench limit.

## Phase 5 - Torch Model Interface

Goal: make the classifier swappable while the model is still untrained.

Add:

```text
mocha_runtime/model_runtime.py
```

Interface:

```python
class MochaClassifier:
    def __init__(self, weights_path: Path, config_path: Path):
        ...

    def predict_one(self, features: np.ndarray, meta: dict) -> int:
        ...
```

Initial placeholder:

- Load `assets/model/classifier.pt` if present.
- If no weights are present, use a deterministic fallback class, e.g. `0`.
- Clamp/validate output to `{0, 1, 2, 3}`.
- Use `torch.no_grad()`.
- Move tensors to CUDA only if the official Docker exposes GPU reliably.

Feature contract:

```text
input: BSM feature matrix, shape (T, D)
optional masks: valid frame mask, original fps, subject/walk ids
output: integer class
```

Acceptance criteria:

- `predict` works before the real model exists.
- Replacing `classifier.pt` and `model_config.json` does not require editing
  `run.py`.
- Invalid model output cannot break CodaBench scoring.

## Phase 6 - `run.py`

Goal: implement the challenge entrypoint.

Target control flow:

```python
_RUNTIME = None

def _get_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = MochaRuntime.from_submission_root(Path(__file__).parent)
    return _RUNTIME

def predict(data: dict) -> dict:
    runtime = _get_runtime()
    predictions = {}
    for subject_id, walks in data.items():
        predictions[subject_id] = {}
        for walk_id, payload in walks.items():
            try:
                walk = runtime.adapter.parse(subject_id, walk_id, payload)
                smplx_seq = runtime.smpl_to_smplx.convert(walk)
                features = runtime.smplx_to_bsm.extract_features(smplx_seq, walk)
                pred = runtime.model.predict_one(features, walk_meta)
            except Exception:
                pred = runtime.fallback_class
            predictions[subject_id][walk_id] = int(pred)
    return predictions
```

Rules:

- Do not initialize native stacks at module import if avoidable; initialize on
  first `predict`.
- Never let one failed walk abort the whole batch.
- Do not print large logs.
- Do not write hidden data outside a temp directory.
- Keep a compact per-walk diagnostic in memory during local tests, but disable
  verbose diagnostics in submission mode.

Acceptance criteria:

- `run.py` imports cleanly.
- `predict({})` returns `{}`.
- A mock input with two subjects and multiple walks returns the same nested
  structure.
- Returned classes are plain Python `int` values.

## Phase 7 - Performance Budget

Goal: prove the pipeline fits within hidden evaluation time.

Measure each stage:

```text
adapter parse:
SMPL -> SMPL-X direct:
SMPL -> SMPL-X fitted_fast:
SMPL-X forward:
marker extraction:
TRC write:
AddBiomechanics engine:
CSV/features load:
Torch inference:
cleanup:
```

Benchmarks:

- 100 frames
- 300 frames
- 1000 frames
- 10 walks in one batch
- worst known CARE-PD walking sequence length

Timeout policy:

- Per-walk timeout: configurable, e.g. 120 seconds.
- Whole-call soft budget: e.g. 3300 seconds to leave margin below 3600.
- If budget is exceeded, return fallback predictions for remaining walks.

Acceptance criteria:

- Direct SMPL-X mode plus BSM kinematics mode finishes under budget on local
  Docker for a representative batch.
- Fitted transfer is used only if its measured cost is acceptable.
- Fallback path is tested and returns valid predictions.

## Phase 8 - Accuracy Experiments

Goal: decide how much of the risky pipeline is actually useful.

Experiments:

1. Raw SMPL baseline:
   - features from `pose`, `trans`, velocities, accelerations, fps-normalized.
2. Direct SMPL-X expansion + BSM kinematics:
   - zero hands/face/eyes, copied root/body pose.
3. Fast fitted SMPL-X + BSM kinematics:
   - reduced optimizer iterations.
4. Full BSM with inverse dynamics:
   - only if runtime and dependencies are stable.

Use the same train/validation split for all experiments. Report Macro F1,
accuracy, and QWK. Because the challenge metric is Macro F1, optimize for class
balance and per-class recall rather than only accuracy.

Acceptance criteria:

- A trained model exists for at least one runtime-compatible feature mode.
- Model config records the exact feature mode used for training.
- Challenge runtime refuses to load weights trained for a different feature
  dimension or mode.

## Phase 9 - Local Challenge Harness

Goal: reproduce CodaBench calls locally.

Add:

```text
tests/mocha/mock_challenge_eval.py
```

The harness should:

- Build the nested `data[subject_id][walk_id]` dict from CARE-PD pickle samples.
- Import submission `run.py` exactly as CodaBench does.
- Call `predict(data)`.
- Validate nested output structure.
- Validate classes in `{0, 1, 2, 3}`.
- Measure wall-clock time and peak memory.
- Optionally score against labels when available.

Acceptance criteria:

- The harness can run inside the official Docker.
- It can run on a small fixture without external paths.
- It fails fast when packaged assets are missing.

## Phase 10 - Packaging Command

Goal: build the exact zip that will be uploaded.

Add:

```text
scripts/build_mocha_submission.py
```

Responsibilities:

- Create a clean staging directory.
- Copy `run.py`, `mocha_runtime`, required `opensim_batch_dynamics` modules,
  assets, model weights, and vendor dependencies.
- Exclude caches, test outputs, local datasets, `.DS_Store`, notebooks, large
  logs, and source-control metadata.
- Emit a manifest:

```text
submission_manifest.json
```

with file sizes, hashes, dependency versions, and selected runtime modes.

Acceptance criteria:

- Rebuilding the zip is deterministic.
- Zip root contains `run.py`.
- The zip passes the local challenge harness from a clean directory.
- Manifest flags missing required assets before upload.

## Phase 11 - Failure Handling

Required fallback behavior:

- Bad input shape: fallback class for that walk.
- SMPL->SMPL-X conversion failure: fallback class for that walk.
- AddBiomechanics failure: optional raw-SMPL fallback model if available,
  otherwise fallback class.
- Torch model failure: fallback class.
- Timeout: fallback class for current and remaining walks if necessary.

Preferred fallback class:

- Use the training-set majority class only if computed from allowed training
  data and stored in `model_config.json`.
- Otherwise use `0`.

Do not use hidden leaderboard feedback to tune fallback logic in a way that
violates submission limits or challenge rules.

## Phase 12 - Open Legal/License Questions

Before submission, confirm redistribution rights for:

- SMPL model files.
- SMPL-X model files.
- SMPL-X correspondence files:
  - `smpl2smplx_deftrafo_setup.pkl`
  - `smplx_mask_ids.npy`
- BSM model and geometry.
- AddBiomechanics code and assets.
- SMPL2AddBiomechanics marker maps or derived files.

If any asset cannot be redistributed in the CodaBench zip, the only viable
paths are:

- ask organizers to include the asset in the Docker image;
- ask organizers to approve a private submission mechanism;
- remove that stage from challenge runtime.

## Immediate Implementation Checklist

1. Create a local Docker reproduction of `codalab/codalab-legacy:gpu310`.
2. Write the mock challenge harness around the current scripts.
3. Extract SMPL dict parsing into `mocha_runtime/challenge_adapter.py`.
4. Refactor `convert_carepd_smpl_pkl_to_smplx_npz.py` into an importable
   in-memory converter.
5. Implement direct SMPL->SMPL-X expansion as the first runtime-compatible mode.
6. Refactor `run_amass_to_bsm_csv.py` into an importable in-memory feature
   extractor with `skip_inverse_dynamics=True`.
7. Package AddBiomechanics and test a minimal engine run inside Docker.
8. Add `run.py` with fallback-safe prediction structure.
9. Add placeholder Torch model runtime and deterministic fallback.
10. Build a submission zip and run it from a clean temp directory.
11. Train the first model only after feature dimensions and runtime mode are
    frozen.
12. Run final local Docker benchmark before spending a CodaBench submission.

## Definition of Done

The pipeline is challenge-compatible when all of the following are true:

- The submission zip imports `run.py` in the official Docker.
- `predict(data)` handles the exact nested CodaBench input format.
- SMPL input is converted to the selected SMPL-X representation without source
  pickle files.
- BSM features are produced from the SMPL-X sequence or the runtime falls back
  deterministically.
- Torch inference returns one integer class per walk.
- The output dictionary exactly mirrors `subject_id` and `walk_id` keys.
- Runtime stays below the challenge wall-clock limit on a representative batch.
- No runtime network access, package installation, or external local paths are
  required.
- All redistributable assets needed by the selected mode are inside the zip.

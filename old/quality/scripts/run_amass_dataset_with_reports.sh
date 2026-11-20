#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${1:-$REPO_ROOT/data/AMASS}"
WORKERS="${2:-2}"
CONDA_BIN="${CONDA_BIN:-/Users/enricomartini/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-opensim-torque}"
ADDBIO_ROOT="${ADDBIO_ROOT:-/Users/enricomartini/AddBiomechanics}"

cd "$REPO_ROOT" || exit 1

echo "[START] $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] repo=$REPO_ROOT"
echo "[INFO] input_root=$INPUT_ROOT"
echo "[INFO] workers=$WORKERS"
echo "[INFO] env=$ENV_NAME"
echo "[INFO] addbio_root=$ADDBIO_ROOT"
export INPUT_ROOT_ENV="$INPUT_ROOT"

BATCH_RC=0
"$CONDA_BIN" run -n "$ENV_NAME" python scripts/run_amass_batch_parallel.py \
  --input-root "$INPUT_ROOT" \
  --output-dir "$INPUT_ROOT" \
  --workers "$WORKERS" \
  --no-skip-existing-csv \
  --addbio-root "$ADDBIO_ROOT" \
  || BATCH_RC=$?

echo "[INFO] batch_rc=$BATCH_RC"

echo "[STEP] Generating PDFs for successful CSV outputs"
"$CONDA_BIN" run -n "$ENV_NAME" python - <<'PY'
from __future__ import annotations

import json
import subprocess
from pathlib import Path

repo_root = Path.cwd()
input_root = Path(__import__("os").environ.get("INPUT_ROOT_ENV", str(repo_root / "data" / "AMASS"))).resolve()
summary_path = input_root / "batch_summary.json"

pdf_failures: list[dict[str, str]] = []
created = 0
skipped = 0

npz_files = sorted(
    p for p in input_root.rglob("*.npz")
    if p.is_file() and p.name.lower() != "shape.npz"
)

for npz in npz_files:
    csv_path = npz.with_suffix(".csv")
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        skipped += 1
        continue

    pdf_path = csv_path.with_suffix(".pdf")
    cmd = [
        "python",
        str((repo_root / "scripts" / "csv_explorer.py").resolve()),
        "--input-csv",
        str(csv_path),
        "--output-pdf",
        str(pdf_path),
        "--title",
        csv_path.stem,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        pdf_failures.append(
            {
                "npz": str(npz),
                "csv": str(csv_path),
                "pdf": str(pdf_path),
                "cause": detail[-1] if detail else "Unknown PDF generation error",
            }
        )
    else:
        created += 1

pdf_summary_path = input_root / "pdf_summary.json"
pdf_summary_path.write_text(
    json.dumps(
        {
            "npz_total": len(npz_files),
            "pdf_created": created,
            "csv_missing_or_empty": skipped,
            "pdf_failed": len(pdf_failures),
            "failures": pdf_failures,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps({"pdf_summary": str(pdf_summary_path), "pdf_created": created, "pdf_failed": len(pdf_failures)}))

if summary_path.exists():
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
else:
    summary = {"results": []}

results = summary.get("results", [])
failed = [r for r in results if r.get("status") == "failed"]

def infer_fix(cause: str) -> str:
    c = cause.lower()
    if "no module named 'smplx'" in c or "smplx and torch are required" in c:
        return "Usa env Conda `opensim-torque` (o installa `smplx` + `torch`) e rilancia il file."
    if "addbiomechanics root is required" in c:
        return "Passa `--addbio-root /Users/enricomartini/AddBiomechanics` oppure imposta `ADDBIO_ENGINE_ROOT`."
    if "could not find addbiomechanics engine.py" in c:
        return "Verifica checkout AddBiomechanics completo e path `server/engine/src/engine.py`."
    if "missing final model output" in c or "missing final ik motion output" in c:
        return "AddBiomechanics non ha prodotto output finali: controlla `_errors.json` nella cartella trial e log engine."
    if "missing expected output csv" in c:
        return "Pipeline non ha scritto CSV finale: leggi log task e correggi errore precedente (dipendenze/engine/model)."
    if "unsupported amass layout" in c:
        return "Formato `.npz` non compatibile: converti al layout AMASS supportato (`poses/trans` o campi SMPL-X split)."
    return "Apri log indicato, identifica eccezione principale, poi correggi dipendenza/path/modello e riesegui solo quel file."

def extract_cause(log_path: Path) -> str:
    if not log_path.exists():
        return "Log mancante"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith(("Traceback", "COMMAND:")):
            continue
        if "Error" in ln or "Exception" in ln or "No module named" in ln or "FileNotFoundError" in ln:
            return ln
    return lines[-1] if lines else "Errore non determinato"

tofix_lines = [
    "# TOFIX",
    "",
    "Elenco file AMASS falliti durante batch. Per ogni file: causa sintetica + fix consigliato.",
    "",
]

if not failed and not pdf_failures:
    tofix_lines += ["Nessun errore rilevato.", ""]
else:
    if failed:
        tofix_lines += ["## Pipeline Failures", ""]
        for item in failed:
            rel = item.get("relative_path", "<unknown>")
            log_path = Path(item.get("log_path", ""))
            cause = extract_cause(log_path)
            fix = infer_fix(cause)
            npz_path = input_root / rel
            tofix_lines += [
                f"### {npz_path}",
                f"- Causa: {cause}",
                f"- Fix: {fix}",
                f"- Log: {log_path}",
                "",
            ]

    if pdf_failures:
        tofix_lines += ["## PDF Failures", ""]
        for item in pdf_failures:
            cause = item["cause"]
            fix = "Verifica CSV (header/righe) e rilancia `scripts/csv_explorer.py` su quel file."
            tofix_lines += [
                f"### {item['npz']}",
                f"- Causa: {cause}",
                f"- Fix: {fix}",
                f"- CSV: {item['csv']}",
                f"- PDF: {item['pdf']}",
                "",
            ]

tofix_path = repo_root / "TOFIX.md"
tofix_path.write_text("\n".join(tofix_lines), encoding="utf-8")
print(json.dumps({"tofix": str(tofix_path), "pipeline_failures": len(failed), "pdf_failures": len(pdf_failures)}))
PY

echo "[END] $(date '+%Y-%m-%d %H:%M:%S')"

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_INPUT_DIR = Path("/Volumes/MAEVE/HUMAN_MODEL/CARE-PD_torque")
DEFAULT_LABELS_CSV = Path("/Volumes/MAEVE/dataset/CARE-PD/carepd_mds_updrs_gait_severity.csv")
DEFAULT_OUTPUT_DIR = Path("/Volumes/MAEVE/dataset/CARE-PD/mocha_training_dataset")
DEFAULT_FOLDS_DIR = Path("/Volumes/MAEVE/dataset/CARE-PD/folds")

LABEL_FILENAME_HEADER = "filename"
LABEL_SCORE_HEADER = "MDS-UPDRS gait severity score"

METADATA_COLUMNS = {
    "frame",
    "time",
    "subject_mass_kg",
    "subject_height_m",
}


@dataclass(frozen=True)
class LabelRow:
    sample_id: str
    label: int


@dataclass(frozen=True)
class CsvRecord:
    sample_id: str
    path: Path
    label: int


@dataclass(frozen=True)
class ParsedSampleId:
    dataset_id: str
    subject_id: str
    walk_id: str


@dataclass
class RunningStats:
    sum: np.ndarray
    sumsq: np.ndarray
    count: np.ndarray

    @classmethod
    def create(cls, feature_count: int) -> "RunningStats":
        return cls(
            sum=np.zeros((feature_count,), dtype=np.float64),
            sumsq=np.zeros((feature_count,), dtype=np.float64),
            count=np.zeros((feature_count,), dtype=np.int64),
        )

    def update(self, values: np.ndarray) -> None:
        finite = np.isfinite(values)
        safe = np.where(finite, values.astype(np.float64, copy=False), 0.0)
        self.sum += np.sum(safe, axis=0)
        self.sumsq += np.sum(safe * safe, axis=0)
        self.count += np.sum(finite, axis=0)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.full(self.sum.shape, np.nan, dtype=np.float64)
        std = np.full(self.sum.shape, np.nan, dtype=np.float64)
        valid = self.count > 0
        mean[valid] = self.sum[valid] / self.count[valid]
        variance = np.zeros(self.sum.shape, dtype=np.float64)
        variance[valid] = self.sumsq[valid] / self.count[valid] - mean[valid] * mean[valid]
        variance = np.maximum(variance, 0.0)
        std[valid] = np.sqrt(variance[valid])
        std_safe = std.copy()
        bad = ~np.isfinite(std_safe) | (std_safe < 1e-8)
        std_safe[bad] = 1.0
        return mean.astype(np.float32), std.astype(np.float32), std_safe.astype(np.float32)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a MoCha-friendly CARE-PD training dataset from frame-wise "
            "OpenSim CSV files and MDS-UPDRS gait severity labels."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sequence-dir-name",
        default="sequences",
        help="Directory name under output-dir where per-sample .npz files are written.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.csv",
        help="Manifest filename under output-dir.",
    )
    parser.add_argument(
        "--summary-name",
        default="summary.json",
        help="Build summary filename under output-dir.",
    )
    parser.add_argument(
        "--norm-stats-name",
        default="norm_stats.npz",
        help="Normalization statistics filename under output-dir.",
    )
    parser.add_argument(
        "--on-feature-mismatch",
        choices=("error", "skip", "fill"),
        default="error",
        help=(
            "Behavior when a CSV feature schema differs from the first valid sample. "
            "'fill' uses the first schema and fills missing columns with NaN."
        ),
    )
    parser.add_argument(
        "--split-source",
        choices=("random", "carepd-folds"),
        default="random",
        help=(
            "Use random train/val/test split controls, or TaatiTeam/CARE-PD "
            "participant folds from --folds-dir."
        ),
    )
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument(
        "--carepd-fold-set",
        choices=("fixed", "6fold", "loso"),
        default="fixed",
        help=(
            "Which CARE-PD fold files to use. fixed uses *_fixed.pkl, 6fold uses "
            "*_6fold_participants.pkl, loso uses the dataset-specific N-fold "
            "participants files."
        ),
    )
    parser.add_argument(
        "--carepd-fold-index",
        type=int,
        default=1,
        help="Fold key to read from the CARE-PD fold pickle.",
    )
    parser.add_argument(
        "--carepd-eval-split-name",
        default="eval",
        help="Split label to write in the manifest for the CARE-PD 'eval' partition.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Optional validation split ratio. Splits are assigned by subject by default.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Optional test split ratio. Splits are assigned by subject by default.",
    )
    parser.add_argument(
        "--split-by",
        choices=("subject", "sample"),
        default="subject",
        help="Assign splits by subject_id to reduce leakage, or by individual sample.",
    )
    parser.add_argument("--split-seed", type=int, default=13)
    parser.add_argument(
        "--norm-split",
        choices=("train", "all"),
        default="train",
        help="Which samples are used to compute normalization statistics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on matched CSV files, useful for smoke tests.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing per-sample .npz files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read schemas and labels, but do not write outputs.",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed for sequence files.",
    )
    parser.add_argument(
        "--keep-unmatched-report-limit",
        type=int,
        default=50,
        help="Maximum number of unmatched IDs to include in summary.json.",
    )
    return parser.parse_args(argv)


def _sniff_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return ";"
    return dialect.delimiter


def _normalize_label_key(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", raw.lower())


def _find_label_field(fieldnames: Sequence[str], target: str) -> str:
    normalized_target = _normalize_label_key(target)
    for field in fieldnames:
        if _normalize_label_key(field) == normalized_target:
            return field
    raise ValueError(f"Missing expected label CSV column {target!r}. Found: {list(fieldnames)}")


def _coerce_label(raw: Any, sample_id: str) -> int:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid label for {sample_id}: {raw!r}") from exc
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"Invalid non-integer label for {sample_id}: {raw!r}")
    label = int(value)
    if label < 0 or label > 3:
        raise ValueError(f"Label out of MoCha range 0..3 for {sample_id}: {label}")
    return label


def load_labels(path: Path) -> dict[str, LabelRow]:
    labels_csv = path.expanduser().resolve()
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    delimiter = _sniff_delimiter(labels_csv)
    labels: dict[str, LabelRow] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"Labels CSV has no header: {labels_csv}")
        filename_field = _find_label_field(reader.fieldnames, LABEL_FILENAME_HEADER)
        score_field = _find_label_field(reader.fieldnames, LABEL_SCORE_HEADER)
        for row in reader:
            sample_id = str(row.get(filename_field, "")).strip()
            if not sample_id:
                continue
            if sample_id in labels:
                raise ValueError(f"Duplicate label for sample_id {sample_id!r}")
            labels[sample_id] = LabelRow(sample_id=sample_id, label=_coerce_label(row.get(score_field), sample_id))
    if not labels:
        raise ValueError(f"No labels loaded from {labels_csv}")
    return labels


def discover_csv_records(input_dir: Path, labels: dict[str, LabelRow], limit: int | None = None) -> tuple[list[CsvRecord], list[str], list[str]]:
    resolved_input = input_dir.expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input directory not found: {resolved_input}")

    all_csv = sorted(path for path in resolved_input.rglob("*.csv") if path.is_file())
    records: list[CsvRecord] = []
    csv_without_label: list[str] = []
    seen_sample_ids: set[str] = set()

    max_records = None if limit is None else max(0, limit)
    for csv_path in all_csv:
        sample_id = csv_path.stem
        seen_sample_ids.add(sample_id)
        label_row = labels.get(sample_id)
        if label_row is None:
            csv_without_label.append(sample_id)
            continue
        if max_records is None or len(records) < max_records:
            records.append(CsvRecord(sample_id=sample_id, path=csv_path, label=label_row.label))

    label_without_csv = sorted(sample_id for sample_id in labels if sample_id not in seen_sample_ids)
    return records, sorted(csv_without_label), label_without_csv


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc
    if not header:
        raise ValueError(f"CSV has empty header: {path}")
    return [name.strip() for name in header]


def _is_non_motion_base_column(column: str) -> bool:
    if column in METADATA_COLUMNS:
        return True
    if column.startswith("grf_total_"):
        return True
    if "_grf_" in column or column.endswith("_contact"):
        return True
    if column.endswith("_vel") or column.endswith("_acc") or column.endswith("_tau"):
        return True
    if column.endswith("_scale_x") or column.endswith("_scale_y") or column.endswith("_scale_z"):
        return True
    return False


def discover_motion_bases(header: Sequence[str]) -> list[str]:
    columns = set(header)
    bases: list[str] = []
    for column in header:
        if _is_non_motion_base_column(column):
            continue
        if f"{column}_vel" in columns and f"{column}_acc" in columns:
            bases.append(column)
    if not bases:
        raise ValueError("No position/velocity/acceleration feature triples found in CSV header.")
    return bases


def build_feature_names(bases: Sequence[str]) -> list[str]:
    names: list[str] = []
    for base in bases:
        names.extend([base, f"{base}_vel", f"{base}_acc"])
    return names


def read_sequence_csv(path: Path, feature_names: Sequence[str], fill_missing: bool = False) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [name.strip() for name in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"CSV is empty: {path}") from exc

        column_to_idx = {name: idx for idx, name in enumerate(header)}
        missing_features = [name for name in feature_names if name not in column_to_idx]
        if missing_features and not fill_missing:
            raise ValueError(
                f"CSV is missing {len(missing_features)} expected feature columns: "
                f"{path}. First missing: {missing_features[:5]}"
            )

        feature_indices = [column_to_idx.get(name) for name in feature_names]
        time_idx = column_to_idx.get("time")
        rows: list[list[float]] = []
        time_values: list[float] = []

        for row in reader:
            if not row:
                continue
            values: list[float] = []
            for idx in feature_indices:
                if idx is None or idx >= len(row):
                    values.append(math.nan)
                else:
                    text = row[idx].strip()
                    try:
                        values.append(float(text) if text else math.nan)
                    except ValueError:
                        values.append(math.nan)
            rows.append(values)
            if time_idx is None or time_idx >= len(row):
                time_values.append(float(len(time_values)))
            else:
                text = row[time_idx].strip()
                try:
                    time_values.append(float(text) if text else math.nan)
                except ValueError:
                    time_values.append(math.nan)

    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    x = np.asarray(rows, dtype=np.float32)
    time = np.asarray(time_values, dtype=np.float32)
    return x, time


def infer_fps(time_values: np.ndarray) -> float:
    if time_values.size < 2:
        return math.nan
    diffs = np.diff(time_values.astype(np.float64, copy=False))
    valid = diffs[np.isfinite(diffs) & (diffs > 0)]
    if valid.size == 0:
        return math.nan
    dt = float(np.median(valid))
    if dt <= 0:
        return math.nan
    return 1.0 / dt


def parse_sample_id(sample_id: str) -> ParsedSampleId:
    parts = sample_id.split("__")
    if len(parts) >= 3:
        return ParsedSampleId(dataset_id=parts[0], subject_id=parts[1], walk_id="__".join(parts[2:]))
    if len(parts) == 2:
        return ParsedSampleId(dataset_id=parts[0], subject_id=parts[1], walk_id=sample_id)
    return ParsedSampleId(dataset_id="", subject_id=sample_id, walk_id=sample_id)


def normalize_carepd_dataset_id(dataset_id: str) -> str:
    normalized = dataset_id
    if normalized.endswith("_canonical"):
        normalized = normalized[: -len("_canonical")]
    return normalized


def safe_npz_name(sample_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._-")
    if cleaned:
        return f"{cleaned}.npz"
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    return f"sample_{digest}.npz"


def build_split_map(records: Sequence[CsvRecord], split_by: str, val_ratio: float, test_ratio: float, seed: int) -> dict[str, str]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("--val-ratio and --test-ratio must be >= 0 and sum to less than 1.")

    if split_by == "subject":
        keys = sorted({parse_sample_id(record.sample_id).subject_id for record in records})
    else:
        keys = sorted(record.sample_id for record in records)

    rng = random.Random(seed)
    rng.shuffle(keys)

    n_keys = len(keys)
    n_test = int(round(n_keys * test_ratio))
    n_val = int(round(n_keys * val_ratio))
    if n_test + n_val > n_keys:
        n_val = max(0, n_keys - n_test)

    split_map: dict[str, str] = {}
    for idx, key in enumerate(keys):
        if idx < n_test:
            split_map[key] = "test"
        elif idx < n_test + n_val:
            split_map[key] = "val"
        else:
            split_map[key] = "train"
    return split_map


def _carepd_fold_dataset_name(path: Path) -> str | None:
    stem = path.stem
    suffixes = (
        "_authors_fixed",
        "_PD_fixed",
        "_fixed",
        "_6fold_participants",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]

    match = re.match(r"^(?P<dataset>.+)_\d+fold_participants$", stem)
    if match:
        return match.group("dataset")
    return None


def _fold_file_rank(path: Path, fold_set: str) -> tuple[int, str]:
    stem = path.stem
    if fold_set == "fixed":
        if stem.endswith("_authors_fixed"):
            return (0, stem)
        if stem.endswith("_PD_fixed"):
            return (1, stem)
        if stem.endswith("_fixed"):
            return (2, stem)
        return (99, stem)
    if fold_set == "6fold":
        return (0 if stem.endswith("_6fold_participants") else 99, stem)
    match = re.match(r"^.+_(?P<n>\d+)fold_participants$", stem)
    if not match:
        return (99, stem)
    fold_count = int(match.group("n"))
    return (-fold_count, stem)


def discover_carepd_fold_files(folds_dir: Path, fold_set: str, dataset_ids: set[str]) -> dict[str, Path]:
    resolved = folds_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"CARE-PD folds directory not found: {resolved}")

    candidates: dict[str, list[Path]] = {dataset_id: [] for dataset_id in dataset_ids}
    for path in sorted(resolved.rglob("*.pkl")):
        dataset_name = _carepd_fold_dataset_name(path)
        if dataset_name not in dataset_ids:
            continue
        stem = path.stem
        if fold_set == "fixed" and not stem.endswith("_fixed"):
            continue
        if fold_set == "6fold" and not stem.endswith("_6fold_participants"):
            continue
        if fold_set == "loso":
            if not re.match(r"^.+_\d+fold_participants$", stem):
                continue
            if stem.endswith("_6fold_participants"):
                continue
        candidates[dataset_name].append(path)

    selected: dict[str, Path] = {}
    for dataset_id, paths in candidates.items():
        if not paths:
            raise ValueError(
                f"No CARE-PD {fold_set!r} fold file found for dataset {dataset_id!r} under {resolved}"
            )
        selected[dataset_id] = sorted(paths, key=lambda path: _fold_file_rank(path, fold_set))[0]
    return selected


def load_carepd_fold_split_map(
    records: Sequence[CsvRecord],
    folds_dir: Path,
    fold_set: str,
    fold_index: int,
    eval_split_name: str,
) -> tuple[dict[str, str], dict[str, str]]:
    dataset_ids = {
        normalize_carepd_dataset_id(parse_sample_id(record.sample_id).dataset_id)
        for record in records
    }
    fold_files = discover_carepd_fold_files(folds_dir, fold_set=fold_set, dataset_ids=dataset_ids)
    fold_participants: dict[str, dict[str, set[str]]] = {}

    for dataset_id, path in fold_files.items():
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if fold_index not in payload:
            available = sorted(payload.keys())
            raise ValueError(
                f"Fold {fold_index} not found in {path}. Available folds: {available[:20]}"
            )
        fold_payload = payload[fold_index]
        if not isinstance(fold_payload, dict) or "train" not in fold_payload or "eval" not in fold_payload:
            raise ValueError(f"Unexpected CARE-PD fold structure in {path} fold {fold_index}")
        train_subjects = {str(value) for value in fold_payload["train"]}
        eval_subjects = {str(value) for value in fold_payload["eval"]}
        overlap = train_subjects & eval_subjects
        if overlap:
            raise ValueError(f"Train/eval subject overlap in {path}: {sorted(overlap)[:10]}")
        fold_participants[dataset_id] = {
            "train": train_subjects,
            "eval": eval_subjects,
        }

    split_map: dict[str, str] = {}
    for record in records:
        parsed = parse_sample_id(record.sample_id)
        dataset_id = normalize_carepd_dataset_id(parsed.dataset_id)
        subject_id = str(parsed.subject_id)
        split = fold_participants[dataset_id]
        if subject_id in split["train"]:
            split_map[record.sample_id] = "train"
        elif subject_id in split["eval"]:
            split_map[record.sample_id] = eval_split_name
        else:
            fold_file = fold_files[dataset_id]
            raise ValueError(
                f"Subject {subject_id!r} from {record.sample_id!r} not found in "
                f"CARE-PD fold file {fold_file} fold {fold_index}"
            )

    return split_map, {dataset_id: str(path) for dataset_id, path in sorted(fold_files.items())}


def label_histogram(labels: Sequence[int]) -> dict[str, int]:
    hist = {str(i): 0 for i in range(4)}
    for label in labels:
        hist[str(label)] = hist.get(str(label), 0) + 1
    return hist


def write_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty manifest.")
    fieldnames = [
        "sample_id",
        "dataset_id",
        "subject_id",
        "walk_id",
        "label",
        "split",
        "n_frames",
        "fps",
        "duration_s",
        "feature_count",
        "input_npz",
        "source_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _maybe_write_sequence(
    path: Path,
    compress: bool,
    force: bool,
    **payload: Any,
) -> None:
    if path.exists() and not force:
        return
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    labels = load_labels(args.labels_csv)
    records, csv_without_label, label_without_csv = discover_csv_records(args.input_dir, labels, args.limit)
    if not records:
        raise ValueError("No CSV records matched labels.")

    output_dir = args.output_dir.expanduser().resolve()
    sequence_dir = output_dir / args.sequence_dir_name
    manifest_path = output_dir / args.manifest_name
    summary_path = output_dir / args.summary_name
    norm_stats_path = output_dir / args.norm_stats_name

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        sequence_dir.mkdir(parents=True, exist_ok=True)

    expected_feature_names: list[str] | None = None
    manifest_rows: list[dict[str, Any]] = []
    skipped_feature_mismatch: list[dict[str, Any]] = []
    failed_samples: list[dict[str, Any]] = []
    carepd_fold_files: dict[str, str] = {}
    if args.split_source == "carepd-folds":
        split_map, carepd_fold_files = load_carepd_fold_split_map(
            records,
            folds_dir=args.folds_dir,
            fold_set=args.carepd_fold_set,
            fold_index=args.carepd_fold_index,
            eval_split_name=args.carepd_eval_split_name,
        )
    else:
        split_map = build_split_map(
            records,
            split_by=args.split_by,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
    stats: RunningStats | None = None

    for index, record in enumerate(records, start=1):
        header = read_header(record.path)
        bases = discover_motion_bases(header)
        feature_names = build_feature_names(bases)

        if expected_feature_names is None:
            expected_feature_names = feature_names
            stats = RunningStats.create(len(expected_feature_names))
        elif feature_names != expected_feature_names:
            mismatch = {
                "sample_id": record.sample_id,
                "csv": str(record.path),
                "expected_feature_count": len(expected_feature_names),
                "actual_feature_count": len(feature_names),
            }
            if args.on_feature_mismatch == "error":
                raise ValueError(
                    f"Feature schema mismatch for {record.sample_id}. "
                    f"Expected {len(expected_feature_names)} columns, got {len(feature_names)}. "
                    "Use --on-feature-mismatch skip or fill to continue."
                )
            if args.on_feature_mismatch == "skip":
                skipped_feature_mismatch.append(mismatch)
                continue
            skipped_feature_mismatch.append({**mismatch, "mode": "fill"})

        assert expected_feature_names is not None
        assert stats is not None

        try:
            x, time = read_sequence_csv(
                record.path,
                expected_feature_names,
                fill_missing=args.on_feature_mismatch == "fill",
            )
        except Exception as exc:
            failed_samples.append({"sample_id": record.sample_id, "csv": str(record.path), "error": str(exc)})
            continue

        parsed = parse_sample_id(record.sample_id)
        split_key = (
            record.sample_id
            if args.split_source == "carepd-folds"
            else (parsed.subject_id if args.split_by == "subject" else record.sample_id)
        )
        split = split_map[split_key]
        fps = infer_fps(time)
        duration_s = float(time[-1] - time[0]) if time.size > 1 and np.isfinite(time[[0, -1]]).all() else math.nan
        npz_relpath = Path(args.sequence_dir_name) / safe_npz_name(record.sample_id)
        npz_path = output_dir / npz_relpath

        if args.norm_split == "all" or split == "train":
            stats.update(x)

        if not args.dry_run:
            _maybe_write_sequence(
                npz_path,
                compress=not args.no_compress,
                force=args.force,
                x=x,
                time=time,
                label=np.asarray(record.label, dtype=np.int64),
                sample_id=np.asarray(record.sample_id),
                dataset_id=np.asarray(parsed.dataset_id),
                subject_id=np.asarray(parsed.subject_id),
                walk_id=np.asarray(parsed.walk_id),
                fps=np.asarray(fps, dtype=np.float32),
                feature_names=np.asarray(expected_feature_names),
                source_csv=np.asarray(str(record.path)),
            )

        manifest_rows.append(
            {
                "sample_id": record.sample_id,
                "dataset_id": parsed.dataset_id,
                "subject_id": parsed.subject_id,
                "walk_id": parsed.walk_id,
                "label": record.label,
                "split": split,
                "n_frames": int(x.shape[0]),
                "fps": "" if not np.isfinite(fps) else f"{fps:.8g}",
                "duration_s": "" if not np.isfinite(duration_s) else f"{duration_s:.8g}",
                "feature_count": int(x.shape[1]),
                "input_npz": npz_relpath.as_posix(),
                "source_csv": str(record.path),
            }
        )

        if index % 100 == 0:
            print(f"processed {index}/{len(records)} matched CSVs", file=sys.stderr)

    if expected_feature_names is None or stats is None:
        raise ValueError("No valid samples could establish a feature schema.")
    if not manifest_rows:
        raise ValueError("No valid samples were exported.")

    split_hist: dict[str, int] = {}
    split_label_hist: dict[str, dict[str, int]] = {}
    for row in manifest_rows:
        split = str(row["split"])
        split_hist[split] = split_hist.get(split, 0) + 1
        split_label_hist.setdefault(split, {str(i): 0 for i in range(4)})
        label = str(row["label"])
        split_label_hist[split][label] = split_label_hist[split].get(label, 0) + 1

    feature_mean, feature_std, feature_std_safe = stats.finalize()

    if not args.dry_run:
        write_manifest(manifest_path, manifest_rows)
        np.savez_compressed(
            norm_stats_path,
            feature_names=np.asarray(expected_feature_names),
            feature_mean=feature_mean,
            feature_std=feature_std,
            feature_std_safe=feature_std_safe,
            feature_count=np.asarray(len(expected_feature_names), dtype=np.int64),
            norm_split=np.asarray(args.norm_split),
        )

    report_limit = max(0, int(args.keep_unmatched_report_limit))
    summary = {
        "input_dir": str(args.input_dir.expanduser().resolve()),
        "labels_csv": str(args.labels_csv.expanduser().resolve()),
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "sequence_dir": str(sequence_dir),
        "norm_stats_path": str(norm_stats_path),
        "matched_csvs": len(records),
        "exported_samples": len(manifest_rows),
        "feature_count": len(expected_feature_names),
        "motion_bases": [name for name in expected_feature_names[::3]],
        "label_histogram": label_histogram([int(row["label"]) for row in manifest_rows]),
        "split_histogram": split_hist,
        "split_label_histogram": split_label_hist,
        "csv_without_label_count": len(csv_without_label),
        "label_without_csv_count": len(label_without_csv),
        "skipped_feature_mismatch_count": len(skipped_feature_mismatch),
        "failed_sample_count": len(failed_samples),
        "csv_without_label_examples": csv_without_label[:report_limit],
        "label_without_csv_examples": label_without_csv[:report_limit],
        "skipped_feature_mismatch_examples": skipped_feature_mismatch[:report_limit],
        "failed_sample_examples": failed_samples[:report_limit],
        "dry_run": bool(args.dry_run),
        "split_source": args.split_source,
        "folds_dir": str(args.folds_dir.expanduser().resolve()) if args.split_source == "carepd-folds" else "",
        "carepd_fold_set": args.carepd_fold_set if args.split_source == "carepd-folds" else "",
        "carepd_fold_index": args.carepd_fold_index if args.split_source == "carepd-folds" else "",
        "carepd_fold_files": carepd_fold_files,
    }

    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"numpy\.core\.numeric is deprecated.*",
)

DEFAULT_INPUT_ROOT = Path("/Volumes/MAEVE/dataset/CARE-PD/Canonicalized_SMPL_pickles")
DEFAULT_OUTPUT_CSV = Path("/Volumes/MAEVE/dataset/CARE-PD/carepd_mds_updrs_gait_severity.csv")
OUTPUT_FILENAME_HEADER = "filename"
OUTPUT_SCORE_HEADER = "MDS-UPDRS gait severity score"

_TAKE_HINT_KEYS = {
    "pose",
    "poses",
    "trans",
    "transl",
    "translation",
    "beta",
    "betas",
    "fps",
    "mocapframerate",
    "mocapframeratehz",
    "bodypose",
    "globalorient",
    "rootorient",
}


@dataclass(frozen=True)
class ScoreRow:
    filename: str
    score: int


@dataclass
class ExtractionStats:
    pickle_files: int = 0
    take_like_records: int = 0
    rows_written: int = 0
    skipped_missing: int = 0
    skipped_invalid: int = 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MDS-UPDRS gait severity scores from CARE-PD pickle files "
            "and write them to a semicolon-delimited CSV."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="CARE-PD pickle directory or a single .pkl file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path.",
    )
    return parser.parse_args(argv)


def _normalize_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _sanitize_component(text: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return cleaned or "take"


def _unwrap(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _unwrap(value.item())
        if value.size == 1:
            return _unwrap(value.reshape(()).item())
    return value


def _discover_pickle_files(input_root: Path) -> list[Path]:
    if input_root.is_file():
        if input_root.suffix.lower() not in {".pkl", ".pickle"}:
            raise ValueError(f"Expected a .pkl file, got: {input_root}")
        return [input_root.resolve()]

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    seen: dict[str, Path] = {}
    for pattern in ("**/*.pkl", "**/*.pickle"):
        for path in input_root.glob(pattern):
            if path.is_file():
                seen[str(path.resolve())] = path.resolve()
    return [seen[key] for key in sorted(seen)]


def _looks_like_take_mapping(mapping: Mapping[str, Any]) -> bool:
    keys = {_normalize_key(key) for key in mapping.keys()}
    return bool(keys & _TAKE_HINT_KEYS)


def _score_priority(normalized_key: str) -> int | None:
    if normalized_key == "updrsgait":
        return 0
    if "mds" in normalized_key and "updrs" in normalized_key and "gait" in normalized_key and "severity" in normalized_key:
        return 1
    if "updrs" in normalized_key and "gait" in normalized_key:
        return 2
    if "gait" in normalized_key and "severity" in normalized_key:
        return 3
    return None


def _find_score_item(mapping: Mapping[str, Any]) -> tuple[str, Any] | None:
    best: tuple[int, str, Any] | None = None
    for key, value in mapping.items():
        priority = _score_priority(_normalize_key(key))
        if priority is None:
            continue
        candidate = (priority, str(key), value)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return best[1], best[2]


def _coerce_score(value: Any) -> int | None:
    value = _unwrap(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    score = int(number)
    if score < 0 or score > 3:
        return None
    return score


def _build_filename(source_path: Path, path_parts: Sequence[str]) -> str:
    parts = [_sanitize_component(source_path.stem)]
    parts.extend(_sanitize_component(part) for part in path_parts if part)
    return "__".join(parts)


def _filename_sort_key(filename: str) -> tuple[Any, ...]:
    key: list[Any] = []
    for part in filename.split("__"):
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return tuple(key)


def _iter_child_items(obj: Any) -> list[tuple[str, Any]]:
    obj = _unwrap(obj)
    if isinstance(obj, Mapping):
        return [(str(key), value) for key, value in obj.items()]
    if isinstance(obj, (list, tuple)):
        return [(str(idx), value) for idx, value in enumerate(obj)]
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        if obj.ndim == 0:
            return _iter_child_items(obj.item())
        return [(str(idx), value) for idx, value in enumerate(obj.tolist())]
    return []


def _collect_rows(
    obj: Any,
    source_path: Path,
    path_parts: tuple[str, ...],
    stats: ExtractionStats,
) -> list[ScoreRow]:
    obj = _unwrap(obj)
    if isinstance(obj, Mapping):
        if _looks_like_take_mapping(obj):
            stats.take_like_records += 1
            score_item = _find_score_item(obj)
            if score_item is None or score_item[1] is None:
                stats.skipped_missing += 1
                return []
            score = _coerce_score(score_item[1])
            if score is None:
                stats.skipped_invalid += 1
                return []
            return [ScoreRow(filename=_build_filename(source_path, path_parts), score=score)]

        rows: list[ScoreRow] = []
        for key, value in obj.items():
            rows.extend(_collect_rows(value, source_path, path_parts + (str(key),), stats))
        return rows

    rows: list[ScoreRow] = []
    for key, value in _iter_child_items(obj):
        rows.extend(_collect_rows(value, source_path, path_parts + (key,), stats))
    return rows


def extract_score_rows(input_root: Path) -> tuple[list[ScoreRow], ExtractionStats]:
    input_root = input_root.expanduser().resolve()
    pickle_files = _discover_pickle_files(input_root)
    if not pickle_files:
        raise FileNotFoundError(f"No .pkl files found under {input_root}")

    stats = ExtractionStats(pickle_files=len(pickle_files))
    rows: list[ScoreRow] = []
    for pickle_path in pickle_files:
        with pickle_path.open("rb") as handle:
            loaded = pickle.load(handle)
        rows.extend(_collect_rows(loaded, pickle_path, (), stats))

    rows.sort(key=lambda row: _filename_sort_key(row.filename))
    stats.rows_written = len(rows)
    return rows, stats


def write_score_csv(rows: Sequence[ScoreRow], output_csv: Path) -> Path:
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[OUTPUT_FILENAME_HEADER, OUTPUT_SCORE_HEADER],
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    OUTPUT_FILENAME_HEADER: row.filename,
                    OUTPUT_SCORE_HEADER: row.score,
                }
            )
    return output_csv


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, stats = extract_score_rows(args.input_root)
    output_csv = write_score_csv(rows, args.output_csv)

    print(
        f"Wrote {stats.rows_written} row(s) to {output_csv} "
        f"from {stats.pickle_files} pickle file(s)."
    )
    print(
        f"Scanned {stats.take_like_records} take-like record(s): "
        f"{stats.skipped_missing} without a gait severity score, "
        f"{stats.skipped_invalid} with an invalid score."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

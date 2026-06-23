#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from add_babel_action_column import (
    BabelLabel,
    BabelMatch,
    action_at_time,
    extract_frame_labels,
    extract_sequence_labels,
    iter_babel_json_paths,
    normalize_feat_path,
    sequence_action_text,
    split_priority,
)


@dataclass(frozen=True)
class ProcessResult:
    input_csv: Path
    output_csv: Path
    status: str
    message: str
    match: BabelMatch | None = None
    rows: int = 0


class BabelIndex:
    def __init__(self, matches: Iterable[BabelMatch]) -> None:
        best_by_feat: dict[str, BabelMatch] = {}
        for match in matches:
            current = best_by_feat.get(match.feat_p)
            if current is None:
                best_by_feat[match.feat_p] = match
                continue
            if len(match.labels) > len(current.labels):
                best_by_feat[match.feat_p] = match
                continue
            if len(match.labels) == len(current.labels) and len(match.seq_labels) > len(current.seq_labels):
                best_by_feat[match.feat_p] = match
                continue
            if len(match.labels) == len(current.labels) and split_priority(match.split_path) < split_priority(
                current.split_path
            ):
                best_by_feat[match.feat_p] = match

        self._by_suffix: dict[str, list[BabelMatch]] = {}
        self._by_normalized_suffix: dict[str, list[BabelMatch]] = {}
        for match in best_by_feat.values():
            parts = match.feat_p.replace("\\", "/").split("/")
            for start in range(len(parts)):
                suffix = "/".join(parts[start:])
                self._by_suffix.setdefault(suffix, []).append(match)
                self._by_normalized_suffix.setdefault(normalize_feat_path(suffix), []).append(match)

    @classmethod
    def from_json_paths(cls, json_paths: list[Path], label_field: str) -> "BabelIndex":
        matches: list[BabelMatch] = []
        seen: set[tuple[str, float, tuple[BabelLabel, ...], tuple[str, ...]]] = set()
        for json_path in json_paths:
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            for key, record in data.items():
                feat_p = str(record.get("feat_p", "")).replace("\\", "/")
                if not feat_p:
                    continue
                labels = extract_frame_labels(record, label_field)
                seq_labels = extract_sequence_labels(record, label_field)
                match = BabelMatch(
                    split_path=json_path,
                    key=str(key),
                    feat_p=feat_p,
                    dur=float(record.get("dur", 0.0)),
                    labels=labels,
                    seq_labels=seq_labels,
                )
                signature = (match.feat_p, match.dur, tuple(match.labels), tuple(match.seq_labels))
                if signature in seen:
                    continue
                seen.add(signature)
                matches.append(match)
        return cls(matches)

    def find(self, candidates: list[str]) -> tuple[BabelMatch | None, list[BabelMatch]]:
        for candidate in sorted(set(candidates), key=lambda item: (item.count("/"), len(item)), reverse=True):
            matches = self._by_suffix.get(candidate.replace("\\", "/"))
            if not matches:
                matches = self._by_normalized_suffix.get(normalize_feat_path(candidate))
            if not matches:
                continue
            if len(matches) == 1:
                return matches[0], matches
            return None, matches
        return None, []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively add a BABEL action column to AMASS-derived CSV files. "
            "The output folder mirrors the input CSV folder structure."
        )
    )
    parser.add_argument("--babel-dir", required=True, help="Folder containing BABEL JSON split files.")
    parser.add_argument("--input-csv-root", required=True, help="Root folder containing AMASS CSV files.")
    parser.add_argument("--output-root", required=True, help="Output folder for annotated CSV files.")
    parser.add_argument("--glob", default="*.csv", help="CSV glob relative to input root (default: *.csv).")
    parser.add_argument(
        "--action-column",
        default="action",
        help="Name of the column to add (default: action).",
    )
    parser.add_argument(
        "--label-field",
        choices=["act_cat", "proc_label", "raw_label"],
        default="act_cat",
        help="BABEL label field to write into the action column (default: act_cat).",
    )
    parser.add_argument(
        "--time-column",
        default="time",
        help="CSV time column in seconds (default: time).",
    )
    parser.add_argument(
        "--frame-column",
        default="frame",
        help="CSV frame column used only when --fps is provided and no time column exists.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate used if the CSV has no time column.",
    )
    parser.add_argument(
        "--unlabeled",
        default="",
        help="Value used for frames outside any frame-level BABEL interval (default: empty).",
    )
    parser.add_argument(
        "--sequence-fallback",
        choices=["none", "all_frames"],
        default="none",
        help=(
            "When a matched BABEL record has no frame-level labels, write its sequence-level "
            "labels on every frame (all_frames) or leave action empty (none, default)."
        ),
    )
    parser.add_argument(
        "--on-missing",
        choices=["empty", "skip", "error"],
        default="empty",
        help="Behavior when no BABEL record matches a CSV (default: empty).",
    )
    parser.add_argument(
        "--on-ambiguous",
        choices=["empty", "skip", "error"],
        default="empty",
        help="Behavior when several BABEL records match a CSV (default: empty).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output CSVs that already exist and are non-empty.",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help="Path for the processing report CSV (default: <output-root>/babel_action_report.csv).",
    )
    return parser.parse_args()


def csv_to_feat_candidates(csv_path: Path, input_root: Path) -> list[str]:
    relative_path = csv_path.relative_to(input_root)
    rel_parent = relative_path.parent.as_posix()
    stem = relative_path.stem
    suffixes = [
        f"{stem}.npz",
        f"{stem.removesuffix('_stageii')}_poses.npz",
        f"{stem.removesuffix('_stageii')}.npz",
    ]

    candidates: list[str] = []
    prefixes = []
    if rel_parent != ".":
        parts = rel_parent.split("/")
        prefixes.extend("/".join(parts[start:]) for start in range(len(parts)))
    prefixes.append("")

    for prefix in prefixes:
        for suffix in suffixes:
            candidate = f"{prefix}/{suffix}" if prefix else suffix
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def collect_input_csvs(input_root: Path, output_root: Path, pattern: str) -> list[Path]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    paths = []
    for path in sorted(input_root.rglob(pattern)):
        if not path.is_file():
            continue
        try:
            path.resolve().relative_to(output_root)
            continue
        except ValueError:
            pass
        paths.append(path)
    return paths


def get_row_time(row: dict[str, str], args: argparse.Namespace) -> float:
    if args.time_column in row and row[args.time_column] != "":
        return float(row[args.time_column])
    if args.fps is not None and args.frame_column in row and row[args.frame_column] != "":
        return float(row[args.frame_column]) / args.fps
    raise ValueError(
        f"CSV row has no usable {args.time_column!r} column; provide --fps "
        f"and a {args.frame_column!r} column."
    )


def write_annotated_csv(
    input_csv: Path,
    output_csv: Path,
    labels: list[BabelLabel],
    fallback_action: str | None,
    args: argparse.Namespace,
) -> int:
    with input_csv.open("r", encoding="utf-8", newline="") as in_handle:
        reader = csv.DictReader(in_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_csv}")
        fieldnames = list(reader.fieldnames)
        if args.action_column not in fieldnames:
            fieldnames.append(args.action_column)

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as out_handle:
            writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
            writer.writeheader()
            rows = 0
            for row in reader:
                if fallback_action is not None:
                    row[args.action_column] = fallback_action
                else:
                    t = get_row_time(row, args)
                    row[args.action_column] = action_at_time(t, labels, args.unlabeled)
                writer.writerow(row)
                rows += 1
    return rows


def process_one(
    input_csv: Path,
    input_root: Path,
    output_root: Path,
    index: BabelIndex,
    args: argparse.Namespace,
) -> ProcessResult:
    relative_path = input_csv.relative_to(input_root)
    output_csv = output_root / relative_path
    if args.skip_existing and output_csv.exists() and output_csv.stat().st_size > 0:
        return ProcessResult(input_csv, output_csv, "skipped_existing", "output exists")

    match, ambiguous = index.find(csv_to_feat_candidates(input_csv, input_root))
    if match is not None:
        fallback_action = None
        status = "matched" if match.labels else "matched_no_frame_labels"
        if not match.labels and args.sequence_fallback == "all_frames" and match.seq_labels:
            fallback_action = sequence_action_text(match.seq_labels)
            status = "matched_sequence_fallback"
        rows = write_annotated_csv(input_csv, output_csv, match.labels, fallback_action, args)
        message = f"{match.split_path.name}:{match.key}:{match.feat_p}"
        return ProcessResult(input_csv, output_csv, status, message, match=match, rows=rows)

    if ambiguous:
        message = "; ".join(f"{m.split_path.name}:{m.key}:{m.feat_p}" for m in ambiguous)
        if args.on_ambiguous == "skip":
            return ProcessResult(input_csv, output_csv, "ambiguous", message)
        if args.on_ambiguous == "error":
            raise LookupError(f"Ambiguous BABEL match for {input_csv}: {message}")
        rows = write_annotated_csv(input_csv, output_csv, [], None, args)
        return ProcessResult(input_csv, output_csv, "ambiguous_empty", message, rows=rows)

    if match is None:
        message = "no BABEL record matched generated feat_p candidates"
        if args.on_missing == "skip":
            return ProcessResult(input_csv, output_csv, "missing", message)
        if args.on_missing == "error":
            raise LookupError(f"Missing BABEL match for {input_csv}")
        rows = write_annotated_csv(input_csv, output_csv, [], None, args)
        return ProcessResult(input_csv, output_csv, "missing_empty", message, rows=rows)


def write_report(report_path: Path, results: list[ProcessResult]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_csv",
                "output_csv",
                "status",
                "rows",
                "babel_split",
                "babel_key",
                "babel_feat_p",
                "babel_frame_labels",
                "babel_sequence_labels",
                "message",
            ],
        )
        writer.writeheader()
        for result in results:
            match = result.match
            writer.writerow(
                {
                    "input_csv": result.input_csv,
                    "output_csv": result.output_csv,
                    "status": result.status,
                    "rows": result.rows,
                    "babel_split": match.split_path.name if match else "",
                    "babel_key": match.key if match else "",
                    "babel_feat_p": match.feat_p if match else "",
                    "babel_frame_labels": len(match.labels) if match else "",
                    "babel_sequence_labels": len(match.seq_labels) if match else "",
                    "message": result.message,
                }
            )


def main() -> int:
    args = parse_args()
    babel_dir = Path(args.babel_dir).expanduser().resolve()
    input_root = Path(args.input_csv_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    report_path = Path(args.report_csv).expanduser().resolve() if args.report_csv else output_root / "babel_action_report.csv"

    try:
        json_paths = iter_babel_json_paths([str(babel_dir)])
        index = BabelIndex.from_json_paths(json_paths, args.label_field)
        csv_paths = collect_input_csvs(input_root, output_root, args.glob)
        results = []
        for csv_path in csv_paths:
            result = process_one(csv_path, input_root, output_root, index, args)
            results.append(result)
            print(f"{result.status}: {csv_path.relative_to(input_root)}")
        write_report(report_path, results)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print(f"processed {len(results)} CSV file(s)")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

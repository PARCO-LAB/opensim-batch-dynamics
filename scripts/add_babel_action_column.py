#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class BabelLabel:
    start_t: float
    end_t: float
    text: str


@dataclass(frozen=True)
class BabelMatch:
    split_path: Path
    key: str
    feat_p: str
    dur: float
    labels: list[BabelLabel]
    seq_labels: list[str]


DEFAULT_DATASET_ALIASES = {
    "DFaust": "DFaust67",
    "DFAUST": "DFaust67",
    "Eyes_Japan_Dataset": "EyesJapanDataset",
    "HDM05": "MPIHDM05",
    "MoSh": "MPImosh",
    "SSM": "SSMsynced",
    "TCDHands": "TCDhandMocap",
    "Transitions": "Transitionsmocap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add a per-frame BABEL action column to an AMASS-derived CSV. "
            "BABEL intervals are in seconds and are matched against the CSV time column."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Input AMASS-derived CSV.")
    parser.add_argument("--output-csv", required=True, help="Output CSV with the new action column.")
    parser.add_argument(
        "--babel",
        nargs="+",
        required=True,
        help="One or more BABEL JSON files, or directories containing .json split files.",
    )
    parser.add_argument(
        "--feat-p",
        default=None,
        help=(
            "AMASS path as stored in BABEL feat_p. If omitted, the script tries to infer "
            "a filename/suffix match from --input-csv."
        ),
    )
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
    return parser.parse_args()


def iter_babel_json_paths(paths: Iterable[str]) -> list[Path]:
    found: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            found.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"BABEL path does not exist: {path}")
    if not found:
        raise FileNotFoundError("No BABEL JSON files found.")
    return found


def candidate_feat_paths(input_csv: Path, explicit_feat_p: str | None) -> list[str]:
    if explicit_feat_p:
        return [explicit_feat_p.replace("\\", "/")]

    name = input_csv.name
    stem = input_csv.stem
    parent = input_csv.parent.as_posix()
    suffixes = [
        name.removesuffix(".csv") + ".npz",
        stem + ".npz",
        stem.removesuffix("_stageii") + "_poses.npz",
        stem.removesuffix("_stageii") + ".npz",
    ]

    candidates: list[str] = []
    for item in suffixes:
        if item not in candidates:
            candidates.append(item)
    parts = parent.split("/")
    for start in range(max(0, len(parts) - 6), len(parts)):
        prefix = "/".join(parts[start:])
        if not prefix:
            continue
        for item in suffixes:
            candidate = f"{prefix}/{item}"
            if candidate not in candidates:
                candidates.append(candidate)
    return expand_dataset_alias_candidates(candidates)


def expand_dataset_alias_candidates(candidates: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for candidate in candidates:
        normalized = candidate.replace("\\", "/")
        for item in dataset_alias_variants(normalized):
            if item not in expanded:
                expanded.append(item)
    return expanded


def dataset_alias_variants(path: str) -> list[str]:
    parts = path.replace("\\", "/").split("/")
    variants = ["/".join(parts)]
    for index, part in enumerate(parts):
        alias = DEFAULT_DATASET_ALIASES.get(part)
        if not alias:
            continue
        aliased = list(parts)
        aliased[index] = alias
        variants.append("/".join(aliased))
    return variants


def feat_matches(feat_p: str, candidates: list[str]) -> bool:
    normalized = feat_p.replace("\\", "/")
    if any(normalized == c or normalized.endswith("/" + c) for c in candidates):
        return True
    normalized_feat = normalize_feat_path(normalized)
    return any(
        normalized_feat == normalize_feat_path(candidate)
        or normalized_feat.endswith("/" + normalize_feat_path(candidate))
        for candidate in candidates
    )


def normalize_feat_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    normalized_parts = []
    for part in parts:
        normalized = re.sub(r"[^a-z0-9]+", "", part.lower())
        if normalized:
            normalized_parts.append(normalized)
    return "/".join(normalized_parts)


def label_text(label: dict[str, Any], field: str) -> str:
    value = label.get(field)
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def iter_frame_annotations(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    singular = record.get("frame_ann")
    if isinstance(singular, dict):
        yield singular
    plural = record.get("frame_anns")
    if isinstance(plural, list):
        for item in plural:
            if isinstance(item, dict):
                yield item


def iter_sequence_annotations(record: dict[str, Any]) -> Iterable[dict[str, Any]]:
    singular = record.get("seq_ann")
    if isinstance(singular, dict):
        yield singular
    plural = record.get("seq_anns")
    if isinstance(plural, list):
        for item in plural:
            if isinstance(item, dict):
                yield item


def extract_frame_labels(record: dict[str, Any], label_field: str) -> list[BabelLabel]:
    labels: list[BabelLabel] = []
    seen: set[BabelLabel] = set()
    for annotation in iter_frame_annotations(record):
        for label in annotation.get("labels") or []:
            if "start_t" not in label or "end_t" not in label:
                continue
            text = label_text(label, label_field)
            if not text:
                continue
            item = BabelLabel(
                start_t=float(label["start_t"]),
                end_t=float(label["end_t"]),
                text=text,
            )
            if item in seen:
                continue
            seen.add(item)
            labels.append(item)
    return sorted(labels, key=lambda item: (item.start_t, item.end_t, item.text))


def extract_sequence_labels(record: dict[str, Any], label_field: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for annotation in iter_sequence_annotations(record):
        for label in annotation.get("labels") or []:
            text = label_text(label, label_field)
            if not text or text in seen:
                continue
            seen.add(text)
            labels.append(text)
    return labels


def sequence_action_text(seq_labels: list[str]) -> str:
    return ";".join(dict.fromkeys(seq_labels))


def split_priority(path: Path) -> int:
    name = path.stem
    if name in {"train", "val", "test"}:
        return 0
    return 1


def load_matching_babel_record(
    json_paths: list[Path],
    candidates: list[str],
    label_field: str,
) -> BabelMatch:
    matches: list[BabelMatch] = []
    for json_path in json_paths:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for key, record in data.items():
            feat_p = str(record.get("feat_p", ""))
            if not feat_matches(feat_p, candidates):
                continue
            labels = extract_frame_labels(record, label_field)
            seq_labels = extract_sequence_labels(record, label_field)
            matches.append(
                BabelMatch(
                    split_path=json_path,
                    key=str(key),
                    feat_p=feat_p,
                    dur=float(record.get("dur", 0.0)),
                    labels=sorted(labels, key=lambda item: (item.start_t, item.end_t, item.text)),
                    seq_labels=seq_labels,
                )
            )

    unique_matches: list[BabelMatch] = []
    seen: set[tuple[str, float, tuple[BabelLabel, ...], tuple[str, ...]]] = set()
    for match in matches:
        signature = (match.feat_p, match.dur, tuple(match.labels), tuple(match.seq_labels))
        if signature in seen:
            continue
        seen.add(signature)
        unique_matches.append(match)
    matches = unique_matches

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
    matches = list(best_by_feat.values())

    if not matches:
        tried = "\n  ".join(candidates[:20])
        raise LookupError(f"No BABEL record matched the input. Tried:\n  {tried}")
    if len(matches) > 1:
        lines = "\n".join(f"  {m.split_path.name}:{m.key} {m.feat_p}" for m in matches)
        raise LookupError(f"Multiple BABEL records matched; pass --feat-p explicitly:\n{lines}")
    return matches[0]


def row_time(row: dict[str, str], args: argparse.Namespace) -> float:
    if args.time_column in row and row[args.time_column] != "":
        return float(row[args.time_column])
    if args.fps is not None and args.frame_column in row and row[args.frame_column] != "":
        return float(row[args.frame_column]) / args.fps
    raise ValueError(
        f"CSV row has no usable {args.time_column!r} column; provide --fps "
        f"and a {args.frame_column!r} column."
    )


def action_at_time(t: float, labels: list[BabelLabel], unlabeled: str) -> str:
    active: list[str] = []
    for label in labels:
        if label.start_t <= t < label.end_t:
            active.append(label.text)
    return ";".join(dict.fromkeys(active)) if active else unlabeled


def annotate_csv(args: argparse.Namespace) -> BabelMatch:
    input_csv = Path(args.input_csv).expanduser()
    output_csv = Path(args.output_csv).expanduser()
    json_paths = iter_babel_json_paths(args.babel)
    candidates = candidate_feat_paths(input_csv, args.feat_p)
    match = load_matching_babel_record(json_paths, candidates, args.label_field)
    fallback_action = ""
    if args.sequence_fallback == "all_frames" and not match.labels:
        fallback_action = sequence_action_text(match.seq_labels)

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
            last_t = 0.0
            for row in reader:
                if fallback_action:
                    row[args.action_column] = fallback_action
                    try:
                        last_t = row_time(row, args)
                    except ValueError:
                        pass
                else:
                    t = row_time(row, args)
                    row[args.action_column] = action_at_time(t, match.labels, args.unlabeled)
                    last_t = t
                writer.writerow(row)
                rows += 1

    if rows and match.dur and abs(last_t - match.dur) > 0.25:
        print(
            f"warning: CSV last time is {last_t:.3f}s, BABEL duration is {match.dur:.3f}s",
            file=sys.stderr,
        )
    return match


def main() -> int:
    args = parse_args()
    try:
        match = annotate_csv(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"matched {match.split_path.name}:{match.key} {match.feat_p} "
        f"({len(match.labels)} frame labels, {len(match.seq_labels)} sequence labels)"
    )
    print(f"wrote {Path(args.output_csv).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

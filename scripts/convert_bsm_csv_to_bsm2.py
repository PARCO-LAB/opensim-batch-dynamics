#!/usr/bin/env python3
"""Convert bsm.osim stage-II CSV files to the bsm2.osim coordinate convention."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Convert stage-II CSV files produced with bsm.osim so that they preserve "
            "the same physical motion and generalized power with bsm2.osim."
        )
    )
    parser.add_argument("input", help="Input CSV file or directory")
    parser.add_argument(
        "--output",
        help=(
            "Output CSV for a file input, or output root for a directory input. "
            "By default, writes beside each source using the suffix from --suffix."
        ),
    )
    parser.add_argument(
        "--target-model",
        default=str(repo_root / "model" / "bsm" / "bsm2.osim"),
        help="Target bsm2 model used to validate axes and coordinate ranges",
    )
    parser.add_argument(
        "--pattern",
        default="*_stageii.csv",
        help="Glob used for directory inputs (default: *_stageii.csv)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directory inputs recursively",
    )
    parser.add_argument(
        "--suffix",
        default="_bsm2",
        help="Suffix added before .csv (default: _bsm2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files",
    )
    parser.add_argument(
        "--allow-out-of-range",
        action="store_true",
        help="Allow converted values outside clamped bsm2 coordinate ranges",
    )
    return parser


def _discover(input_path: Path, pattern: str, recursive: bool, suffix: str) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    iterator = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
    files = sorted(path for path in iterator if not path.stem.endswith(suffix))
    if not files:
        mode = "recursively" if recursive else ""
        raise FileNotFoundError(f"No files matching {pattern!r} found {mode} in {input_path}")
    return files


def main() -> int:
    _add_src_to_path()
    from opensim_batch_dynamics.bsm_csv_conversion import convert_bsm_csv_to_bsm2

    args = build_arg_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_arg = Path(args.output).expanduser().resolve() if args.output else None
    if input_path.is_file() and input_path.stem.endswith(args.suffix):
        raise ValueError(
            f"Input name already ends with {args.suffix!r}; refusing a likely double conversion"
        )
    files = _discover(input_path, args.pattern, args.recursive, args.suffix)

    if input_path.is_file():
        destination = output_arg or input_path.with_name(
            input_path.stem + args.suffix + input_path.suffix
        )
        destinations = [destination]
    else:
        destinations = []
        for source in files:
            relative = source.relative_to(input_path)
            filename = relative.stem + args.suffix + relative.suffix
            if output_arg:
                destinations.append(output_arg / relative.parent / filename)
            else:
                destinations.append(source.with_name(filename))

    if len(set(destinations)) != len(destinations):
        raise ValueError("Multiple inputs resolve to the same output path")

    for source, destination in zip(files, destinations):
        summary = convert_bsm_csv_to_bsm2(
            source,
            destination,
            target_model_path=args.target_model,
            overwrite=args.overwrite,
            allow_out_of_range=args.allow_out_of_range,
        )
        ranges = ", ".join(
            f"{name}=[{bounds[0]:.6g}, {bounds[1]:.6g}]"
            for name, bounds in summary.converted_position_ranges.items()
        )
        print(
            f"converted {summary.rows} rows: {summary.input_path} -> "
            f"{summary.output_path} ({ranges})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

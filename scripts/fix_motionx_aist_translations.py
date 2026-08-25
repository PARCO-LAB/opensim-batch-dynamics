#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the official Motion-X AIST translation correction."
    )
    parser.add_argument("input_dir", type=Path, help="Motion-X smplx_322/aist directory")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        help="Corrected output directory (default: <input_dir>_fixed)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_dir.with_name(f"{input_dir.name}_fixed")
    )
    if not input_dir.is_dir():
        parser.error(f"input directory not found: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        parser.error("output directory must be outside input_dir")

    files = sorted(input_dir.rglob("*.npy"))
    if not files:
        parser.error(f"no .npy files found under: {input_dir}")

    for index, source in enumerate(files, start=1):
        motion = np.load(source, allow_pickle=False)
        if motion.ndim != 2 or motion.shape[1] != 322:
            raise ValueError(f"expected (T, 322), got {motion.shape}: {source}")
        if not np.issubdtype(motion.dtype, np.floating):
            raise TypeError(f"expected floating-point data, got {motion.dtype}: {source}")

        motion[:, 309:312] /= 94.0
        motion[:, 311] *= -1.0

        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npy.tmp")
        with temporary.open("wb") as file:
            np.save(file, motion, allow_pickle=False)
        os.replace(temporary, destination)

        if index % 100 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {source.relative_to(input_dir)}")

    print(f"Corrected files written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

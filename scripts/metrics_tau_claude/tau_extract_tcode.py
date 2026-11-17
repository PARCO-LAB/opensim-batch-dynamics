#!/usr/bin/env python3
"""Forward to the repository's read/write-safe T_code target extractor."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=os.environ.get("OPEN_SIM_BATCH_DYNAMICS_ROOT"),
        help="Clone containing scripts/extract_groundlink_torque_targets.py",
    )
    args, passthrough = parser.parse_known_args()
    if args.repo_root is None:
        parser.error("--repo-root or OPEN_SIM_BATCH_DYNAMICS_ROOT is required")
    script = args.repo_root.expanduser().resolve() / "scripts" / "extract_groundlink_torque_targets.py"
    if not script.is_file():
        parser.error(f"Target extractor is missing: {script}")
    sys.argv = [str(script), *passthrough]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the unified GroundLink Table I evaluator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groundlink import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

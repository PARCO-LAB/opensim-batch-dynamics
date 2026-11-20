#!/usr/bin/env python3
"""Compatibility wrapper for the legacy OpenCap/OpenSim pipeline."""

from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    legacy_script = Path(__file__).with_name("run_amass_to_opencap_legacy.py")
    runpy.run_path(str(legacy_script), run_name="__main__")


if __name__ == "__main__":
    main()

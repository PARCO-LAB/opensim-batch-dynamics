#!/usr/bin/env python3
"""Run one OpenSim inverse-dynamics setup with an alternate external-load XML.

The setup, model, coordinates, and load files are treated as inputs. The
rewritten setup and generated ID output go only into the requested new output
directory.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def _resolved(value: str, setup_dir: Path) -> Path:
    path = Path(value.strip()).expanduser()
    return path if path.is_absolute() else (setup_dir / path).resolve()


def run_id(setup_path: Path, loads_path: Path, output_dir: Path, output_name: str, forces_to_exclude: str | None) -> Path:
    setup_path = setup_path.expanduser().resolve()
    loads_path = loads_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not setup_path.is_file() or not loads_path.is_file():
        raise FileNotFoundError(f"Missing setup or external-load XML: {setup_path}, {loads_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(setup_path)
    tool = tree.getroot().find("InverseDynamicsTool")
    if tool is None:
        raise ValueError(f"No InverseDynamicsTool in {setup_path}")
    for tag in ("model_file", "coordinates_file"):
        node = tool.find(tag)
        if node is None or not node.text:
            raise ValueError(f"Missing {tag} in {setup_path}")
        resolved = _resolved(node.text, setup_path.parent)
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing {tag} input: {resolved}")
        node.text = str(resolved)
    loads_node = tool.find("external_loads_file")
    if loads_node is None:
        raise ValueError(f"Missing external_loads_file in {setup_path}")
    loads_node.text = str(loads_path)
    results_node = tool.find("results_directory")
    if results_node is None:
        results_node = ET.SubElement(tool, "results_directory")
    results_node.text = str(output_dir)
    output_node = tool.find("output_gen_force_file")
    if output_node is None:
        raise ValueError(f"Missing output_gen_force_file in {setup_path}")
    output_node.text = output_name
    if forces_to_exclude is not None:
        excluded = tool.find("forces_to_exclude")
        if excluded is None:
            excluded = ET.SubElement(tool, "forces_to_exclude")
        excluded.text = forces_to_exclude

    output_setup = output_dir / "setup.xml"
    tree.write(output_setup, encoding="utf-8", xml_declaration=True)
    command = shutil.which("opensim-cmd")
    if command:
        subprocess.run([command, "-o", "error", "run-tool", str(output_setup)], cwd=output_dir, check=True)
    else:
        try:
            import opensim
        except ImportError as exc:
            raise RuntimeError("Install OpenSim 4.5.2 Python bindings or provide opensim-cmd") from exc
        tool = opensim.InverseDynamicsTool(str(output_setup))
        if tool.run() is False:
            raise RuntimeError(f"OpenSim ID failed for {output_setup}")
    output_sto = output_dir / output_name
    if not output_sto.is_file():
        raise FileNotFoundError(f"OpenSim did not create {output_sto}")
    return output_sto


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--external-loads", type=Path, required=True, help="Alternative ExternalLoads XML")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="id.sto")
    parser.add_argument("--forces-to-exclude", default="Actuators")
    args = parser.parse_args()
    result = run_id(args.setup, args.external_loads, args.output_dir, args.output_name, args.forces_to_exclude)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

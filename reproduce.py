#!/usr/bin/env python3
"""Render the submitted Figures 2-6 from frozen inputs."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = {str(n): ROOT / "scripts" / ("figure" + str(n)) / ("plot_figure" + str(n) + ".py") for n in range(2, 7)}
MODULES = {"2": ["numpy", "scipy", "matplotlib", "cartopy"], "3": ["numpy", "scipy", "matplotlib"], "4": ["numpy", "scipy", "matplotlib", "shapefile"], "5": ["numpy", "scipy", "matplotlib", "rasterio", "pyproj", "shapefile"], "6": ["numpy", "matplotlib", "cartopy"]}

def sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def check_inputs(figures, data_dir, verify):
    entries = json.loads((ROOT / "docs" / "data_manifest.json").read_text(encoding="utf-8"))["files"]
    problems = []
    for item in entries:
        if not set(figures).intersection(map(str, item.get("plot_figures", []))):
            continue
        filename = data_dir / item["path"]
        if not filename.is_file():
            problems.append("Missing input: " + str(filename))
        elif verify and sha256(filename) != item["sha256"]:
            problems.append("Input checksum differs from frozen data: " + str(filename))
    needed = set().union(*(set(MODULES[f]) for f in figures))
    for name in sorted(needed):
        if importlib.util.find_spec(name) is None:
            problems.append("Missing Python package: " + name)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print("Install requirements.txt and import the reproduction-data bundle; see README.md.", file=sys.stderr)
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["all", "2", "3", "4", "5", "6"], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("LRVF_DATA_DIR", ROOT / "data")))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("LRVF_OUTPUT_DIR", ROOT / "outputs")))
    parser.add_argument("--check", action="store_true", help="Check dependencies and required inputs, then exit")
    parser.add_argument("--verify-data", action="store_true", help="Also verify frozen input SHA-256 values")
    args = parser.parse_args()
    figures = list(SCRIPTS) if args.figure == "all" else [args.figure]
    data_dir, output_dir = args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if not check_inputs(figures, data_dir, args.verify_data):
        return 2
    if args.check:
        print("Inputs and dependencies verified for figures " + ", ".join(figures))
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(LRVF_DATA_DIR=str(data_dir), LRVF_OUTPUT_DIR=str(output_dir), MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    env.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    for figure in figures:
        print("Rendering Figure " + figure, flush=True)
        result = subprocess.run([sys.executable, str(SCRIPTS[figure])], cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    print("Figures saved in " + str(output_dir))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

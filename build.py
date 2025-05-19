#!/usr/bin/env python3

import argparse
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

from visualize.const import (
    CACHE_DIR as CONST_CACHE_DIR,
    VIDEO_DIR as CONST_VIDEO_DIR,
    BLENDER_PATH,
)
from visualize.process_pkl import process_pkl_file

# Constants (matching Makefile)
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
CACHE_DIR = Path(CONST_CACHE_DIR)
RESULT_DIR = Path(CONST_VIDEO_DIR)
RENDER_SMPL_SCRIPT = "blender/render_smpl.py"
RENDER_PRIM_SCRIPT = "blender/render_prim.py"

def run_command(cmd: List[str], cwd: Optional[str] = None) -> None:
    """Run a shell command and print its output."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

def render_sequence(script: str, camera_type: int, output_name: str) -> None:
    """Render a sequence using Blender."""
    cmd = [
        "blender",
        BLENDER_PATH,  # Use constant from visualize.const
        "--background",
        "--python", script,
        "--",
        "-t", str(camera_type),
        str(OUTPUT_DIR / output_name)
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    subprocess.run(cmd, check=True, env=env)

def process_and_render_file(pkl_file: Path) -> None:
    """Process a single .pkl file and render its sequences."""
    output_dir = OUTPUT_DIR / pkl_file.stem
    rendered_marker = output_dir.with_suffix(".rendered")

    # Skip if already rendered
    if rendered_marker.exists():
        print(f"Skipping {pkl_file.name} - already rendered")
        return

    # Generate .obj files
    print(f"Generating .obj files from {pkl_file}")
    process_pkl_file(str(pkl_file))

    # Render sequences
    print(f"Rendering {pkl_file.stem}")
    render_sequence(RENDER_PRIM_SCRIPT, 1, pkl_file.stem)
    render_sequence(RENDER_PRIM_SCRIPT, 2, pkl_file.stem)
    render_sequence(RENDER_SMPL_SCRIPT, 1, pkl_file.stem)
    render_sequence(RENDER_SMPL_SCRIPT, 2, pkl_file.stem)
    render_sequence(RENDER_PRIM_SCRIPT, 0, pkl_file.stem)

    # Mark as rendered
    rendered_marker.touch()

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and render SMPL meshes")
    parser.add_argument("--file", help="Process specific .pkl file (without .pkl extension)")

    args = parser.parse_args()

    # Create necessary directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    # Process files
    if args.file:
        # Process single file
        pkl_file = DATA_DIR / f"{args.file}.pkl"
        if not pkl_file.exists():
            print(f"Error: {pkl_file} does not exist")
            return
        process_and_render_file(pkl_file)
    else:
        # Process all .pkl files
        for pkl_file in DATA_DIR.glob("*.pkl"):
            process_and_render_file(pkl_file)

if __name__ == "__main__":
    main() 
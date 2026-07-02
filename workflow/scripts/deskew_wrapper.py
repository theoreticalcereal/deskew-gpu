import argparse
import shutil
import subprocess
from pathlib import Path
import sys


DEFAULT_MATLAB_BIN = "/home1/apps/MATLAB/R2024a/bin/matlab"


def resolve_matlab_bin(matlab_bin):
    candidates = [matlab_bin, "matlab", DEFAULT_MATLAB_BIN]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        candidate_path = Path(candidate)
        if candidate_path.is_file() and candidate_path.stat().st_mode & 0o111:
            return str(candidate_path)
    raise FileNotFoundError(
        "MATLAB executable not found. Checked requested matlab_bin="
        f"{matlab_bin!r}, PATH, and {DEFAULT_MATLAB_BIN}."
    )


def run_deskew(image_path, cell_name, cell_index, dx, dz, angle, flip, output_dir,
               matlab_bin=DEFAULT_MATLAB_BIN):

    script_dir = str(Path(__file__).parent.absolute())
    cell_name = "" if cell_name is None else str(cell_name).strip()

    # Only set CellIndex if a non-empty value was provided.
    cell_index_line = ""
    if cell_index and str(cell_index).strip():
        cell_index_line = f"CellIndex=int32({cell_index}); "

    matlab_cmd = (
        f"addpath('{script_dir}'); "
        f"imagePath='{image_path}'; "
        f"CellName='{cell_name}'; "
        + cell_index_line
        + f"dx={dx}; "
        f"dz={dz}; "
        f"angle={angle}; "
        f"flip={flip}; "
        f"output_dir='{output_dir}'; "
        f"run('deskew.m');"
    )

    try:
        matlab_bin = resolve_matlab_bin(matlab_bin)
        print(f"Running deskew with image: {image_path}, cell name: {cell_name}, "
              f"cell index: {cell_index!r}, dx: {dx}, dz: {dz}, "
              f"angle: {angle}, flip: {flip}, output_dir: {output_dir}, "
              f"matlab_bin: {matlab_bin}")
        command = [matlab_bin, "-batch", matlab_cmd]
        subprocess.run(command, check=True)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(127)
    except subprocess.CalledProcessError as e:
        print(f"MATLAB execution failed with error code: {e.returncode}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    parser.add_argument('--cell_name', default='')
    parser.add_argument('--cell_index', default='')
    parser.add_argument('--dx')
    parser.add_argument('--dz')
    parser.add_argument('--angle')
    parser.add_argument('--flip')
    parser.add_argument('--output_dir')
    parser.add_argument('--matlab_bin', default=DEFAULT_MATLAB_BIN)
    args = parser.parse_args()

    run_deskew(args.image_path, args.cell_name, args.cell_index,
               args.dx, args.dz, args.angle, args.flip, args.output_dir,
               args.matlab_bin)

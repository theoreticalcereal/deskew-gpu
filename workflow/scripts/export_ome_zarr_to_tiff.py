#!/usr/bin/env python3
"""Export deskewed OME-Zarr outputs to one merged TIFF stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from ome_zarr_io import TIFF_SUFFIXES, is_ome_zarr_path, log_progress, open_ome_zarr_array


SUPPORTED_OUTPUT_FORMATS = {"tiff"}


def _ensure_3d(volume, source: Path):
    array = np.asarray(volume)
    if array.ndim == 2:
        return array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D deskew output for {source}, got shape {array.shape}")
    return array


def discover_deskew_outputs(input_dir: Path | str) -> list[Path]:
    root = Path(input_dir)
    if root.is_dir() and is_ome_zarr_path(root):
        return [root]
    outputs = [
        path
        for path in root.iterdir()
        if path.is_dir() and is_ome_zarr_path(path)
    ]
    outputs.extend(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )
    return sorted(outputs, key=lambda path: path.name)


def _read_volume(path: Path):
    if is_ome_zarr_path(path):
        log_progress(f"Reading deskewed OME-Zarr output for TIFF export: {path.name}")
        return _ensure_3d(open_ome_zarr_array(path, mode="r"), path)
    log_progress(f"Reading deskewed TIFF output for merged TIFF export: {path.name}")
    return _ensure_3d(tifffile.imread(str(path)), path)


def _merge_volumes(paths: list[Path]):
    arrays = [_read_volume(path) for path in paths]
    yx_shapes = {tuple(array.shape[-2:]) for array in arrays}
    if len(yx_shapes) != 1:
        raise ValueError(f"Cannot merge deskew outputs with different YX shapes: {sorted(yx_shapes)}")
    return np.concatenate(arrays, axis=0)


def export_directory(input_dir: Path | str, output_dir: Path | str, output_format: str = "tiff") -> Path:
    normalized_format = str(output_format).lower()
    if normalized_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {output_format}. Supported export formats: tiff")

    outputs = discover_deskew_outputs(input_dir)
    if not outputs:
        raise FileNotFoundError(f"No deskew outputs found in {input_dir}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "deskewed_merged.tif"
    merged = _merge_volumes(outputs)
    log_progress(f"Writing merged deskew TIFF stack: {destination}")
    tifffile.imwrite(str(destination), merged, bigtiff=True)
    return destination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export deskewed OME-Zarr outputs to one merged TIFF stack.")
    parser.add_argument("--input", required=True, help="Directory containing deskewed Top_shear outputs.")
    parser.add_argument("--output", required=True, help="Directory where the merged TIFF stack should be written.")
    parser.add_argument("--output-format", default="tiff", help="Requested output format. Currently supports tiff.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    export_directory(args.input, args.output, output_format=args.output_format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

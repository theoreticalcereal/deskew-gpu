#!/usr/bin/env python3
"""Export deskewed OME-Zarr outputs to selected transport formats."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import numpy as np
import tifffile

from ome_zarr_io import TIFF_SUFFIXES, image_stem, is_ome_zarr_path, log_progress, open_ome_zarr_array


SUPPORTED_OUTPUT_FORMATS = {"tiff", "ozx"}


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


def _write_ozx_archive(path: Path, output_root: Path) -> Path:
    destination = output_root / f"{image_stem(path)}.ozx"
    log_progress(f"Writing zipped OME-Zarr archive: {destination}")
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for item in sorted(path.rglob("*")):
            if item.is_file():
                archive.write(item, Path(path.name) / item.relative_to(path))
    return destination


def _export_ozx_archives(paths: list[Path], output_root: Path) -> list[Path]:
    zarr_paths = [path for path in paths if path.is_dir() and is_ome_zarr_path(path)]
    if len(zarr_paths) != len(paths):
        unsupported = [path.name for path in paths if path not in zarr_paths]
        raise ValueError(f"OZX export only supports OME-Zarr directories, got: {unsupported}")
    return [_write_ozx_archive(path, output_root) for path in zarr_paths]


def export_directory(input_dir: Path | str, output_dir: Path | str, output_format: str = "tiff") -> Path | list[Path]:
    normalized_format = str(output_format).lower()
    if normalized_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format: {output_format}. "
            f"Supported export formats: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}"
        )

    outputs = discover_deskew_outputs(input_dir)
    if not outputs:
        raise FileNotFoundError(f"No deskew outputs found in {input_dir}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if normalized_format == "ozx":
        return _export_ozx_archives(outputs, output_root)

    destination = output_root / "deskewed_merged.tif"
    merged = _merge_volumes(outputs)
    log_progress(f"Writing merged deskew TIFF stack: {destination}")
    tifffile.imwrite(str(destination), merged, bigtiff=True)
    return destination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export deskewed OME-Zarr outputs to TIFF or OZX.")
    parser.add_argument("--input", required=True, help="Directory containing deskewed Top_shear outputs.")
    parser.add_argument("--output", required=True, help="Directory where exported output should be written.")
    parser.add_argument("--output-format", default="tiff", help="Requested output format. Supports tiff and ozx.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    export_directory(args.input, args.output, output_format=args.output_format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Small OME-Zarr v0.4 helpers for native workflow image volumes."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from datetime import datetime
from typing import Iterable


TIFF_SUFFIXES = {".tif", ".tiff"}
OME_ZARR_SUFFIX = ".ome.zarr"


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def is_ome_zarr_path(path: Path | str) -> bool:
    return Path(path).name.lower().endswith(OME_ZARR_SUFFIX)


def image_stem(path: Path | str) -> str:
    path = Path(path)
    if is_ome_zarr_path(path):
        return path.name[: -len(OME_ZARR_SUFFIX)]
    if path.suffix.lower() in TIFF_SUFFIXES:
        return path.name[: -len(path.suffix)]
    return path.stem


def discover_image_volumes(input_dir: Path | str) -> list[Path]:
    root = Path(input_dir)
    if root.is_dir() and is_ome_zarr_path(root):
        return [root]
    paths = [
        path
        for path in root.iterdir()
        if path.is_dir() and is_ome_zarr_path(path)
    ]
    paths.extend(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )
    return sorted(paths, key=lambda path: path.name)


def multiscales_metadata(layer_name: str) -> dict:
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": layer_name,
                "axes": [
                    {"name": "z", "type": "space"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1, 1, 1]}
                        ],
                    }
                ],
            }
        ]
    }


def open_ome_zarr_array(path: Path | str, mode: str = "r"):
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'zarr' for OME-Zarr volume access") from exc

    return zarr.open(str(Path(path) / "0"), mode=mode)


def create_ome_zarr_array(
    path: Path | str,
    *,
    shape: Iterable[int],
    dtype,
    chunks: Iterable[int],
    layer_name: str | None = None,
    overwrite: bool = True,
):
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'zarr' for OME-Zarr volume access") from exc

    zarr_path = Path(path)
    if overwrite and zarr_path.exists():
        log_progress(f"Removing existing OME-Zarr output: {zarr_path}")
        shutil.rmtree(zarr_path)
    zarr_path.mkdir(parents=True, exist_ok=True)
    (zarr_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}) + "\n")
    (zarr_path / ".zattrs").write_text(
        json.dumps(
            multiscales_metadata(layer_name or image_stem(zarr_path)),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    log_progress(
        "Creating OME-Zarr array: "
        f"path={zarr_path}, shape={tuple(int(axis) for axis in shape)}, "
        f"chunks={tuple(int(axis) for axis in chunks)}, dtype={dtype}"
    )
    return zarr.open(
        str(zarr_path / "0"),
        mode="w",
        shape=tuple(int(axis) for axis in shape),
        chunks=tuple(int(axis) for axis in chunks),
        dtype=dtype,
        compressor=None,
    )


def write_ome_zarr_array(path: Path | str, array, *, chunks=None, layer_name: str | None = None) -> Path:
    shape = tuple(int(axis) for axis in array.shape)
    if len(shape) != 3:
        raise ValueError(f"OME-Zarr workflow volumes must be 3-D, got shape {shape}")
    if chunks is None:
        chunks = (min(16, shape[0]), min(256, shape[1]), min(256, shape[2]))
    output = Path(path)
    log_progress(
        "Writing OME-Zarr volume: "
        f"path={output}, shape={shape}, chunks={tuple(int(axis) for axis in chunks)}, "
        f"dtype={array.dtype}"
    )
    zarr_array = create_ome_zarr_array(
        output,
        shape=shape,
        dtype=array.dtype,
        chunks=chunks,
        layer_name=layer_name,
    )
    zarr_array[:] = array
    log_progress(f"Finished OME-Zarr volume: {output.resolve()}")
    return output.resolve()

#!/usr/bin/env python3
"""Small OME-Zarr v0.4 helpers for native workflow image volumes."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from datetime import datetime
from typing import Iterable


TIFF_SUFFIXES = {".tif", ".tiff"}
OME_ZARR_SUFFIX = ".ome.zarr"
PYRAMID_DOWNSAMPLE_FACTORS = (1, 2, 4, 8, 16)


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


def default_ome_zarr_compressor():
    try:
        from numcodecs import Blosc
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'numcodecs' for OME-Zarr compression") from exc

    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


def pyramid_downsample_factors(
    *,
    max_downsample: int = 16,
    downsample_factors: Iterable[int] = PYRAMID_DOWNSAMPLE_FACTORS,
) -> tuple[int, ...]:
    max_downsample = max(1, int(max_downsample))
    factors = tuple(int(factor) for factor in downsample_factors if int(factor) <= max_downsample)
    return factors or (1,)


def multiscales_metadata(
    layer_name: str,
    downsample_factors: Iterable[int] = PYRAMID_DOWNSAMPLE_FACTORS,
    *,
    max_downsample: int = 16,
    axes: Iterable[str] = ("z", "y", "x"),
) -> dict:
    axis_names = tuple(str(axis) for axis in axes)
    if len(axis_names) != 3:
        raise ValueError(f"OME-Zarr workflow volumes must have three axes, got {axis_names}")
    factors = pyramid_downsample_factors(
        max_downsample=max_downsample,
        downsample_factors=downsample_factors,
    )
    datasets = [
        {
            "path": str(level),
            "coordinateTransformations": [
                {
                    "type": "scale",
                    "scale": [
                        int(factor) if axis_name in ("x", "y") else 1
                        for axis_name in axis_names
                    ],
                }
            ],
        }
        for level, factor in enumerate(factors)
    ]
    return {
        "multiscales": [
            {
                "version": "0.4",
                "name": layer_name,
                "axes": [
                    {"name": axis_name, "type": "space"}
                    for axis_name in axis_names
                ],
                "datasets": datasets,
            }
        ]
    }


def downsample_xy(array, factor: int):
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"downsample factor must be >= 1, got {factor}")
    if factor == 1:
        return array
    return array[:, ::factor, ::factor]


def _normalise_chunks(chunks, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if chunks is None:
        return (min(16, shape[0]), min(256, shape[1]), min(256, shape[2]))
    normalised = []
    for axis_chunks, axis_size in zip(chunks, shape, strict=True):
        if isinstance(axis_chunks, (tuple, list)):
            chunk_size = int(axis_chunks[0])
        else:
            chunk_size = int(axis_chunks)
        normalised.append(max(1, min(chunk_size, int(axis_size))))
    return tuple(normalised)


def _downsampled_chunks(base_chunks: tuple[int, int, int], shape: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    return _downsampled_chunks_for_axes(
        base_chunks,
        shape,
        factor,
        downsample_axes=(1, 2),
    )


def _downsampled_chunks_for_axes(
    base_chunks: tuple[int, int, int],
    shape: tuple[int, int, int],
    factor: int,
    *,
    downsample_axes: Iterable[int],
) -> tuple[int, int, int]:
    axes = set(int(axis) for axis in downsample_axes)
    return tuple(
        max(
            1,
            min(
                max(1, int(chunk) // int(factor)) if index in axes else int(chunk),
                int(axis_size),
            ),
        )
        for index, (chunk, axis_size) in enumerate(zip(base_chunks, shape, strict=True))
    )


def _downsampled_shape_xy(shape: tuple[int, int, int], factor: int) -> tuple[int, int, int]:
    return _downsampled_shape_for_axes(shape, factor, downsample_axes=(1, 2))


def _downsampled_shape_for_axes(
    shape: tuple[int, int, int],
    factor: int,
    *,
    downsample_axes: Iterable[int],
) -> tuple[int, int, int]:
    axes = set(int(axis) for axis in downsample_axes)
    return tuple(
        max(1, ((int(axis_size) - 1) // int(factor)) + 1)
        if index in axes
        else int(axis_size)
        for index, axis_size in enumerate(shape)
    )


def _axis_names(zarr_path: Path) -> tuple[str, str, str]:
    attrs_path = zarr_path / ".zattrs"
    if not attrs_path.exists():
        return ("z", "y", "x")
    attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    multiscales = attrs.get("multiscales") or []
    if not multiscales:
        return ("z", "y", "x")
    axes = multiscales[0].get("axes") or []
    names = tuple(str(axis.get("name", "")) for axis in axes)
    return names if len(names) == 3 and all(names) else ("z", "y", "x")


def _downsample_axes_for_pyramid(axis_names: Iterable[str]) -> tuple[int, ...]:
    return tuple(
        index
        for index, axis_name in enumerate(axis_names)
        if axis_name in ("x", "y")
    )


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
    max_downsample: int = 16,
    axes: Iterable[str] = ("z", "y", "x"),
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
            multiscales_metadata(
                layer_name or image_stem(zarr_path),
                max_downsample=max_downsample,
                axes=axes,
            ),
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
        compressor=default_ome_zarr_compressor(),
    )


def write_downsampled_pyramid(
    path: Path | str,
    *,
    max_downsample: int = 16,
    downsample_factors: Iterable[int] = PYRAMID_DOWNSAMPLE_FACTORS,
) -> None:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'zarr' for OME-Zarr pyramid generation") from exc

    zarr_path = Path(path)
    source = zarr.open(str(zarr_path / "0"), mode="r")
    source_shape = tuple(int(axis) for axis in source.shape)
    if len(source_shape) != 3:
        raise ValueError(f"OME-Zarr pyramid source must be 3-D, got shape {source_shape}")
    source_chunks = _normalise_chunks(getattr(source, "chunks", None), source_shape)
    dtype = getattr(source, "dtype", None)
    downsample_axes = _downsample_axes_for_pyramid(_axis_names(zarr_path))

    for level, factor in enumerate(downsample_factors):
        factor = int(factor)
        if level == 0 or factor == 1:
            continue
        if factor > int(max_downsample):
            continue
        shape = _downsampled_shape_for_axes(source_shape, factor, downsample_axes=downsample_axes)
        chunks = _downsampled_chunks_for_axes(
            source_chunks,
            shape,
            factor,
            downsample_axes=downsample_axes,
        )
        log_progress(
            "Writing OME-Zarr pyramid level: "
            f"path={zarr_path / str(level)}, downsample={factor}x, "
            f"shape={shape}, chunks={chunks}"
        )
        target = zarr.open(
            str(zarr_path / str(level)),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=default_ome_zarr_compressor(),
        )
        level_start = time.perf_counter()
        last_progress = level_start
        chunks_written = 0
        total_chunks = ((int(shape[0]) - 1) // int(chunks[0])) + 1
        for chunks_written, z_start in enumerate(range(0, shape[0], chunks[0]), start=1):
            z_stop = min(z_start + chunks[0], shape[0])
            target_selection = [slice(None), slice(None), slice(None)]
            target_selection[0] = slice(z_start, z_stop)
            source_selection = [slice(None), slice(None), slice(None)]
            if 0 in downsample_axes:
                source_selection[0] = slice(z_start * factor, z_stop * factor, factor)
            else:
                source_selection[0] = slice(z_start, z_stop)
            for axis in range(1, 3):
                if axis in downsample_axes:
                    source_selection[axis] = slice(None, None, factor)
            target[tuple(target_selection)] = source[tuple(source_selection)]
            now = time.perf_counter()
            if chunks_written == total_chunks or now - last_progress >= 60:
                log_progress(
                    "OME-Zarr pyramid level progress: "
                    f"path={zarr_path / str(level)}, downsample={factor}x, "
                    f"chunks={chunks_written}/{total_chunks}, "
                    f"elapsed={now - level_start:.2f}s"
                )
                last_progress = now
        log_progress(
            "Finished OME-Zarr pyramid level: "
            f"path={zarr_path / str(level)}, downsample={factor}x, "
            f"chunks_written={chunks_written}, elapsed={time.perf_counter() - level_start:.2f}s"
        )


def write_ome_zarr_array(
    path: Path | str,
    array,
    *,
    chunks=None,
    layer_name: str | None = None,
    max_downsample: int = 16,
) -> Path:
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
        max_downsample=max_downsample,
    )
    zarr_array[:] = array
    write_downsampled_pyramid(output, max_downsample=max_downsample)
    log_progress(f"Finished OME-Zarr volume: {output.resolve()}")
    return output.resolve()

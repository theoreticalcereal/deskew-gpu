#!/usr/bin/env python3
"""Chunked deskew/top-view TIFF writer.

This follows the existing MATLAB geometry but avoids materialising the
resized/rotated top-view volume.  The final TIFF is written one output page at
a time.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import time

import numpy as np
import tifffile

from ome_zarr_io import (
    create_ome_zarr_array,
    discover_image_volumes,
    image_stem,
    is_ome_zarr_path,
    log_progress,
    open_ome_zarr_array,
    write_downsampled_pyramid,
)


def _selected_input_dir(image_path: str, cell_name: str | None) -> Path:
    base = Path(image_path)
    return base / cell_name if cell_name else base


def _discover_inputs(input_dir: Path) -> list[Path]:
    paths = discover_image_volumes(input_dir)
    if not paths:
        raise FileNotFoundError(f"No TIFF or OME-Zarr volumes found in {input_dir}")
    return paths


def _open_volume(path: Path) -> np.ndarray:
    try:
        import dask.array as da
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'dask' for lazy deskew input loading") from exc

    if is_ome_zarr_path(path):
        log_progress(f"Opening OME-Zarr input volume: {path}")
        volume = open_ome_zarr_array(path, mode="r")
        chunks = getattr(volume, "chunks", None) or _default_dask_chunks(volume.shape)
        array = da.from_array(volume, chunks=chunks)
        if array.ndim != 3:
            raise ValueError(f"Expected a 3-D OME-Zarr volume at {path}, got {array.shape}")
        return array

    try:
        log_progress(f"Opening TIFF input volume with memmap: {path}")
        volume = tifffile.memmap(str(path), mode="r")
    except Exception:
        log_progress(f"TIFF memmap failed; reading full TIFF volume: {path}")
        volume = tifffile.imread(str(path))
    array = da.from_array(volume, chunks=_default_dask_chunks(volume.shape))
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2-D image or 3-D stack at {path}, got {array.shape}")
    return array


def _default_dask_chunks(shape) -> tuple[int, ...]:
    if len(shape) == 2:
        return (min(512, int(shape[0])), min(512, int(shape[1])))
    if len(shape) == 3:
        return (min(16, int(shape[0])), min(512, int(shape[1])), min(512, int(shape[2])))
    return tuple(max(1, int(axis)) for axis in shape)


def _compute_lazy_array(array):
    return array.compute() if hasattr(array, "compute") else array


def _materialize_plane(volume_zyx, z_index: int) -> np.ndarray:
    return np.asarray(_compute_lazy_array(volume_zyx[int(z_index)]))


def _materialize_volume(
    volume_zyx,
    message: str = "Materializing lazy input volume for GPU transfer",
) -> np.ndarray:
    log_progress(message)
    volume = np.asarray(_compute_lazy_array(volume_zyx))
    if volume.dtype.byteorder not in ("=", "|"):
        volume = volume.astype(volume.dtype.newbyteorder("="), copy=False)
    return np.ascontiguousarray(volume)


def _resize_source_z(output_z: np.ndarray, source_z_size: int, scaled_z_size: int) -> np.ndarray:
    # Match image-resize center mapping closely enough for the MATLAB top-view
    # path: output pixel centers map into input pixel centers.
    source = ((output_z.astype(np.float64) + 0.5) * source_z_size / scaled_z_size) - 0.5
    return np.clip(source, 0.0, float(source_z_size - 1))


def _build_rotation_lookup(
    *,
    shear_y: int,
    x_size: int,
    x_out: int,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_out = np.arange(shear_y, dtype=np.float64)
    center_y = (shear_y - 1) / 2.0
    center_x = (x_size - 1) / 2.0
    theta = math.radians(-angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    yy = y_out - center_y
    xx = float(x_out) - center_x
    src_y = (cos_t * yy) - (sin_t * xx) + center_y
    src_x = (sin_t * yy) + (cos_t * xx) + center_x
    src_y_i = np.rint(src_y).astype(np.int64)
    src_x_i = np.rint(src_x).astype(np.int64)
    valid = (
        (src_y_i >= 0)
        & (src_y_i < shear_y)
        & (src_x_i >= 0)
        & (src_x_i < x_size)
    )
    return src_y_i, src_x_i, valid


def _resolve_deskew_backend(value: str) -> str:
    backend = str(value or "cpu_blocked").strip().lower().replace("-", "_")
    aliases = {
        "gpu": "gpu",
        "cuda": "gpu",
        "cpu": "cpu_blocked",
        "cpu_blocked": "cpu_blocked",
        "cpu_block": "cpu_blocked",
    }
    if backend not in aliases:
        raise ValueError("deskew_backend must be one of: gpu, cuda, cpu, cpu_blocked")
    return aliases[backend]


_GPU_KERNEL = None
_GPU_AFFINE_KERNEL = None
_GPU_AFFINE_KERNEL_FLOAT32 = None


@dataclass(frozen=True)
class ClearExAffineGeometry:
    matrix_xyz: np.ndarray
    offset_xyz: np.ndarray
    inverse_matrix_xyz: np.ndarray
    inverse_offset_xyz: np.ndarray
    output_origin_xyz: tuple[float, float, float]
    output_shape_zyx: tuple[int, int, int]
    voxel_size_um_zyx: tuple[float, float, float]
    applied_rotation_deg_xyz: tuple[float, float, float]
    shear_yz: float


def _rotation_matrix_x(deg_x: float) -> np.ndarray:
    theta = math.radians(float(deg_x))
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)],
        ],
        dtype=np.float64,
    )


def _resolve_deskew_output_dtype(output_dtype: str | np.dtype) -> np.dtype:
    dtype = np.dtype(output_dtype)
    if dtype.name not in {"uint16", "float32"}:
        raise ValueError("deskew_output_dtype must be one of: uint16, float32")
    return dtype


def _coerce_bool(value: bool | str | int) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _cast_resampled_value(value: float, *, output_dtype: np.dtype) -> float | int:
    if np.issubdtype(output_dtype, np.integer):
        info = np.iinfo(output_dtype)
        return int(min(float(info.max), max(float(info.min), math.floor(float(value) + 0.5))))
    return float(value)


def _source_x_bounds_for_output_tile(
    output_x_start: int,
    output_x_stop: int,
    source_x_size: int,
) -> tuple[int, int]:
    start = max(0, int(output_x_start) - 1)
    stop = min(int(source_x_size), int(output_x_stop) + 1)
    if stop <= start:
        raise ValueError(
            f"Invalid ClearEx-affine source X tile bounds for output "
            f"{output_x_start}:{output_x_stop} and source size {source_x_size}"
        )
    return start, stop


def _clearex_affine_geometry(
    *,
    source_shape_zyx: tuple[int, int, int],
    dx: float,
    dz: float,
    angle: float,
    flip: int,
    affine_rotate: bool = False,
) -> ClearExAffineGeometry:
    z_size, y_size, x_size = (int(v) for v in source_shape_zyx)
    z_um, y_um, x_um = (float(dz), float(dx), float(dx))
    flip_sign = float(flip)
    # This follows ClearEx's shear-transform convention: shear in physical YZ.
    # The additional X rotation is optional because it rotates the Z/Y view.
    shear_yz = flip_sign * math.sin(math.radians(float(angle)))
    rotation_deg_x = -flip_sign * float(angle) if _coerce_bool(affine_rotate) else 0.0
    shear_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, shear_yz],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    matrix_xyz = _rotation_matrix_x(rotation_deg_x) @ shear_matrix
    center_xyz = np.asarray(
        [
            ((x_size - 1) * x_um) / 2.0,
            ((y_size - 1) * y_um) / 2.0,
            ((z_size - 1) * z_um) / 2.0,
        ],
        dtype=np.float64,
    )
    offset_xyz = center_xyz - (matrix_xyz @ center_xyz)
    x_max = (x_size - 1) * x_um
    y_max = (y_size - 1) * y_um
    z_max = (z_size - 1) * z_um
    corners_xyz = np.asarray(
        [
            [x, y, z]
            for x in (0.0, float(x_max))
            for y in (0.0, float(y_max))
            for z in (0.0, float(z_max))
        ],
        dtype=np.float64,
    )
    transformed_xyz = (corners_xyz @ matrix_xyz.T) + offset_xyz
    min_xyz = np.min(transformed_xyz, axis=0)
    max_xyz = np.max(transformed_xyz, axis=0)
    output_shape_zyx = (
        max(1, int(math.floor((max_xyz[2] - min_xyz[2]) / z_um)) + 1),
        max(1, int(math.floor((max_xyz[1] - min_xyz[1]) / y_um)) + 1),
        max(1, int(math.floor((max_xyz[0] - min_xyz[0]) / x_um)) + 1),
    )
    inverse_matrix_xyz = np.linalg.inv(matrix_xyz)
    inverse_offset_xyz = -inverse_matrix_xyz @ offset_xyz
    return ClearExAffineGeometry(
        matrix_xyz=matrix_xyz,
        offset_xyz=offset_xyz,
        inverse_matrix_xyz=inverse_matrix_xyz,
        inverse_offset_xyz=inverse_offset_xyz,
        output_origin_xyz=(float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])),
        output_shape_zyx=output_shape_zyx,
        voxel_size_um_zyx=(z_um, y_um, x_um),
        applied_rotation_deg_xyz=(float(rotation_deg_x), 0.0, 0.0),
        shear_yz=float(shear_yz),
    )


def _write_clearex_affine_metadata(output_path: Path, geometry: ClearExAffineGeometry) -> None:
    attrs_path = output_path / ".zattrs"
    if not attrs_path.exists():
        return
    import json

    attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    attrs["clearex_affine"] = {
        "voxel_size_um_zyx": [float(v) for v in geometry.voxel_size_um_zyx],
        "output_origin_xyz_um": [float(v) for v in geometry.output_origin_xyz],
        "affine_matrix_xyz": geometry.matrix_xyz.tolist(),
        "affine_offset_xyz_um": geometry.offset_xyz.tolist(),
        "inverse_affine_matrix_xyz": geometry.inverse_matrix_xyz.tolist(),
        "inverse_affine_offset_xyz_um": geometry.inverse_offset_xyz.tolist(),
        "applied_rotation_deg_xyz": [
            float(v) for v in geometry.applied_rotation_deg_xyz
        ],
        "shear_yz": float(geometry.shear_yz),
    }
    attrs_path.write_text(json.dumps(attrs, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_gpu_kernel():
    global _GPU_KERNEL
    try:
        from numba import cuda
    except ImportError as exc:
        raise RuntimeError(
            "deskew_backend=gpu requires numba in the decon runtime; "
            "use --deskew_backend cpu_blocked to run the CPU reference path"
        ) from exc

    if not cuda.is_available():
        raise RuntimeError(
            "deskew_backend=gpu requested, but no CUDA device is available; "
            "use --deskew_backend cpu_blocked to run the CPU reference path"
        )

    if _GPU_KERNEL is None:

        @cuda.jit(device=True)
        def sample_sheared(volume, offsets, z_index, src_y_i, src_x_i, max_yoffset, y_size, z_size):
            raw_y = src_y_i - offsets[z_index] - max_yoffset
            value = 0.0
            if raw_y >= 0 and raw_y < y_size:
                value = float(volume[z_index, raw_y, src_x_i])
            if z_index >= z_size - 1:
                return value

            raw_y_next = src_y_i - offsets[z_index + 1] - max_yoffset
            next_value = 0.0
            if raw_y_next >= 0 and raw_y_next < y_size:
                next_value = float(volume[z_index + 1, raw_y_next, src_x_i])
            return 0.5 * (value + next_value)

        @cuda.jit
        def cuda_deskew_top_view(
            volume,
            offsets,
            output,
            x_start,
            z_size,
            y_size,
            x_size,
            shear_y,
            scaled_z,
            max_yoffset,
            cos_t,
            sin_t,
            center_y,
            center_x,
        ):
            local_x, y_out, z_out = cuda.grid(3)
            if local_x >= output.shape[0] or y_out >= shear_y or z_out >= scaled_z:
                return

            x_out = x_start + local_x
            yy = float(y_out) - center_y
            xx = float(x_out) - center_x
            src_y = (cos_t * yy) - (sin_t * xx) + center_y
            src_x = (sin_t * yy) + (cos_t * xx) + center_x
            src_y_i = int(math.floor(src_y + 0.5))
            src_x_i = int(math.floor(src_x + 0.5))
            if src_y_i < 0 or src_y_i >= shear_y or src_x_i < 0 or src_x_i >= x_size:
                output[local_x, y_out, z_out] = 0
                return

            source_z = ((float(z_out) + 0.5) * float(z_size) / float(scaled_z)) - 0.5
            if source_z < 0.0:
                source_z = 0.0
            max_source_z = float(z_size - 1)
            if source_z > max_source_z:
                source_z = max_source_z
            z0 = int(math.floor(source_z))
            z1 = z0 + 1
            if z1 >= z_size:
                z1 = z_size - 1
            weight = source_z - float(z0)

            a = sample_sheared(volume, offsets, z0, src_y_i, src_x_i, max_yoffset, y_size, z_size)
            b = sample_sheared(volume, offsets, z1, src_y_i, src_x_i, max_yoffset, y_size, z_size)
            value = ((1.0 - weight) * a) + (weight * b)
            if value < 0.0:
                output[local_x, y_out, z_out] = 0
            elif value > 65535.0:
                output[local_x, y_out, z_out] = 65535
            else:
                output[local_x, y_out, z_out] = int(math.floor(value + 0.5))

        _GPU_KERNEL = cuda_deskew_top_view
    return cuda, _GPU_KERNEL


def _shear_column(
    volume_zyx: np.ndarray,
    *,
    source_y: np.ndarray,
    source_x: np.ndarray,
    valid_yx: np.ndarray,
    z_index: int,
    offsets: np.ndarray,
    max_yoffset: int,
) -> np.ndarray:
    z_index = int(z_index)
    y_size = int(volume_zyx.shape[1])
    out = np.zeros(source_y.shape, dtype=np.float32)

    def sample_one(z: int) -> np.ndarray:
        raw_y = source_y - int(offsets[z]) - int(max_yoffset)
        valid = valid_yx & (raw_y >= 0) & (raw_y < y_size)
        values = np.zeros(source_y.shape, dtype=np.float32)
        if np.any(valid):
            plane = _materialize_plane(volume_zyx, z)
            values[valid] = plane[raw_y[valid], source_x[valid]].astype(np.float32)
        return values

    if z_index >= int(volume_zyx.shape[0]) - 1:
        return sample_one(z_index)
    out = 0.5 * (sample_one(z_index) + sample_one(z_index + 1))
    return out


def _write_top_shear(
    volume_zyx: np.ndarray,
    output_path: Path,
    *,
    dx: float,
    dz: float,
    angle: float,
    flip: int,
    z_chunk: int,
    pyramid_max_downsample: int,
) -> tuple[int, int, int]:
    start_time = time.perf_counter()
    z_size, y_size, x_size = (int(v) for v in volume_zyx.shape)
    new_dz = float(dz) * math.cos(math.radians(float(angle)))
    cz = math.floor(z_size / 2) + 1
    z_one_based = np.arange(1, z_size + 1, dtype=np.float64)
    offsets = np.rint(float(flip) * (z_one_based - cz) * (new_dz / float(dx))).astype(np.int64)
    max_yoffset = int(np.max(np.abs(offsets)))
    shear_y = int(y_size + (2 * max_yoffset))

    scale_z = float(dz) * math.sin(math.radians(float(angle))) / float(dx)
    scaled_z = max(1, int(round(z_size * scale_z)))
    output_shape = (shear_y, scaled_z, x_size)
    print(
        "Chunked top-view geometry: "
        f"input_zyx={volume_zyx.shape}, shear_y={shear_y}, "
        f"scaled_z={scaled_z}, output_yzx={output_shape}, "
        f"scale_z={scale_z:.6g}",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ome_zarr = is_ome_zarr_path(output_path)
    if write_ome_zarr:
        log_progress(f"Writing deskew output as OME-Zarr: {output_path}")
        zarr_output = create_ome_zarr_array(
            output_path,
            shape=(scaled_z, shear_y, x_size),
            chunks=(min(16, scaled_z), min(256, shear_y), min(256, x_size)),
            dtype=np.dtype("uint16"),
            layer_name=image_stem(output_path),
            max_downsample=int(pyramid_max_downsample),
        )
        writer_context = nullcontext()
        writer = None
    else:
        log_progress(f"Writing deskew output as TIFF: {output_path}")
        zarr_output = None
        writer_context = tifffile.TiffWriter(str(output_path), bigtiff=True)

    z_positions = np.arange(scaled_z, dtype=np.float64)
    source_z = _resize_source_z(z_positions, z_size, scaled_z)
    z0_all = np.floor(source_z).astype(np.int64)
    z1_all = np.clip(z0_all + 1, 0, z_size - 1)
    wz_all = (source_z - z0_all).astype(np.float32)

    z_chunk = max(1, int(z_chunk))

    def compute_block(x_start: int, x_stop: int) -> tuple[int, np.ndarray]:
        block_start = time.perf_counter()
        lookup_seconds = 0.0
        sampling_seconds = 0.0
        block = np.empty((x_stop - x_start, shear_y, scaled_z), dtype=np.uint16)
        tile = np.empty((shear_y, min(z_chunk, scaled_z)), dtype=np.float32)

        for local_x, x_out in enumerate(range(x_start, x_stop)):
            lookup_start = time.perf_counter()
            src_y, src_x, valid_yx = _build_rotation_lookup(
                shear_y=shear_y,
                x_size=x_size,
                x_out=x_out,
                angle_deg=float(flip) * float(angle),
            )
            lookup_seconds += time.perf_counter() - lookup_start

            page = block[local_x]
            for z_start in range(0, scaled_z, z_chunk):
                z_stop = min(z_start + z_chunk, scaled_z)
                z0 = z0_all[z_start:z_stop]
                z1 = z1_all[z_start:z_stop]
                wz = wz_all[z_start:z_stop]
                chunk_tile = tile[:, : z_stop - z_start]

                sampling_start = time.perf_counter()
                unique_z = sorted(set(z0.tolist()) | set(z1.tolist()))
                columns = {
                    z: _shear_column(
                        volume_zyx,
                        source_y=src_y,
                        source_x=src_x,
                        valid_yx=valid_yx,
                        z_index=z,
                        offsets=offsets,
                        max_yoffset=max_yoffset,
                    )
                    for z in unique_z
                }
                for local_i, (a, b, w) in enumerate(zip(z0, z1, wz, strict=False)):
                    chunk_tile[:, local_i] = ((1.0 - float(w)) * columns[int(a)]) + (
                        float(w) * columns[int(b)]
                    )
                page[:, z_start:z_stop] = np.clip(np.rint(chunk_tile), 0, 65535).astype(np.uint16)
                sampling_seconds += time.perf_counter() - sampling_start

        total_seconds = time.perf_counter() - block_start
        print(
            f"  Computed top-view block {x_start + 1}-{x_stop}/{x_size}: "
            f"lookup={lookup_seconds:.2f}s, "
            f"sampling_interpolation={sampling_seconds:.2f}s, "
            f"total={total_seconds:.2f}s",
            flush=True,
        )
        return x_start, block

    def write_page(page_index: int, page: np.ndarray) -> None:
        if zarr_output is not None:
            zarr_output[:, :, page_index] = page.T
        else:
            writer.write(
                page,
                photometric="minisblack",
                compression=None,
                contiguous=True,
            )

    print(
        "  CPU deskew scheduler: mode=page_serial, pages are computed and written one at a time",
        flush=True,
    )
    with writer_context as writer:
        for x_out in range(x_size):
            _, block = compute_block(x_out, x_out + 1)
            write_start = time.perf_counter()
            write_page(x_out, block[0])
            write_seconds = time.perf_counter() - write_start
            print(
                f"  Wrote top-view page {x_out + 1}/{x_size}: write={write_seconds:.2f}s",
                flush=True,
            )
            if (x_out + 1) % 50 == 0 or (x_out + 1) == x_size:
                log_progress(f"Wrote top-view page {x_out + 1}/{x_size}")
    if write_ome_zarr:
        write_downsampled_pyramid(output_path, max_downsample=int(pyramid_max_downsample))
    log_progress(
        f"Finished top-view deskew output: {output_path} "
        f"in {time.perf_counter() - start_time:.2f}s"
    )
    return output_shape


def _write_top_shear_gpu(
    volume_zyx,
    output_path: Path,
    *,
    dx: float,
    dz: float,
    angle: float,
    flip: int,
    deskew_prefetch: int,
    pyramid_max_downsample: int,
) -> tuple[int, int, int]:
    start_time = time.perf_counter()
    cuda, kernel = _load_gpu_kernel()

    z_size, y_size, x_size = (int(v) for v in volume_zyx.shape)
    new_dz = float(dz) * math.cos(math.radians(float(angle)))
    cz = math.floor(z_size / 2) + 1
    z_one_based = np.arange(1, z_size + 1, dtype=np.float64)
    offsets = np.rint(float(flip) * (z_one_based - cz) * (new_dz / float(dx))).astype(np.int32)
    max_yoffset = int(np.max(np.abs(offsets)))
    shear_y = int(y_size + (2 * max_yoffset))

    scale_z = float(dz) * math.sin(math.radians(float(angle))) / float(dx)
    scaled_z = max(1, int(round(z_size * scale_z)))
    output_shape = (shear_y, scaled_z, x_size)
    print(
        "GPU top-view geometry: "
        f"input_zyx={volume_zyx.shape}, shear_y={shear_y}, "
        f"scaled_z={scaled_z}, output_yzx={output_shape}, "
        f"scale_z={scale_z:.6g}",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ome_zarr = is_ome_zarr_path(output_path)
    if write_ome_zarr:
        log_progress(f"Writing GPU deskew output as OME-Zarr: {output_path}")
        zarr_output = create_ome_zarr_array(
            output_path,
            shape=(scaled_z, shear_y, x_size),
            chunks=(min(16, scaled_z), min(256, shear_y), min(256, x_size)),
            dtype=np.dtype("uint16"),
            layer_name=image_stem(output_path),
            max_downsample=int(pyramid_max_downsample),
        )
        writer_context = nullcontext()
        writer = None
    else:
        log_progress(f"Writing GPU deskew output as TIFF: {output_path}")
        zarr_output = None
        writer_context = tifffile.TiffWriter(str(output_path), bigtiff=True)

    transfer_start = time.perf_counter()
    host_volume = _materialize_volume(volume_zyx)
    device_volume = cuda.to_device(host_volume)
    device_offsets = cuda.to_device(offsets)
    print(
        f"  GPU deskew input transfer: shape={host_volume.shape}, dtype={host_volume.dtype}, "
        f"time={time.perf_counter() - transfer_start:.2f}s",
        flush=True,
    )

    pages_per_batch = max(1, int(deskew_prefetch))
    threads = (4, 8, 8)
    theta = math.radians(-float(flip) * float(angle))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    center_y = (shear_y - 1) / 2.0
    center_x = (x_size - 1) / 2.0

    def write_page(page_index: int, page: np.ndarray) -> None:
        if zarr_output is not None:
            zarr_output[:, :, page_index] = page.T
        else:
            writer.write(
                page,
                photometric="minisblack",
                compression=None,
                contiguous=True,
            )

    with writer_context as writer:
        for x_start in range(0, x_size, pages_per_batch):
            x_stop = min(x_start + pages_per_batch, x_size)
            batch_pages = x_stop - x_start
            batch_start = time.perf_counter()
            device_output = cuda.device_array((batch_pages, shear_y, scaled_z), dtype=np.uint16)
            blocks = (
                math.ceil(batch_pages / threads[0]),
                math.ceil(shear_y / threads[1]),
                math.ceil(scaled_z / threads[2]),
            )
            kernel[blocks, threads](
                device_volume,
                device_offsets,
                device_output,
                int(x_start),
                int(z_size),
                int(y_size),
                int(x_size),
                int(shear_y),
                int(scaled_z),
                int(max_yoffset),
                float(cos_t),
                float(sin_t),
                float(center_y),
                float(center_x),
            )
            cuda.synchronize()
            kernel_seconds = time.perf_counter() - batch_start

            copy_start = time.perf_counter()
            batch = device_output.copy_to_host()
            copy_seconds = time.perf_counter() - copy_start

            write_start = time.perf_counter()
            for local_x, page in enumerate(batch):
                write_page(x_start + local_x, page)
            write_seconds = time.perf_counter() - write_start
            print(
                f"  GPU deskew pages {x_start + 1}-{x_stop}/{x_size}: "
                f"kernel={kernel_seconds:.2f}s, copy={copy_seconds:.2f}s, "
                f"write={write_seconds:.2f}s, total={time.perf_counter() - batch_start:.2f}s",
                flush=True,
            )
            if x_stop % 50 == 0 or x_stop == x_size:
                log_progress(f"Wrote GPU top-view page {x_stop}/{x_size}")

    if write_ome_zarr:
        write_downsampled_pyramid(output_path, max_downsample=int(pyramid_max_downsample))
    log_progress(
        f"Finished GPU top-view deskew output: {output_path} "
        f"in {time.perf_counter() - start_time:.2f}s"
    )
    return output_shape


def _linear_sample_support_normalized(
    volume_zyx: np.ndarray,
    *,
    zf: float,
    yf: float,
    xf: float,
) -> float:
    z_size, y_size, x_size = volume_zyx.shape
    if (
        zf < -0.5
        or zf > float(z_size) - 0.5
        or yf < -0.5
        or yf > float(y_size) - 0.5
        or xf < -0.5
        or xf > float(x_size) - 0.5
    ):
        return 0.0
    z0 = int(math.floor(float(zf)))
    y0 = int(math.floor(float(yf)))
    x0 = int(math.floor(float(xf)))
    wz = float(zf) - float(z0)
    wy = float(yf) - float(y0)
    wx = float(xf) - float(x0)
    value = 0.0
    support = 0.0
    for dz_i in (0, 1):
        zz = z0 + dz_i
        wz_i = (1.0 - wz) if dz_i == 0 else wz
        if zz < 0 or zz >= z_size:
            continue
        for dy_i in (0, 1):
            yy = y0 + dy_i
            wy_i = (1.0 - wy) if dy_i == 0 else wy
            if yy < 0 or yy >= y_size:
                continue
            for dx_i in (0, 1):
                xx = x0 + dx_i
                wx_i = (1.0 - wx) if dx_i == 0 else wx
                if xx < 0 or xx >= x_size:
                    continue
                weight = wz_i * wy_i * wx_i
                support += weight
                value += weight * float(volume_zyx[zz, yy, xx])
    if support <= 1.0e-3:
        return 0.0
    return value / support


def _write_clearex_affine(
    volume_zyx,
    output_path: Path,
    *,
    dx: float,
    dz: float,
    angle: float,
    flip: int,
    z_chunk: int,
    pyramid_max_downsample: int,
    output_dtype: str | np.dtype = "uint16",
    affine_rotate: bool = False,
) -> tuple[int, int, int]:
    start_time = time.perf_counter()
    target_dtype = _resolve_deskew_output_dtype(output_dtype)
    host_volume = _materialize_volume(
        volume_zyx,
        message="Materializing input volume for ClearEx-affine CPU resampling",
    ).astype(np.float32, copy=False)
    geometry = _clearex_affine_geometry(
        source_shape_zyx=tuple(int(v) for v in host_volume.shape),
        dx=float(dx),
        dz=float(dz),
        angle=float(angle),
        flip=int(flip),
        affine_rotate=_coerce_bool(affine_rotate),
    )
    print(
        "ClearEx-affine CPU geometry: "
        f"input_zyx={host_volume.shape}, output_zyx={geometry.output_shape_zyx}, "
        f"shear_yz={geometry.shear_yz:.6g}, "
        f"rotation_x={geometry.applied_rotation_deg_xyz[0]:.6g}",
        flush=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ome_zarr = is_ome_zarr_path(output_path)
    if write_ome_zarr:
        zarr_output = create_ome_zarr_array(
            output_path,
            shape=geometry.output_shape_zyx,
            chunks=(
                min(16, geometry.output_shape_zyx[0]),
                min(256, geometry.output_shape_zyx[1]),
                min(256, geometry.output_shape_zyx[2]),
            ),
            dtype=target_dtype,
            layer_name=image_stem(output_path),
            max_downsample=int(pyramid_max_downsample),
        )
        _write_clearex_affine_metadata(output_path, geometry)
    else:
        zarr_output = None

    z_um, y_um, x_um = geometry.voxel_size_um_zyx
    out_z, out_y, out_x = geometry.output_shape_zyx
    z_chunk = max(1, int(z_chunk))
    for z_start in range(0, out_z, z_chunk):
        z_stop = min(z_start + z_chunk, out_z)
        block = np.zeros((z_stop - z_start, out_y, out_x), dtype=target_dtype)
        for local_z, z_index in enumerate(range(z_start, z_stop)):
            out_z_um = geometry.output_origin_xyz[2] + (float(z_index) * z_um)
            for y_index in range(out_y):
                out_y_um = geometry.output_origin_xyz[1] + (float(y_index) * y_um)
                for x_index in range(out_x):
                    out_x_um = geometry.output_origin_xyz[0] + (float(x_index) * x_um)
                    sx, sy, sz = (
                        geometry.inverse_matrix_xyz
                        @ np.asarray([out_x_um, out_y_um, out_z_um], dtype=np.float64)
                    ) + geometry.inverse_offset_xyz
                    value = _linear_sample_support_normalized(
                        host_volume,
                        zf=float(sz / z_um),
                        yf=float(sy / y_um),
                        xf=float(sx / x_um),
                    )
                    block[local_z, y_index, x_index] = _cast_resampled_value(
                        value,
                        output_dtype=target_dtype,
                    )
        if zarr_output is not None:
            zarr_output[z_start:z_stop, :, :] = block
        else:
            mode = "wb" if z_start == 0 else "ab"
            tifffile.imwrite(output_path, block, bigtiff=True, append=(mode == "ab"))
        log_progress(f"Wrote ClearEx-affine CPU z slices {z_start + 1}-{z_stop}/{out_z}")
    if write_ome_zarr:
        write_downsampled_pyramid(output_path, max_downsample=int(pyramid_max_downsample))
    log_progress(
        f"Finished ClearEx-affine CPU output: {output_path} "
        f"in {time.perf_counter() - start_time:.2f}s"
    )
    return geometry.output_shape_zyx


def _load_gpu_affine_kernel(*, output_dtype: np.dtype):
    global _GPU_AFFINE_KERNEL, _GPU_AFFINE_KERNEL_FLOAT32
    try:
        from numba import cuda
    except ImportError as exc:
        raise RuntimeError(
            "deskew_backend=gpu requires numba; use --deskew_backend cpu_blocked "
            "for the CPU reference path"
        ) from exc

    if not cuda.is_available():
        raise RuntimeError(
            "deskew_backend=gpu requested, but no CUDA device is available; "
            "use --deskew_backend cpu_blocked to run the CPU reference path"
        )

    if _GPU_AFFINE_KERNEL is None or _GPU_AFFINE_KERNEL_FLOAT32 is None:

        @cuda.jit(device=True)
        def affine_sample(
            volume,
            zf,
            yf,
            xf,
            source_z_size,
            source_y_size,
            source_x_size,
            source_z0,
            source_y0,
            source_x0,
        ):
            if (
                zf < -0.5
                or zf > float(source_z_size) - 0.5
                or yf < -0.5
                or yf > float(source_y_size) - 0.5
                or xf < -0.5
                or xf > float(source_x_size) - 0.5
            ):
                return 0.0
            zf = zf - float(source_z0)
            yf = yf - float(source_y0)
            xf = xf - float(source_x0)
            z_size = volume.shape[0]
            y_size = volume.shape[1]
            x_size = volume.shape[2]
            z0 = int(math.floor(zf))
            y0 = int(math.floor(yf))
            x0 = int(math.floor(xf))
            wz = zf - float(z0)
            wy = yf - float(y0)
            wx = xf - float(x0)
            value = 0.0
            support = 0.0
            for dz_i in range(2):
                zz = z0 + dz_i
                wz_i = (1.0 - wz) if dz_i == 0 else wz
                if zz < 0 or zz >= z_size:
                    continue
                for dy_i in range(2):
                    yy = y0 + dy_i
                    wy_i = (1.0 - wy) if dy_i == 0 else wy
                    if yy < 0 or yy >= y_size:
                        continue
                    for dx_i in range(2):
                        xx = x0 + dx_i
                        wx_i = (1.0 - wx) if dx_i == 0 else wx
                        if xx < 0 or xx >= x_size:
                            continue
                        weight = wz_i * wy_i * wx_i
                        support += weight
                        value += weight * float(volume[zz, yy, xx])
            if support <= 1.0e-3:
                return 0.0
            return value / support

        @cuda.jit
        def cuda_clearex_affine_uint16(
            volume,
            output,
            z_start,
            inv_matrix,
            inv_offset,
            out_origin,
            spacing,
            source_z_size,
            source_y_size,
            source_x_size,
            source_z0,
            source_y0,
            source_x0,
            x_start,
        ):
            z_local, y_index, x_local = cuda.grid(3)
            if z_local >= output.shape[0] or y_index >= output.shape[1] or x_local >= output.shape[2]:
                return

            z_index = z_start + z_local
            x_index = x_start + x_local
            out_x = out_origin[0] + (float(x_index) * spacing[2])
            out_y = out_origin[1] + (float(y_index) * spacing[1])
            out_z = out_origin[2] + (float(z_index) * spacing[0])

            src_x = (
                inv_matrix[0, 0] * out_x
                + inv_matrix[0, 1] * out_y
                + inv_matrix[0, 2] * out_z
                + inv_offset[0]
            )
            src_y = (
                inv_matrix[1, 0] * out_x
                + inv_matrix[1, 1] * out_y
                + inv_matrix[1, 2] * out_z
                + inv_offset[1]
            )
            src_z = (
                inv_matrix[2, 0] * out_x
                + inv_matrix[2, 1] * out_y
                + inv_matrix[2, 2] * out_z
                + inv_offset[2]
            )
            value = affine_sample(
                volume,
                src_z / spacing[0],
                src_y / spacing[1],
                src_x / spacing[2],
                source_z_size,
                source_y_size,
                source_x_size,
                source_z0,
                source_y0,
                source_x0,
            )
            if value < 0.0:
                output[z_local, y_index, x_local] = 0
            elif value > 65535.0:
                output[z_local, y_index, x_local] = 65535
            else:
                output[z_local, y_index, x_local] = int(math.floor(value + 0.5))

        @cuda.jit
        def cuda_clearex_affine_float32(
            volume,
            output,
            z_start,
            inv_matrix,
            inv_offset,
            out_origin,
            spacing,
            source_z_size,
            source_y_size,
            source_x_size,
            source_z0,
            source_y0,
            source_x0,
            x_start,
        ):
            z_local, y_index, x_local = cuda.grid(3)
            if z_local >= output.shape[0] or y_index >= output.shape[1] or x_local >= output.shape[2]:
                return

            z_index = z_start + z_local
            x_index = x_start + x_local
            out_x = out_origin[0] + (float(x_index) * spacing[2])
            out_y = out_origin[1] + (float(y_index) * spacing[1])
            out_z = out_origin[2] + (float(z_index) * spacing[0])

            src_x = (
                inv_matrix[0, 0] * out_x
                + inv_matrix[0, 1] * out_y
                + inv_matrix[0, 2] * out_z
                + inv_offset[0]
            )
            src_y = (
                inv_matrix[1, 0] * out_x
                + inv_matrix[1, 1] * out_y
                + inv_matrix[1, 2] * out_z
                + inv_offset[1]
            )
            src_z = (
                inv_matrix[2, 0] * out_x
                + inv_matrix[2, 1] * out_y
                + inv_matrix[2, 2] * out_z
                + inv_offset[2]
            )
            output[z_local, y_index, x_index] = affine_sample(
                volume,
                src_z / spacing[0],
                src_y / spacing[1],
                src_x / spacing[2],
                source_z_size,
                source_y_size,
                source_x_size,
                source_z0,
                source_y0,
                source_x0,
            )

        _GPU_AFFINE_KERNEL = cuda_clearex_affine_uint16
        _GPU_AFFINE_KERNEL_FLOAT32 = cuda_clearex_affine_float32
    if np.dtype(output_dtype).name == "float32":
        return cuda, _GPU_AFFINE_KERNEL_FLOAT32
    return cuda, _GPU_AFFINE_KERNEL


def _write_clearex_affine_gpu(
    volume_zyx,
    output_path: Path,
    *,
    dx: float,
    dz: float,
    angle: float,
    flip: int,
    deskew_prefetch: int,
    pyramid_max_downsample: int,
    output_dtype: str | np.dtype = "uint16",
    affine_rotate: bool = False,
) -> tuple[int, int, int]:
    start_time = time.perf_counter()
    target_dtype = _resolve_deskew_output_dtype(output_dtype)
    cuda, kernel = _load_gpu_affine_kernel(output_dtype=target_dtype)
    host_volume = _materialize_volume(
        volume_zyx,
        message="Materializing input volume for ClearEx-affine GPU source tiling",
    )
    geometry = _clearex_affine_geometry(
        source_shape_zyx=tuple(int(v) for v in host_volume.shape),
        dx=float(dx),
        dz=float(dz),
        angle=float(angle),
        flip=int(flip),
        affine_rotate=_coerce_bool(affine_rotate),
    )
    print(
        "ClearEx-affine GPU geometry: "
        f"input_zyx={host_volume.shape}, output_zyx={geometry.output_shape_zyx}, "
        f"shear_yz={geometry.shear_yz:.6g}, "
        f"rotation_x={geometry.applied_rotation_deg_xyz[0]:.6g}",
        flush=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zarr_output = create_ome_zarr_array(
        output_path,
        shape=geometry.output_shape_zyx,
        chunks=(
            min(16, geometry.output_shape_zyx[0]),
            min(256, geometry.output_shape_zyx[1]),
            min(256, geometry.output_shape_zyx[2]),
        ),
        dtype=target_dtype,
        layer_name=image_stem(output_path),
        max_downsample=int(pyramid_max_downsample),
    )
    _write_clearex_affine_metadata(output_path, geometry)

    device_matrix = cuda.to_device(np.asarray(geometry.inverse_matrix_xyz, dtype=np.float64))
    device_offset = cuda.to_device(np.asarray(geometry.inverse_offset_xyz, dtype=np.float64))
    device_origin = cuda.to_device(np.asarray(geometry.output_origin_xyz, dtype=np.float64))
    device_spacing = cuda.to_device(np.asarray(geometry.voxel_size_um_zyx, dtype=np.float64))
    out_z, out_y, out_x = geometry.output_shape_zyx
    source_z, source_y, source_x = (int(v) for v in host_volume.shape)
    z_batch = max(1, min(int(deskew_prefetch), int(out_z)))
    x_tile = min(256, int(out_x))
    threads = (4, 8, 8)
    for z_start in range(0, out_z, z_batch):
        z_stop = min(z_start + z_batch, out_z)
        local_z = z_stop - z_start
        for x_start in range(0, out_x, x_tile):
            x_stop = min(x_start + x_tile, out_x)
            local_x = x_stop - x_start
            source_x_start, source_x_stop = _source_x_bounds_for_output_tile(
                x_start,
                x_stop,
                source_x,
            )
            source_tile = np.ascontiguousarray(host_volume[:, :, source_x_start:source_x_stop])
            device_volume = cuda.to_device(source_tile)
            device_output = cuda.device_array((local_z, out_y, local_x), dtype=target_dtype)
            blocks = (
                math.ceil(local_z / threads[0]),
                math.ceil(out_y / threads[1]),
                math.ceil(local_x / threads[2]),
            )
            batch_start = time.perf_counter()
            kernel[blocks, threads](
                device_volume,
                device_output,
                int(z_start),
                device_matrix,
                device_offset,
                device_origin,
                device_spacing,
                int(source_z),
                int(source_y),
                int(source_x),
                0,
                0,
                int(source_x_start),
                int(x_start),
            )
            cuda.synchronize()
            block = device_output.copy_to_host()
            zarr_output[z_start:z_stop, :, x_start:x_stop] = block
            del device_volume
            del device_output
            print(
                f"  ClearEx-affine GPU z slices {z_start + 1}-{z_stop}/{out_z}, "
                f"x {x_start + 1}-{x_stop}/{out_x}: "
                f"source_x={source_x_start}:{source_x_stop}, "
                f"total={time.perf_counter() - batch_start:.2f}s",
                flush=True,
            )
    write_downsampled_pyramid(output_path, max_downsample=int(pyramid_max_downsample))
    log_progress(
        f"Finished ClearEx-affine GPU output: {output_path} "
        f"in {time.perf_counter() - start_time:.2f}s"
    )
    return geometry.output_shape_zyx


def run_chunked_deskew(
    *,
    image_path: str,
    cell_name: str,
    dx: float,
    dz: float,
    angle: float,
    flip: int,
    output_dir: str,
    deskew_backend: str,
    deskew_geometry: str,
    z_chunk: int,
    deskew_prefetch: int,
    pyramid_max_downsample: int,
    deskew_output_dtype: str = "uint16",
    deskew_affine_rotate: bool | str = False,
) -> None:
    run_start = time.perf_counter()
    input_dir = _selected_input_dir(image_path, cell_name)
    output_root = Path(output_dir)
    top_shear_dir = output_root / "Top_shear"
    top_shear_dir.mkdir(parents=True, exist_ok=True)
    original_name_map = input_dir / "original_filenames.tsv"
    if original_name_map.exists():
        shutil.copy2(original_name_map, top_shear_dir / "original_filenames.tsv")
        log_progress(f"Copied original filename map to {top_shear_dir}")

    inputs = _discover_inputs(input_dir)
    log_progress(f"Chunked deskew discovered {len(inputs)} input volume(s) in {input_dir}")
    for index, path in enumerate(inputs, start=1):
        volume_start = time.perf_counter()
        log_progress(f"Processing deskew input {index}/{len(inputs)}: {path.name}")
        volume = _open_volume(path)
        log_progress(f"Opened {path.name}: shape={volume.shape}, dtype={volume.dtype}")
        geometry_mode = str(deskew_geometry or "top_view").strip().lower().replace("-", "_")
        if geometry_mode not in {"top_view", "clearex_affine"}:
            raise ValueError("deskew_geometry must be one of: top_view, clearex_affine")
        output_dtype = _resolve_deskew_output_dtype(deskew_output_dtype)
        if geometry_mode == "top_view" and output_dtype.name != "uint16":
            raise ValueError("deskew_output_dtype=float32 is only supported with clearex_affine geometry")
        output_name = (
            f"{image_stem(path)}.ome.zarr"
            if is_ome_zarr_path(path) or geometry_mode == "clearex_affine"
            else f"{image_stem(path)}.tif"
        )
        backend = _resolve_deskew_backend(deskew_backend)
        if geometry_mode == "clearex_affine" and backend == "gpu":
            output_shape = _write_clearex_affine_gpu(
                volume,
                top_shear_dir / output_name,
                dx=float(dx),
                dz=float(dz),
                angle=float(angle),
                flip=int(flip),
                deskew_prefetch=int(deskew_prefetch),
                pyramid_max_downsample=int(pyramid_max_downsample),
                output_dtype=output_dtype,
                affine_rotate=_coerce_bool(deskew_affine_rotate),
            )
            note_prefix = "ClearEx-affine GPU deskew output. "
            note_shape = f"output_zyx={output_shape}; "
        elif geometry_mode == "clearex_affine":
            output_shape = _write_clearex_affine(
                volume,
                top_shear_dir / output_name,
                dx=float(dx),
                dz=float(dz),
                angle=float(angle),
                flip=int(flip),
                z_chunk=int(z_chunk),
                pyramid_max_downsample=int(pyramid_max_downsample),
                output_dtype=output_dtype,
                affine_rotate=_coerce_bool(deskew_affine_rotate),
            )
            note_prefix = "ClearEx-affine CPU deskew output. "
            note_shape = f"output_zyx={output_shape}; "
        elif backend == "gpu":
            output_shape = _write_top_shear_gpu(
                volume,
                top_shear_dir / output_name,
                dx=float(dx),
                dz=float(dz),
                angle=float(angle),
                flip=int(flip),
                deskew_prefetch=int(deskew_prefetch),
                pyramid_max_downsample=int(pyramid_max_downsample),
            )
            note_prefix = "Chunked top-view deskew output. "
            note_shape = (
                f"output_yzx={output_shape}; "
                f"ome_zarr_level0_zyx={(output_shape[1], output_shape[0], output_shape[2])}; "
            )
        else:
            output_shape = _write_top_shear(
                volume,
                top_shear_dir / output_name,
                dx=float(dx),
                dz=float(dz),
                angle=float(angle),
                flip=int(flip),
                z_chunk=int(z_chunk),
                pyramid_max_downsample=int(pyramid_max_downsample),
            )
            note_prefix = "Chunked top-view deskew output. "
            note_shape = (
                f"output_yzx={output_shape}; "
                f"ome_zarr_level0_zyx={(output_shape[1], output_shape[0], output_shape[2])}; "
            )
        (top_shear_dir / "note.txt").write_text(
            note_prefix
            + note_shape
            + "z pixel = x(y) pixel.\n"
        )
        log_progress(
            f"Finished deskew input {path.name}: output={output_name}, "
            f"elapsed={time.perf_counter() - volume_start:.2f}s"
        )
    log_progress(f"Chunked deskew complete in {time.perf_counter() - run_start:.2f}s")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunked deskew/top-view TIFF writer.")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--cell_name", default="")
    parser.add_argument("--cell_index", default="")
    parser.add_argument("--dx", type=float, required=True)
    parser.add_argument("--dz", type=float, required=True)
    parser.add_argument("--angle", type=float, required=True)
    parser.add_argument("--flip", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--z_chunk", type=int, default=256)
    parser.add_argument(
        "--deskew_backend",
        default="cpu_blocked",
        choices=["gpu", "cpu", "cpu_blocked", "cpu_block", "cuda"],
    )
    parser.add_argument(
        "--deskew_geometry",
        default="top_view",
        choices=["top_view", "clearex_affine"],
    )
    parser.add_argument("--deskew_prefetch", type=int, default=64)
    parser.add_argument("--pyramid_max_downsample", type=int, default=16)
    parser.add_argument(
        "--deskew_output_dtype",
        default="uint16",
        choices=["uint16", "float32"],
        help="Output dtype; float32 matches ClearEx affine output and is only supported with clearex_affine",
    )
    parser.add_argument(
        "--deskew_affine_rotate",
        default="false",
        help="For clearex_affine, also rotate around X by -flip*angle after shearing.",
    )
    args = parser.parse_args(argv)
    if args.cell_index:
        print("cell_index is accepted for compatibility but ignored by chunked deskew.", flush=True)
    run_chunked_deskew(
        image_path=args.image_path,
        cell_name=args.cell_name,
        dx=args.dx,
        dz=args.dz,
        angle=args.angle,
        flip=args.flip,
        output_dir=args.output_dir,
        deskew_backend=_resolve_deskew_backend(args.deskew_backend),
        deskew_geometry=args.deskew_geometry,
        z_chunk=max(1, args.z_chunk),
        deskew_prefetch=max(1, args.deskew_prefetch),
        pyramid_max_downsample=max(1, args.pyramid_max_downsample),
        deskew_output_dtype=args.deskew_output_dtype,
        deskew_affine_rotate=_coerce_bool(args.deskew_affine_rotate),
    )


if __name__ == "__main__":
    main()

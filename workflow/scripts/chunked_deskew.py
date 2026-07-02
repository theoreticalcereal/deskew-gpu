#!/usr/bin/env python3
"""Chunked deskew/top-view TIFF writer.

This follows the existing MATLAB geometry but avoids materialising the
resized/rotated top-view volume.  The final TIFF is written one output page at
a time.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from concurrent import futures
import math
from pathlib import Path
import shutil
import time
from typing import Callable

import numpy as np
import tifffile

from ome_zarr_io import (
    create_ome_zarr_array,
    discover_image_volumes,
    image_stem,
    is_ome_zarr_path,
    log_progress,
    open_ome_zarr_array,
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


def _materialize_volume(volume_zyx) -> np.ndarray:
    log_progress("Materializing lazy input volume for GPU transfer")
    return np.ascontiguousarray(_compute_lazy_array(volume_zyx))


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


def _iter_x_blocks(x_size: int, deskew_x_block: int):
    block_size = max(1, int(deskew_x_block))
    for start in range(0, max(0, int(x_size)), block_size):
        yield start, min(start + block_size, int(x_size))


def _flush_ready_blocks(
    write_buffer: dict[int, np.ndarray],
    next_write: int,
    write_page: Callable[[int, np.ndarray], None],
) -> tuple[int, float]:
    write_start = time.perf_counter()
    while next_write in write_buffer:
        block = write_buffer.pop(next_write)
        for local_x, page in enumerate(block):
            write_page(next_write + local_x, page)
        next_write += int(block.shape[0])
    return next_write, time.perf_counter() - write_start


def _resolve_deskew_backend(value: str) -> str:
    backend = str(value or "gpu").strip().lower().replace("-", "_")
    aliases = {
        "gpu": "gpu",
        "cuda": "gpu",
        "cpu": "cpu_blocked",
        "cpu_blocked": "cpu_blocked",
        "cpu_block": "cpu_blocked",
    }
    if backend not in aliases:
        raise ValueError("deskew_backend must be one of: gpu, cpu_blocked")
    return aliases[backend]


_GPU_KERNEL = None


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
    deskew_workers: int,
    deskew_prefetch: int,
    deskew_x_block: int,
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
            shape=(x_size, shear_y, scaled_z),
            chunks=(1, min(256, shear_y), min(256, scaled_z)),
            dtype=np.dtype("uint16"),
            layer_name=image_stem(output_path),
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

    deskew_workers = max(1, int(deskew_workers))
    deskew_prefetch = max(1, int(deskew_prefetch))
    deskew_prefetch = max(deskew_workers, deskew_prefetch)
    deskew_x_block = max(1, int(deskew_x_block))
    pending_block_limit = max(deskew_workers, math.ceil(deskew_prefetch / deskew_x_block))
    x_blocks = list(_iter_x_blocks(x_size, deskew_x_block))
    print(
        f"  Chunked deskew scheduler: workers={deskew_workers}, "
        f"prefetch_pages={deskew_prefetch}, x_block={deskew_x_block}, "
        f"pending_block_limit={pending_block_limit}, "
        f"blocks={len(x_blocks)}, pages={x_size}",
        flush=True,
    )

    def write_page(page_index: int, page: np.ndarray) -> None:
        if zarr_output is not None:
            zarr_output[page_index, :, :] = page
        else:
            writer.write(
                page,
                photometric="minisblack",
                compression=None,
                contiguous=True,
            )

    with writer_context as writer:
        pending_blocks: set[futures.Future[tuple[int, np.ndarray]]] = set()
        write_buffer: dict[int, np.ndarray] = {}
        next_submit = 0
        next_write = 0
        completed = 0
        last_heartbeat = time.perf_counter()
        heartbeat_seconds = 60.0

        with futures.ThreadPoolExecutor(max_workers=deskew_workers) as executor:
            while next_write < x_size:
                submitted_before = next_submit
                while next_submit < len(x_blocks) and len(pending_blocks) < pending_block_limit:
                    x_start, x_stop = x_blocks[next_submit]
                    pending_blocks.add(executor.submit(compute_block, x_start, x_stop))
                    next_submit += 1
                if next_submit > submitted_before:
                    first_start = x_blocks[submitted_before][0]
                    last_stop = x_blocks[next_submit - 1][1]
                    print(
                        f"  Submitted deskew blocks {submitted_before + 1}-{next_submit}/"
                        f"{len(x_blocks)} pages {first_start + 1}-{last_stop}/"
                        f"{x_size}; pending={len(pending_blocks)}, completed={completed}",
                        flush=True,
                    )

                write_before = next_write
                next_write, write_seconds = _flush_ready_blocks(write_buffer, next_write, write_page)
                if next_write > write_before:
                    print(
                        f"  Wrote top-view pages {write_before + 1}-{next_write}/"
                        f"{x_size}: write={write_seconds:.2f}s",
                        flush=True,
                    )
                    if next_write % 50 == 0 or next_write == x_size:
                        log_progress(f"Wrote top-view page {next_write}/{x_size}")

                if next_write >= x_size:
                    break

                done, pending_blocks = futures.wait(
                    pending_blocks,
                    timeout=heartbeat_seconds,
                    return_when=futures.FIRST_COMPLETED,
                )
                if not done:
                    now = time.perf_counter()
                    if now - last_heartbeat >= heartbeat_seconds:
                        print(
                            f"  Chunked deskew heartbeat: submitted_blocks={next_submit}/"
                            f"{len(x_blocks)}, completed_blocks={completed}, "
                            f"written_pages={next_write}/{x_size}, pending={len(pending_blocks)}, "
                            f"buffered={len(write_buffer)}",
                            flush=True,
                        )
                        last_heartbeat = now
                    continue

                for future in done:
                    x_start, block = future.result()
                    write_buffer[int(x_start)] = block
                    completed += 1
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
            shape=(x_size, shear_y, scaled_z),
            chunks=(1, min(256, shear_y), min(256, scaled_z)),
            dtype=np.dtype("uint16"),
            layer_name=image_stem(output_path),
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
            zarr_output[page_index, :, :] = page
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

    log_progress(
        f"Finished GPU top-view deskew output: {output_path} "
        f"in {time.perf_counter() - start_time:.2f}s"
    )
    return output_shape


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
    z_chunk: int,
    deskew_workers: int,
    deskew_prefetch: int,
    deskew_x_block: int,
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
        output_name = f"{image_stem(path)}.ome.zarr" if is_ome_zarr_path(path) else f"{image_stem(path)}.tif"
        backend = _resolve_deskew_backend(deskew_backend)
        if backend == "gpu":
            output_shape = _write_top_shear_gpu(
                volume,
                top_shear_dir / output_name,
                dx=float(dx),
                dz=float(dz),
                angle=float(angle),
                flip=int(flip),
                deskew_prefetch=int(deskew_prefetch),
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
                deskew_workers=int(deskew_workers),
                deskew_prefetch=int(deskew_prefetch),
                deskew_x_block=int(deskew_x_block),
            )
        (top_shear_dir / "note.txt").write_text(
            "Chunked top-view deskew output. "
            f"output_yzx={output_shape}; z pixel = x(y) pixel.\n"
        )
        log_progress(
            f"Finished deskew input {path.name}: output={output_name}, "
            f"elapsed={time.perf_counter() - volume_start:.2f}s"
        )
    log_progress(f"Chunked deskew complete in {time.perf_counter() - run_start:.2f}s")


def main() -> None:
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
    parser.add_argument("--deskew_backend", default="gpu", choices=["gpu", "cpu", "cpu_blocked", "cuda"])
    parser.add_argument("--deskew_workers", type=int, default=32)
    parser.add_argument("--deskew_prefetch", type=int, default=64)
    parser.add_argument("--deskew_x_block", type=int, default=1)
    args = parser.parse_args()
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
        z_chunk=max(1, args.z_chunk),
        deskew_workers=max(1, args.deskew_workers),
        deskew_prefetch=max(1, args.deskew_prefetch),
        deskew_x_block=max(1, args.deskew_x_block),
    )


if __name__ == "__main__":
    main()

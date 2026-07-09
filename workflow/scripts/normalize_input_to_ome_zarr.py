#!/usr/bin/env python3
"""Normalize selected image inputs into workflow-native OME-Zarr volumes."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

from ome_zarr_io import image_stem, is_ome_zarr_path, log_progress, write_ome_zarr_array


TIFF_SUFFIXES = {".tif", ".tiff"}
CZI_SUFFIXES = {".czi"}
ND2_SUFFIXES = {".nd2"}
LIF_SUFFIXES = {".lif"}
HDF5_SUFFIXES = {".h5", ".hdf5"}
OZX_SUFFIXES = {".ozx"}
SUPPORTED_FILE_SUFFIXES = TIFF_SUFFIXES | CZI_SUFFIXES | ND2_SUFFIXES | LIF_SUFFIXES | HDF5_SUFFIXES | OZX_SUFFIXES


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'numpy' for input normalization") from exc
    return np


def coerce_volume(data, *, source: Path | str | None = None):
    np = _require_numpy()
    array = np.asarray(data)
    array = np.squeeze(array)
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    if array.ndim != 3:
        location = f" at {source}" if source else ""
        raise ValueError(f"Expected a 2-D or 3-D image volume{location}, got shape {array.shape}")
    return array


def load_tiff_volume(path: Path):
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError("Missing required dependency 'tifffile' for TIFF input normalization") from exc

    try:
        data = tifffile.memmap(str(path), mode="r")
    except Exception:
        data = tifffile.imread(str(path))
    return coerce_volume(data, source=path)


def load_czi_volume(path: Path):
    try:
        from aicsimageio import AICSImage
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency 'aicsimageio' for CZI input normalization") from exc

    image = AICSImage(str(path))
    return coerce_volume(image.get_image_data("ZYX", T=0, C=0), source=path)


def load_nd2_volume(path: Path):
    try:
        import nd2
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency 'nd2' for ND2 input normalization") from exc

    with nd2.ND2File(str(path)) as handle:
        return coerce_volume(handle.asarray(), source=path)


def load_lif_volume(path: Path):
    try:
        from readlif.reader import LifFile
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency 'readlif' for LIF input normalization") from exc

    lif = LifFile(str(path))
    image = next(iter(lif.get_iter_image()), None)
    if image is None:
        raise ValueError(f"No image series found in LIF file: {path}")
    z_size = lif_z_size(image)
    if z_size > 1:
        np = _require_numpy()
        return coerce_volume(
            np.stack([lif_get_frame(image, z=z) for z in range(z_size)], axis=0),
            source=path,
        )
    return coerce_volume(lif_get_frame(image, z=0), source=path)


def lif_get_frame(image, *, z: int):
    try:
        return image.get_frame(z=z, t=0, c=0)
    except TypeError:
        return image.get_frame(z=z, t=0)


def lif_z_size(image) -> int:
    for dims in (getattr(image, "dims_n", None), getattr(image, "dims", None)):
        if dims is None:
            continue
        if isinstance(dims, dict):
            value = dims.get("z") or dims.get("Z")
        else:
            value = getattr(dims, "z", None) or getattr(dims, "Z", None)
            if value is None and isinstance(dims, (tuple, list)) and len(dims) >= 3:
                value = dims[2]
        if value:
            return int(value)
    return 1


def _first_hdf5_dataset(group):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency 'h5py' for HDF5 input normalization") from exc

    for value in group.values():
        if isinstance(value, h5py.Dataset):
            return value
        if isinstance(value, h5py.Group):
            dataset = _first_hdf5_dataset(value)
            if dataset is not None:
                return dataset
    return None


def load_hdf5_volume(path: Path):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency 'h5py' for HDF5 input normalization") from exc

    with h5py.File(str(path), "r") as handle:
        dataset = _first_hdf5_dataset(handle)
        if dataset is None:
            raise ValueError(f"No dataset found in HDF5 file: {path}")
        return coerce_volume(dataset[()], source=path)


LOADERS_BY_SUFFIX = {
    **{suffix: load_tiff_volume for suffix in TIFF_SUFFIXES},
    **{suffix: load_czi_volume for suffix in CZI_SUFFIXES},
    **{suffix: load_nd2_volume for suffix in ND2_SUFFIXES},
    **{suffix: load_lif_volume for suffix in LIF_SUFFIXES},
    **{suffix: load_hdf5_volume for suffix in HDF5_SUFFIXES},
}


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_name = member.filename
            if not member_name:
                continue
            target = destination / member_name
            try:
                target.resolve().relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe path in OZX archive {archive_path}: {member_name}") from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def unpack_ozx_archive(path: Path, output_dir: Path) -> Path:
    log_progress(f"Unpacking OZX OME-Zarr input: {path.name}")
    with tempfile.TemporaryDirectory(prefix=f"{path.stem}_ozx_", dir=output_dir.parent) as tmpdir:
        tmp_root = Path(tmpdir)
        _safe_extract_zip(path, tmp_root)
        zarr_candidates = sorted(
            candidate
            for candidate in tmp_root.rglob("*")
            if candidate.is_dir() and is_ome_zarr_path(candidate)
        )
        if len(zarr_candidates) != 1:
            raise ValueError(
                f"OZX archive {path} must contain exactly one .ome.zarr directory; "
                f"found {len(zarr_candidates)}"
            )
        output_path = output_dir / zarr_candidates[0].name
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.move(str(zarr_candidates[0]), output_path)
    log_progress(f"Finished {output_path.name}")
    return output_path


def supported_input_paths(input_dir: Path) -> list[Path]:
    paths = []
    for path in sorted(input_dir.iterdir(), key=lambda item: item.name):
        if path.is_dir() and is_ome_zarr_path(path):
            paths.append(path)
        elif path.is_file() and path.suffix.lower() in SUPPORTED_FILE_SUFFIXES:
            paths.append(path)
    return paths


def normalize_one(path: Path, output_dir: Path) -> Path:
    output_path = output_dir / f"{image_stem(path)}.ome.zarr"
    if path.is_dir() and is_ome_zarr_path(path):
        log_progress(f"Copying existing OME-Zarr input: {path.name} -> {output_path.name}")
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(path, output_path)
        log_progress(f"Finished {output_path.name}")
        return output_path
    if path.is_file() and path.suffix.lower() in OZX_SUFFIXES:
        return unpack_ozx_archive(path, output_dir)

    loader = LOADERS_BY_SUFFIX.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported input format for OME-Zarr normalization: {path}")
    log_progress(f"Normalizing {path.name} -> {output_path.name}")
    log_progress(f"Loading {path.name} with {loader.__name__ if hasattr(loader, '__name__') else loader}")
    array = loader(path)
    log_progress(f"Loaded {path.name}: shape={array.shape}, dtype={array.dtype}")
    write_ome_zarr_array(
        output_path,
        array,
        layer_name=image_stem(path),
    )
    log_progress(f"Finished {output_path.name}")
    return output_path


def normalize_directory(input_dir: Path, output_dir: Path) -> list[Path]:
    log_progress(f"Input normalization: scanning {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    original_map = input_dir / "original_filenames.tsv"
    if original_map.exists():
        (output_dir / "original_filenames.tsv").write_text(original_map.read_text())
        log_progress(f"Copied original filename map: {original_map.name}")

    input_paths = supported_input_paths(input_dir)
    log_progress(f"Found {len(input_paths)} supported image input(s)")
    outputs = []
    for index, path in enumerate(input_paths, start=1):
        log_progress(f"Starting input {index}/{len(input_paths)}: {path.name}")
        outputs.append(normalize_one(path, output_dir))
    if not outputs:
        suffixes = ", ".join(sorted([*SUPPORTED_FILE_SUFFIXES, ".ome.zarr"]))
        raise FileNotFoundError(f"No supported image inputs found in {input_dir}; expected one of: {suffixes}")
    log_progress(f"Input normalization complete: wrote {len(outputs)} OME-Zarr volume(s)")
    return outputs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Normalize selected image inputs to OME-Zarr.")
    parser.add_argument("--input", required=True, help="Directory containing selected image inputs")
    parser.add_argument("--output", required=True, help="Output directory for normalized OME-Zarr inputs")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        outputs = normalize_directory(Path(args.input), Path(args.output))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(outputs)} normalized OME-Zarr input volume(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

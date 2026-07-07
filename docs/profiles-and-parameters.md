# Profiles and Parameters

## Profiles

| Profile | Purpose |
| --- | --- |
| `light_sheet` | Uses the default ctASLM angle and flip settings. |
| `conda_runtime` | Builds the workflow conda runtime inside the run directory. |

## Required Inputs

| Parameter | Description |
| --- | --- |
| `input` | Selected raw image files or OME-Zarr volumes. |
| `dx` | X/Y voxel size in microns. |
| `dz` | Z voxel size in microns. |

## Geometry

`angle` defaults to `40` degrees and `flip` defaults to `1`. Adjust these for
datasets acquired with different scanner geometry.

## Runtime Backend

`deskew_backend` defaults to `cpu_blocked`. Use `gpu` or `cuda` to run the
Numba CUDA backend and request one Slurm GPU. CPU runs always compute and write
one output X page at a time. Use `z_chunk` to tune the CPU sampling tile depth.
GPU runs use `deskew_prefetch` as the output-page batch size.

`pyramid_max_downsample` controls the largest XY OME-Zarr pyramid factor written
after deskewing. The default `16` preserves the full `1x, 2x, 4x, 8x, 16x`
pyramid. Use `1` to write only level `0` and skip downsampled pyramid
generation.

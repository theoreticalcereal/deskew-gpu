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

`deskew_geometry` defaults to `top_view`, the MATLAB-compatible output geometry.
Use `clearex_affine` to write a ClearEx-style physical affine output in `z, y,
x` order. In that mode, `angle` and `flip` are converted to the ClearEx affine
shear/rotation model and outputs are written as OME-Zarr.

`deskew_output_dtype` defaults to `uint16`. Use `float32` with
`clearex_affine` geometry when comparing against ClearEx, because ClearEx keeps
linear interpolation output as `float32` unless an integer `output_dtype` is
requested.

## Runtime Backend

`deskew_backend` defaults to `cpu_blocked`. Use `gpu` or `cuda` to run the
Numba CUDA backend and request one Slurm GPU. CPU runs always compute and write
one output X page at a time for `top_view` geometry. Use `z_chunk` to tune the
CPU sampling tile depth. GPU runs use `deskew_prefetch` as the output-page batch
size for `top_view` geometry and as the output-Z batch size for
`clearex_affine` geometry.

`pyramid_max_downsample` controls the largest XY OME-Zarr pyramid factor written
after deskewing. The default `16` preserves the full `1x, 2x, 4x, 8x, 16x`
pyramid. Use `1` to write only level `0` and skip downsampled pyramid
generation.

## Output Format

`output_formats` defaults to `ome_zarr`, which publishes only the native
`Top_shear/` OME-Zarr output. Choose `tiff` to also export all deskewed
`Top_shear` volumes as one merged BigTIFF stack in `deskewed_tiff/`.

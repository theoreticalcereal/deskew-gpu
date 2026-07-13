# Deskew Process

`DESKEW` corrects oblique light-sheet acquisition geometry. The process reads
OME-Zarr or TIFF volumes through `chunked_deskew.py`, computes top-view output
pages, and writes deskewed OME-Zarr volumes under `Top_shear/`.

CPU chunked mode is the default backend (`deskew_backend = cpu_blocked`). Set
`deskew_backend = gpu` or `deskew_backend = cuda` to use the Numba CUDA
implementation. GPU runs write the same `Top_shear/` output layout as the CPU
path. The top-view CUDA path batches output X pages; the ClearEx-affine CUDA
path batches output Z pages and uploads only the source X slab needed for each
output tile.

The ClearEx-affine implementation is self-contained in this workflow. It does
not import ClearEx at runtime; the affine geometry and sampler behavior are
implemented locally for reproducible Nextflow runs. ClearEx-affine deskew
applies the Y/Z shear without an extra X rotation by default. Set
`deskew_affine_rotate` to apply the older reference-style `-flip * angle`
rotation after shearing.

The implementation computes one or more output X pages internally, but
top-shear OME-Zarr output is written and labelled in `z, y, x` order. TIFF
output keeps the existing page-stack order for compatibility with the original
MATLAB workflow.

CPU tuning parameters:

- CPU deskew always computes and writes one output X page at a time. This keeps
  CPU reference runs serial and deterministic.
- `z_chunk`: number of output Z samples processed per CPU sampling tile.

GPU tuning parameter:

- `deskew_prefetch`: number of output X pages processed in each top-view GPU
  batch, or output Z slices processed in each ClearEx-affine GPU batch.

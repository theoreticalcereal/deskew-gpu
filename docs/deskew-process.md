# Deskew Process

`DESKEW` corrects oblique light-sheet acquisition geometry. The process reads
OME-Zarr or TIFF volumes through `chunked_deskew.py`, computes top-view output
pages, and writes deskewed OME-Zarr volumes under `Top_shear/`.

CPU chunked mode is the default backend (`deskew_backend = cpu_blocked`). Set
`deskew_backend = gpu` or `deskew_backend = cuda` to use the Numba CUDA
implementation. GPU runs materialize each input volume once for transfer to the
device, then write the same `Top_shear/` output layout as the CPU path.

CPU tuning parameters:

- CPU deskew always computes and writes one output X page at a time. This keeps
  CPU reference runs serial and deterministic.
- `z_chunk`: number of output Z samples processed per CPU sampling tile.

GPU tuning parameter:

- `deskew_prefetch`: number of output X pages processed in each GPU batch.

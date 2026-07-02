# Deskew Process

`DESKEW` corrects oblique light-sheet acquisition geometry. The process reads
OME-Zarr or TIFF volumes through `chunked_deskew.py`, computes top-view output
pages, and writes deskewed OME-Zarr volumes under `Top_shear/`.

GPU mode is the default. Set `deskew_backend = cpu_blocked` only when CUDA is
not available or when comparing against the CPU reference path.

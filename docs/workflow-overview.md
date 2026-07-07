# Workflow Overview

The deskew package contains only the deskew half of the original combined
workflow.

```text
BUILD_DESKEW_CONTAINER
STAGE_DESKEW_INPUT
DESKEW
```

`STAGE_DESKEW_INPUT` normalizes selected image files to OME-Zarr and preserves
original filenames in `original_filenames.tsv`. `DESKEW` reads those normalized
volumes and publishes corrected volumes under `Top_shear/`.

Deskew OME-Zarr top-shear outputs are written in `z, y, x` order as multiscale
pyramids. Level `0` is full resolution; levels `1` through `4` downsample the
axes named `y` and `x` by `2x`, `4x`, `8x`, and `16x` while preserving the
axis named `z`.

The conda runtime built by `BUILD_DESKEW_CONTAINER` includes the deconvolution
dependencies as well. Integrated pipelines can pass that runtime directory into
`deconvolution-gpu` with `decon_runtime_dir`.

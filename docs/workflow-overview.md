# Workflow Overview

The deskew package contains only the deskew half of the original combined
workflow.

```text
STAGE_DESKEW_INPUT
DESKEW
EXPORT_OUTPUT_FORMAT
```

All processes run in the BioHPC GitLab container
`git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0`.
GPU deskew loads `cuda/11.8.0` and passes Singularity `--nv` for GPU backend
runs.

`STAGE_DESKEW_INPUT` normalizes selected image files to OME-Zarr and preserves
original filenames in `original_filenames.tsv`. `DESKEW` reads those normalized
volumes and writes task-local corrected volumes under `Top_shear/`.
`EXPORT_OUTPUT_FORMAT` publishes zipped OME-Zarr archives under
`deskewed_ozx/` and, when requested, one merged TIFF stack under
`deskewed_tiff/`.

Deskew OME-Zarr data inside each `.ozx` archive is written in `z, y, x` order
as multiscale pyramids. Level `0` is full resolution; levels `1` through `4`
downsample the axes named `y` and `x` by `2x`, `4x`, `8x`, and `16x` while
preserving the axis named `z`.

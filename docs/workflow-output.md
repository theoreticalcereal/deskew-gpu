# Workflow Output

By default, and when `output_formats = ozx`, the workflow writes one zipped
OME-Zarr archive per deskewed volume:

```text
workflow/output/
`-- deskewed_ozx/
    `-- <sample>.ozx
```

When `output_formats = tiff`, the workflow keeps the OZX archives and also
writes one merged TIFF stack:

```text
workflow/output/
|-- deskewed_ozx/
|   `-- <sample>.ozx
`-- deskewed_tiff/
    `-- deskewed_merged.tif
```

The folder-form `Top_shear/` OME-Zarr output is task-local only and is not
published.

Level `0` is full resolution and top-shear output is stored and labelled in
`z, y, x` axis order. In these outputs, `z` is the scaled top-view depth, `y`
is the deskewed/sheared lateral axis, and `x` is the output page axis computed
from the original X dimension.

By default, levels `1` through `4` are XY downsampled from level `0` by
row/column stride slicing at `2x, 4x, 8x, 16x`; Z is not downsampled. Set
`pyramid_max_downsample` to `1`, `2`, `4`, or `8` to stop generation before the
default `16x` level.

The optional `deskewed_merged.tif` is a single BigTIFF stack made by reading
the deskewed task-local `Top_shear` volumes in sorted filename order and
concatenating them along Z.

Successful runs clean the Nextflow `work/` directory automatically after final
outputs are published. This keeps the published OZX and optional TIFF
outputs, but removes intermediate normalized Zarrs and task-local copies. As a
result, completed runs cannot be resumed from the cleaned task cache.

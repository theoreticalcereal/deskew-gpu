# Workflow Output

```text
workflow/output/
`-- Top_shear/
    `-- <sample>.ome.zarr/
        |-- 0/
        |-- 1/
        |-- 2/
        |-- 3/
        `-- 4/
```

`Top_shear/` is the handoff directory for the separate `deconvolution-gpu`
workflow.

Level `0` is full resolution. Levels `1` through `4` are XY downsampled from
level `0` by row/column stride slicing at `2x, 4x, 8x, 16x`; Z is not
downsampled.

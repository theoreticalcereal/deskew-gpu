# 3D GPU Deskew

Astrocyte/Nextflow package for ctASLM/light-sheet deskewing. This workflow
normalizes selected image files to OME-Zarr, runs the chunked deskew process,
and publishes `Top_shear/` outputs for downstream use.

Run the separate `deconvolution-gpu` workflow after this package when blind PSF
estimation and GPU deconvolution are needed.

## Pipeline

1. `BUILD_DESKEW_CONTAINER` builds the per-run deskew conda runtime.
2. `STAGE_DESKEW_INPUT` links selected files, preserves original filenames, and
   normalizes supported images to `input_zarr/*.ome.zarr`.
3. `DESKEW` writes deskewed OME-Zarr volumes under `Top_shear/`.

The deskew runtime includes the deconvolution Python/CUDA dependencies so an
integrated pipeline can pass the built `deskew_runtime` directory to
`deconvolution-gpu --decon_runtime_dir` and avoid building a second conda
environment.

## Manual Run

```bash
cd workflow
nextflow run main.nf \
  -c configs/biohpc.config \
  -profile light_sheet \
  --input '/path/to/raw/*.tif' \
  --output_dir ./output \
  --dx 0.108 \
  --dz 0.3 \
  --angle 40 \
  --flip 1
```

## Output

```text
workflow/output/
`-- Top_shear/
    `-- <sample>.ome.zarr/
```

## VizApp

`vizapp/` is intentionally a placeholder containing only `.keep`. Visualization
is handled by a separate workflow package.

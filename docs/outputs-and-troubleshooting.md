# Outputs and Troubleshooting

## Missing Sampling Parameters

`dx` and `dz` are required. The workflow fails before running `DESKEW` when
either value is unset.

## CUDA Availability

`deskew_backend = gpu` or `deskew_backend = cuda` requires a visible CUDA device
and the GPU queue. The default `deskew_backend = cpu_blocked` does not request a
GPU and should be used when CUDA is unavailable.

## Runtime Environment

The workflow runs in
`git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0`.
BioHPC staging loads `singularity/3.9.9`, and the BioHPC workflow passes
Singularity `--nv` for GPU backend runs so GPU jobs can see the host NVIDIA
driver. CUDA user-space libraries are supplied by the container image.
Visualization dependencies are handled by a separate package.

## CPU/GPU Output Comparison

Use `workflow/scripts/compare_deskew_outputs.py` to compare CPU and GPU deskew
outputs after both runs finish. Pass explicit output paths:

```bash
python workflow/scripts/compare_deskew_outputs.py \
  --cpu cpu_output/Top_shear/sample.ome.zarr \
  --gpu gpu_output/Top_shear/sample.ome.zarr \
  --output_json deskew-comparison.json
```

The script can also infer output paths and lateral pixel size from matching
workflow parameter YAML files with `--cpu_params` and `--gpu_params`.

## ClearEx Reference Comparison

For ClearEx validation, run deskew with `deskew_geometry = clearex_affine` and
`deskew_output_dtype = float32`. This matches ClearEx's physical affine
shear/rotation output shape and preserves linear interpolation values before any
integer rounding.

ClearEx reference arrays are commonly stored as 6D `(t, p, c, z, y, x)` Zarr
components. Compare one `(t, p, c)` volume with:

```bash
python workflow/scripts/compare_deskew_outputs.py \
  --reference clearex_input.zarr \
  --reference_component clearex/runtime_cache/results/shear_transform/latest/data \
  --reference_tpc 0,0,0 \
  --candidate output/Top_shear/sample.ome.zarr \
  --level 0 \
  --lateral_pixel_size 0.168 \
  --output_json clearex-vs-deskew.json
```

Use `--trim_zyx Z,Y,X` to compare interior regions separately from boundary
slices. If interior metrics are high but full-volume metrics are lower, inspect
low-Z and high-Z edge bands. The ClearEx-affine sampler uses a half-voxel
image-domain boundary convention, matching ANTs/ClearEx behavior at the first
and last transformed slices.

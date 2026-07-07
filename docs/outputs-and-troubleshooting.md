# Outputs and Troubleshooting

## Missing Sampling Parameters

`dx` and `dz` are required. The workflow fails before running `DESKEW` when
either value is unset.

## CUDA Availability

`deskew_backend = gpu` or `deskew_backend = cuda` requires a visible CUDA device
and the GPU queue. The default `deskew_backend = cpu_blocked` does not request a
GPU and should be used when CUDA is unavailable.

## Runtime Environment

The workflow builds its conda runtime from `workflow/envs/deskew-conda.txt` and
`workflow/envs/deskew-pip-requirements.txt`. Visualization dependencies are not
installed by this package.

The runtime also includes the deconvolution dependencies so it can be reused by
`deconvolution-gpu` when `decon_runtime_dir` points at the built deskew runtime.

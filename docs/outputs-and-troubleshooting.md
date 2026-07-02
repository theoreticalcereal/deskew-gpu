# Outputs and Troubleshooting

## Missing Sampling Parameters

`dx` and `dz` are required. The workflow fails before running `DESKEW` when
either value is unset.

## CUDA Availability

`deskew_backend = gpu` requires a visible CUDA device and the GPU queue. If CUDA
is not available, set `deskew_backend = cpu_blocked`.

## Runtime Environment

The workflow builds its conda runtime from `workflow/envs/deskew-conda.txt` and
`workflow/envs/deskew-pip-requirements.txt`. Visualization dependencies are not
installed by this package.

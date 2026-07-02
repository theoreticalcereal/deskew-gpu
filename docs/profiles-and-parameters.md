# Profiles and Parameters

## Profiles

| Profile | Purpose |
| --- | --- |
| `light_sheet` | Uses the default ctASLM angle and flip settings. |
| `conda_runtime` | Builds the workflow conda runtime inside the run directory. |

## Required Inputs

| Parameter | Description |
| --- | --- |
| `input` | Selected raw image files or OME-Zarr volumes. |
| `dx` | X/Y voxel size in microns. |
| `dz` | Z voxel size in microns. |

## Geometry

`angle` defaults to `40` degrees and `flip` defaults to `1`. Adjust these for
datasets acquired with different scanner geometry.

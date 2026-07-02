# 3D GPU Deskew

This package runs the deskew half of the original combined workflow. It does not
run deconvolution and does not launch visualization.

## Workflow

```text
selected images -> STAGE_DESKEW_INPUT -> DESKEW -> Top_shear/
```

Use `deconvolution-gpu` after this workflow when blind PSF estimation and GPU
deconvolution are needed.

## Documentation

| Page | Purpose |
| --- | --- |
| [Workflow Overview](workflow-overview.md) | Process order and data flow. |
| [Deskew Process](deskew-process.md) | Deskew geometry and implementation notes. |
| [Profiles and Parameters](profiles-and-parameters.md) | BioHPC profiles and Astrocyte parameters. |
| [Workflow Output](workflow-output.md) | Published output layout. |
| [Outputs and Troubleshooting](outputs-and-troubleshooting.md) | Common runtime and data issues. |

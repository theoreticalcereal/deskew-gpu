# Container Staging Design

## Goal

Deskew GPU should always run from the prebuilt BioHPC GitLab container
`git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0`
instead of building or reusing a conda runtime at workflow execution time.

## Design

Use the same staging model as `../siftest/scrna-qc`: the Astrocyte package
declares the required Singularity container with a `docker://` URI, and each
Python-running Nextflow process uses the registry image through the `container`
directive without the URI protocol. The workflow no longer exposes runtime
build, runtime reuse, or runtime export parameters.

The affected processes are:

- `STAGE_DESKEW_INPUT`
- `DESKEW`
- `EXPORT_OUTPUT_FORMAT`

The old `BUILD_DESKEW_CONTAINER` process is removed from workflow orchestration.
The old conda and pip lock files are removed so the package has no manual conda
build surface.

## Error Handling

Dependency failures now surface as container execution failures rather than
conda solve or pip install failures. Because the image is mandatory, there is no
fallback to a host conda environment.

## Documentation

README and workflow docs should describe the fixed container runtime and remove
instructions for exporting or reusing `deskew_runtime`.

## Testing

Static tests should verify the workflow no longer references the runtime builder
or runtime parameters. If local Nextflow/Singularity is available, run a syntax
or dry-run check against the updated workflow.

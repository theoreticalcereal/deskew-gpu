def DESKEW_CONTAINER_IMAGE = 'git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0'
def CONTAINER_ENV_PREFIX = '/opt/conda/envs/app'

process STAGE_DESKEW_INPUT {
    tag "deskew_input"
    module 'singularity/3.9.9'
    container DESKEW_CONTAINER_IMAGE

    input:
    path input_files

    output:
    path "input_zarr", emit: deskew_input_dir

    script:
    def shell_quote = { value -> "'${value.toString().replace("'", "'\\''")}'" }
    def link_commands = input_files.collect { input_file ->
        "ln -s \"\$PWD/${input_file}\" ${shell_quote("deskew_input/${input_file.name}")}"
    }.join('\n')
    def metadata_commands = input_files.collect { input_file ->
        "printf '%s\\t%s\\n' ${shell_quote(input_file.name)} ${shell_quote(input_file.name)} >> deskew_input/original_filenames.tsv"
    }.join('\n')

    """
    mkdir -p deskew_input
    : > deskew_input/original_filenames.tsv
    ${link_commands}
    ${metadata_commands}

    export CONDA_PREFIX="${CONTAINER_ENV_PREFIX}"
    export PATH="${CONTAINER_ENV_PREFIX}/bin:\${PATH}"

    python3 ${projectDir}/scripts/normalize_input_to_ome_zarr.py \\
        --input deskew_input \\
        --output input_zarr
    """
}

process DESKEW {
    tag "${cell_name ?: 'deskew'}"
    module 'singularity/3.9.9:cuda/11.8.0'
    container DESKEW_CONTAINER_IMAGE
    containerOptions = { ['gpu', 'cuda'].contains((params.deskew_backend ?: '').toString().trim().toLowerCase().replace('-', '_')) ? '--nv' : '' }

    publishDir "${params.output_dir}", mode: 'copy', enabled: false

    input:
    val image_path
    val cell_name
    val cell_index
    val dx
    val dz
    val angle
    val flip
    val output_dir

    output:
    path "Top_shear", emit: deskewed_path
    path "shear", optional: true, emit: shear_output

    script:
    """
    export CONDA_PREFIX="${CONTAINER_ENV_PREFIX}"
    export PATH="${CONTAINER_ENV_PREFIX}/bin:\${PATH}"

    python3 ${projectDir}/scripts/chunked_deskew.py \\
        --image_path "${image_path}" \\
        --cell_name "${cell_name}" \\
        --cell_index "${cell_index}" \\
        --dx ${dx} \\
        --dz ${dz} \\
        --angle ${angle} \\
        --flip ${flip} \\
        --output_dir . \\
        --deskew_backend ${params.deskew_backend} \\
        --deskew_geometry ${params.deskew_geometry} \\
        --z_chunk ${params.z_chunk} \\
        --deskew_prefetch ${params.deskew_prefetch} \\
        --pyramid_max_downsample ${params.pyramid_max_downsample} \\
        --deskew_output_dtype ${params.deskew_output_dtype}
    """
}

process EXPORT_OUTPUT_FORMAT {
    tag "${output_format}"
    module 'singularity/3.9.9'
    container DESKEW_CONTAINER_IMAGE

    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path deskew_outputs
    val output_format

    output:
    path "deskewed_tiff", optional: true, emit: exported_output
    path "deskewed_ozx", optional: true, emit: exported_ozx

    script:
    """
    export CONDA_PREFIX="${CONTAINER_ENV_PREFIX}"
    export PATH="${CONTAINER_ENV_PREFIX}/bin:\${PATH}"

    output_dir=deskewed_ozx
    python3 ${projectDir}/scripts/export_ome_zarr_to_tiff.py \\
        --input "${deskew_outputs}" \\
        --output "\${output_dir}" \\
        --output-format "ozx"

    if [ "${output_format}" = "tiff" ]; then
        python3 ${projectDir}/scripts/export_ome_zarr_to_tiff.py \\
            --input "${deskew_outputs}" \\
            --output "deskewed_tiff" \\
            --output-format "${output_format}"
    elif [ "${output_format}" = "ozx" ]; then
        true
    else
        echo "ERROR: unsupported output format: ${output_format}" >&2
        exit 1
    fi
    """
}

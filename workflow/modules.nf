process BUILD_DESKEW_CONTAINER {
    tag "deskew_env"

    cpus 2
    memory '8 GB'
    queue 'super'
    publishDir "${params.output_dir}", mode: 'copy', pattern: 'deskew_runtime', enabled: params.export_deskew_runtime.toString() == 'true'

    output:
    path "deskew_runtime", emit: image

    script:
    """
    set -euo pipefail
    mkdir -p deskew_runtime

    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda is required to build the deskew environment." >&2
        exit 127
    fi

    export CONDA_PKGS_DIRS="\$PWD/.conda_pkgs"
    conda create -y -p .conda_libmamba -c conda-forge "conda>=23.7" conda-libmamba-solver
    .conda_libmamba/bin/python -m conda create -y --solver=libmamba \\
        -p deskew_runtime/deskew_env \\
        -c conda-forge \\
        -c bioconda \\
        --file "${projectDir}/envs/deskew-conda.txt"
    deskew_runtime/deskew_env/bin/python -m pip install --constraint "${projectDir}/envs/deskew-pip-constraints.txt" -r "${projectDir}/envs/deskew-pip-requirements.txt"

    if [ ! -x deskew_runtime/deskew_env/bin/python3 ] && [ ! -x deskew_runtime/deskew_env/bin/python ]; then
        echo "ERROR: failed to build a usable deskew conda environment." >&2
        exit 1
    fi
    """
}

process STAGE_DESKEW_INPUT {
    tag "deskew_input"

    input:
    path input_files
    path deskew_runtime

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

    if [ -x "${deskew_runtime}/deskew_env/bin/python3" ] || [ -x "${deskew_runtime}/deskew_env/bin/python" ]; then
        export CONDA_PREFIX="${deskew_runtime}/deskew_env"
    elif [ -x "${deskew_runtime}/bin/python3" ] || [ -x "${deskew_runtime}/bin/python" ]; then
        export CONDA_PREFIX="${deskew_runtime}"
    else
        echo "ERROR: no supported deskew runtime found at ${deskew_runtime}" >&2
        exit 1
    fi
    export CONDA_DEFAULT_ENV=deskew_env
    export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}

    python3 ${projectDir}/scripts/normalize_input_to_ome_zarr.py \\
        --input deskew_input \\
        --output input_zarr
    """
}

process DESKEW {
    tag "${cell_name ?: 'deskew'}"

    publishDir "${params.output_dir}", mode: 'copy', enabled: params.output_formats != 'ozx'

    input:
    val image_path
    val cell_name
    val cell_index
    val dx
    val dz
    val angle
    val flip
    val output_dir
    path deskew_runtime

    output:
    path "Top_shear", emit: deskewed_path
    path "shear", optional: true, emit: shear_output

    script:
    """
    if [ -x "${deskew_runtime}/deskew_env/bin/python3" ] || [ -x "${deskew_runtime}/deskew_env/bin/python" ]; then
        export CONDA_PREFIX="${deskew_runtime}/deskew_env"
    elif [ -x "${deskew_runtime}/bin/python3" ] || [ -x "${deskew_runtime}/bin/python" ]; then
        export CONDA_PREFIX="${deskew_runtime}"
    else
        echo "ERROR: no supported deskew runtime found at ${deskew_runtime}" >&2
        exit 1
    fi
    export CONDA_DEFAULT_ENV=deskew_env
    export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}

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

    publishDir "${params.output_dir}", mode: 'copy'

    input:
    path deskew_outputs
    val output_format
    path deskew_runtime

    output:
    path "deskewed_tiff", optional: true, emit: exported_output
    path "deskewed_ozx", optional: true, emit: exported_ozx

    script:
    """
    if [ -x "${deskew_runtime}/deskew_env/bin/python3" ] || [ -x "${deskew_runtime}/deskew_env/bin/python" ]; then
        export CONDA_PREFIX="${deskew_runtime}/deskew_env"
    elif [ -x "${deskew_runtime}/bin/python3" ] || [ -x "${deskew_runtime}/bin/python" ]; then
        export CONDA_PREFIX="${deskew_runtime}"
    else
        echo "ERROR: no supported deskew runtime found at ${deskew_runtime}" >&2
        exit 1
    fi
    export CONDA_DEFAULT_ENV=deskew_env
    export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
    export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}

    if [ "${output_format}" = "tiff" ]; then
        python3 ${projectDir}/scripts/export_ome_zarr_to_tiff.py \\
            --input "${deskew_outputs}" \\
            --output "deskewed_tiff" \\
            --output-format "${output_format}"
    elif [ "${output_format}" = "ozx" ]; then
        output_dir=deskewed_ozx
        python3 ${projectDir}/scripts/export_ome_zarr_to_tiff.py \\
            --input "${deskew_outputs}" \\
            --output "\${output_dir}" \\
            --output-format "${output_format}"
    else
        echo "ERROR: unsupported output format: ${output_format}" >&2
        exit 1
    fi
    """
}

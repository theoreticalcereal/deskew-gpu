#!/usr/bin/env nextflow
nextflow.enable.dsl=2

include { BUILD_DESKEW_CONTAINER } from './modules'
include { STAGE_DESKEW_INPUT } from './modules'
include { DESKEW } from './modules'

def inputTextFromCommandLine(commandLine) {
    if (!commandLine) {
        return null
    }

    def matcher = (commandLine =~ /(?:^|\s)--input\s+(.+?)(?=\s+--[A-Za-z0-9_][A-Za-z0-9_-]*\b|$)/)
    return matcher.find() ? matcher.group(1).trim() : null
}

def normalizeInputPatterns(input, commandLine = null) {
    if (!input) {
        return []
    }

    def input_text = (input instanceof List)
        ? input.collect { it.toString() }.join(',')
        : input.toString()

    input_text = input_text.trim()
    def command_input_text = inputTextFromCommandLine(commandLine)
    if (command_input_text && (
            input_text.count('[') != input_text.count(']') ||
            command_input_text.startsWith('[') ||
            command_input_text.contains(input_text))) {
        input_text = command_input_text
    }

    if (input_text.startsWith('[') && input_text.endsWith(']')) {
        input_text = input_text.substring(1, input_text.length() - 1).trim()
    }

    return input_text
        .split(/\s*,\s*/)
        .collect { input_pattern ->
            input_pattern
                .trim()
                .replaceAll(/^[\['"\s]+/, '')
                .replaceAll(/[\]'"\s]+$/, '')
        }
        .findAll { it }
}

def isSupplied(value) {
    if (value == null) {
        return false
    }
    def text = value.toString().trim()
    return text && text != '-1' && text != '-1.0'
}

def optionalValue(value) {
    return isSupplied(value) ? value : ''
}

def requireSupplied(name, value, context) {
    if (!isSupplied(value)) {
        throw new IllegalArgumentException("${name} must be provided for ${context}; -1 means unset.")
    }
    return value
}

workflow {
    if (isSupplied(params.deskew_runtime_dir)) {
        log.info "Using prebuilt deskew runtime: ${params.deskew_runtime_dir}"
        deskew_container_ch = Channel.value(file(params.deskew_runtime_dir, checkIfExists: true))
    } else {
        BUILD_DESKEW_CONTAINER()
        deskew_container_ch = BUILD_DESKEW_CONTAINER.out.image
    }

    input_patterns = normalizeInputPatterns(params.input, workflow.commandLine)
    if (input_patterns) {
        log.info "Selected ${input_patterns.size()} input image(s): ${input_patterns.join(', ')}"
        input_files_ch = Channel
            .fromList(input_patterns)
            .map { input_pattern -> file(input_pattern, checkIfExists: true) }
            .collect()
        STAGE_DESKEW_INPUT(input_files_ch, deskew_container_ch)
        deskew_input_ch = STAGE_DESKEW_INPUT.out.deskew_input_dir
        deskew_cell_name = ''
    } else {
        deskew_input_ch = Channel.value(params.image_path)
        deskew_cell_name = params.cell_name
    }

    DESKEW(
        deskew_input_ch,
        deskew_cell_name,
        optionalValue(params.cell_index),
        requireSupplied('dx', params.dx, 'deskew runs'),
        requireSupplied('dz', params.dz, 'deskew runs'),
        params.angle,
        params.flip,
        params.output_dir,
        deskew_container_ch
    )
}

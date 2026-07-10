import contextlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
import tifffile
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "workflow" / "scripts"
DESKEW_CONTAINER_IMAGE = "git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0"
ASTROCYTE_CONTAINER_IMAGE = f"docker://{DESKEW_CONTAINER_IMAGE}"
CUDA_MODULE = "cuda/11.8.0"
CONTAINER_ENV_PREFIX = "/opt/conda/envs/app"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from chunked_deskew import (
    _clearex_affine_geometry,
    _linear_sample_support_normalized,
    _materialize_volume,
    _source_x_bounds_for_output_tile,
    _write_clearex_affine,
    _write_top_shear,
    run_chunked_deskew,
)
from ome_zarr_io import (
    create_ome_zarr_array,
    multiscales_metadata,
    open_ome_zarr_array,
    write_downsampled_pyramid,
)


class DeskewWiringTest(unittest.TestCase):
    def test_main_wires_deskew_without_deconvolution(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")

        self.assertIn("include { STAGE_DESKEW_INPUT } from './modules'", main_text)
        self.assertIn("include { DESKEW } from './modules'", main_text)
        self.assertIn("include { EXPORT_OUTPUT_FORMAT } from './modules'", main_text)
        self.assertIn('def package_relative = "${projectDir}/../${text}"', main_text)
        self.assertNotIn("BUILD_DESKEW_CONTAINER", main_text)
        self.assertNotIn("deskew_container_ch", main_text)
        self.assertNotIn("DECON", main_text)

    def test_modules_keep_only_deskew_processes(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("process STAGE_DESKEW_INPUT", modules_text)
        self.assertIn("process DESKEW", modules_text)
        self.assertIn("process EXPORT_OUTPUT_FORMAT", modules_text)
        self.assertNotIn("process BUILD_DESKEW_CONTAINER", modules_text)
        self.assertNotIn("process DECON", modules_text)

    def test_workflow_uses_fixed_container_runtime(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        package_text = (ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8")
        package = yaml.safe_load(package_text)

        self.assertIn(DESKEW_CONTAINER_IMAGE, modules_text)
        self.assertIn(ASTROCYTE_CONTAINER_IMAGE, package["workflow_containers"])
        self.assertNotIn("neuroglancer-stage", modules_text)
        self.assertNotIn("neuroglancer-stage", package_text)
        self.assertEqual(modules_text.count("container DESKEW_CONTAINER_IMAGE"), 3)
        self.assertIn(CUDA_MODULE, package_text)
        self.assertIn(f"module 'singularity/3.9.9:{CUDA_MODULE}'", modules_text)
        self.assertIn("containerOptions = { ['gpu', 'cuda'].contains", modules_text)
        self.assertIn("? '--nv' : ''", modules_text)
        self.assertNotIn("deskew_runtime", modules_text)
        self.assertIn(f"CONTAINER_ENV_PREFIX = '{CONTAINER_ENV_PREFIX}'", modules_text)
        self.assertIn('export PATH="${CONTAINER_ENV_PREFIX}/bin:\\${PATH}"', modules_text)
        self.assertTrue((ROOT / "workflow/images/deskew-gpu/Dockerfile").exists())
        self.assertTrue((ROOT / "workflow/images/deskew-gpu/environment.yml").exists())
        self.assertTrue((ROOT / "workflow/images/deskew-gpu/build-deskew-image.sh").exists())

    def test_deskew_publishes_terminal_output_without_copying_tree(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn('publishDir "${params.output_dir}", mode: \'copy\'', modules_text)
        self.assertIn(
            "process DESKEW {\n"
            "    tag \"${cell_name ?: 'deskew'}\"\n\n"
            "    publishDir \"${params.output_dir}\", mode: 'copy'",
            modules_text,
        )
        self.assertNotIn("mode: 'move'", modules_text)
        self.assertNotIn("pattern: 'Top_shear'", modules_text)

    def test_config_exposes_optional_gpu_backend_not_psf_mode(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("deskew_backend = 'cpu_blocked'", config_text)
        self.assertIn("params.deskew_backend", config_text)
        self.assertIn("cuda", config_text)
        self.assertIn("withName: EXPORT_OUTPUT_FORMAT", config_text)
        self.assertIn("queue = 'super'", config_text)
        self.assertNotIn("psf_mode", config_text)

    def test_workflow_scripts_do_not_import_clearex_runtime(self):
        for script in (ROOT / "workflow/scripts").glob("*.py"):
            text = script.read_text(encoding="utf-8")
            self.assertNotIn("import clearex", text, msg=str(script))
            self.assertNotIn("from clearex", text, msg=str(script))

    def test_nextflow_passes_cpu_and_gpu_tuning_parameters(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("--deskew_backend ${params.deskew_backend}", modules_text)
        self.assertIn("--z_chunk ${params.z_chunk}", modules_text)
        self.assertIn("--deskew_prefetch ${params.deskew_prefetch}", modules_text)
        self.assertIn("--pyramid_max_downsample ${params.pyramid_max_downsample}", modules_text)
        self.assertIn("--deskew_output_dtype ${params.deskew_output_dtype}", modules_text)
        self.assertNotIn("--deskew_workers", modules_text)
        self.assertNotIn("--deskew_x_block", modules_text)
        self.assertNotIn("--deskew_cpu_schedule", modules_text)

    def test_cpu_scheduler_is_serial_only(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package = yaml.safe_load((ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8"))
        schema = {entry["id"]: entry for entry in package["workflow_parameters"]}

        self.assertNotIn("deskew_cpu_schedule", config_text)
        self.assertNotIn("deskew_cpu_schedule", schema)
        self.assertNotIn("deskew_workers", config_text)
        self.assertNotIn("deskew_workers", schema)
        self.assertNotIn("deskew_x_block", config_text)
        self.assertNotIn("deskew_x_block", schema)

    def test_pyramid_max_downsample_is_exposed_to_nextflow_and_astrocyte(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package = yaml.safe_load((ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8"))
        schema = {entry["id"]: entry for entry in package["workflow_parameters"]}

        self.assertIn("pyramid_max_downsample = 16", config_text)
        self.assertIn("pyramid_max_downsample", schema)
        self.assertEqual(schema["pyramid_max_downsample"]["type"], "select")
        self.assertEqual(schema["pyramid_max_downsample"]["default"], "16")
        self.assertEqual(
            [choice[0] for choice in schema["pyramid_max_downsample"]["choices"]],
            ["1", "2", "4", "8", "16"],
        )

    def test_output_formats_select_controls_ozx_and_optional_merged_tiff_export(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package = yaml.safe_load((ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8"))
        schema = {entry["id"]: entry for entry in package["workflow_parameters"]}

        self.assertIn('output_dir = "${baseDir}/output"', config_text)
        self.assertIn("cleanup = true", config_text)
        self.assertTrue((ROOT / "workflow/output" / ".keep").exists())
        self.assertIn("output_formats = 'ozx'", config_text)
        self.assertEqual(schema["output_formats"]["type"], "select")
        self.assertEqual(schema["output_formats"]["default"], "ozx")
        self.assertEqual([choice[0] for choice in schema["output_formats"]["choices"]], ["ozx", "tiff"])
        self.assertNotIn("'ome_zarr'", main_text)
        self.assertNotIn("'ome_zarr'", config_text)
        self.assertNotIn("'ome_zarr'", modules_text)
        self.assertIn("EXPORT_OUTPUT_FORMAT(DESKEW.out.deskewed_path, params.output_formats)", main_text)
        self.assertNotIn("if (params.output_formats", main_text)
        self.assertIn("enabled: false", modules_text)
        self.assertIn('path "deskewed_tiff", optional: true, emit: exported_output', modules_text)
        self.assertIn('path "deskewed_ozx", optional: true, emit: exported_ozx', modules_text)
        self.assertIn("--output \"deskewed_tiff\"", modules_text)
        self.assertIn("output_dir=deskewed_ozx", modules_text)
        self.assertIn('--output-format "ozx"', modules_text)
        self.assertIn("--output-format \"${output_format}\"", modules_text)
        self.assertNotIn("workflow.launchDir", modules_text)

    def test_ozx_is_supported_as_file_based_ome_zarr_transport(self):
        package = yaml.safe_load((ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8"))
        schema = {entry["id"]: entry for entry in package["workflow_parameters"]}
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("\\.ozx", schema["input"]["regex"])
        self.assertIn("\\.OZX", schema["input"]["regex"])
        self.assertIn("deskewed_ozx", modules_text)
        self.assertIn("output_dir=deskewed_ozx", modules_text)

    def test_workflow_does_not_reuse_or_build_conda_runtime(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package = yaml.safe_load((ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8"))
        schema = {entry["id"]: entry for entry in package["workflow_parameters"]}

        self.assertNotIn("deskew_runtime_dir", config_text)
        self.assertNotIn("deskew_runtime_dir", schema)
        self.assertNotIn("params.deskew_runtime_dir", main_text)
        self.assertNotIn("BUILD_DESKEW_CONTAINER", main_text)

    def test_workflow_does_not_export_runtime_for_deconvolution(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package = yaml.safe_load((ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8"))
        schema = {entry["id"]: entry for entry in package["workflow_parameters"]}

        self.assertNotIn("export_deskew_runtime", config_text)
        self.assertNotIn("export_deskew_runtime", modules_text)
        self.assertNotIn("export_deskew_runtime", schema)
        self.assertNotIn("pattern: 'deskew_runtime'", modules_text)

    def test_processes_do_not_accept_host_runtime_root_or_direct_conda_env(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertNotIn("${deskew_runtime}/deskew_env/bin", modules_text)
        self.assertNotIn("${deskew_runtime}/bin", modules_text)
        self.assertNotIn("deskew_env", modules_text)

    def test_runtime_builder_docs_removed(self):
        checked = [
            ROOT / "astrocyte_pkg.yml",
            ROOT / "README.md",
            ROOT / "docs" / "workflow-overview.md",
            ROOT / "docs" / "profiles-and-parameters.md",
            ROOT / "docs" / "outputs-and-troubleshooting.md",
        ]
        forbidden = [
            "deskew_runtime_dir",
            "export_deskew_runtime",
            "BUILD_DESKEW_CONTAINER",
            "conda runtime",
            "built deskew runtime",
        ]

        for path in checked:
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, msg=f"{value!r} remained in {path}")

    def test_stage_deskew_input_uses_super_queue(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("withName: STAGE_DESKEW_INPUT", config_text)
        self.assertIn("queue = 'super'", config_text)

    def test_sample_parameter_files_are_not_packaged(self):
        for params_name in (
            "tai_ricky_fused_skin_561.yml",
            "tai_ricky_fused_skin_561_cpu.yml",
            "tai_ricky_fused_skin_561_x32.yml",
            "tai_ricky_fused_skin_561_x32_cpu.yml",
        ):
            self.assertFalse((ROOT / params_name).exists(), params_name)

    def test_gpu_materialization_converts_big_endian_uint16_to_native(self):
        source = np.asarray([[[1, 256], [512, 1024]]], dtype=">u2")

        materialized = _materialize_volume(source)

        self.assertEqual(materialized.dtype, np.dtype("uint16"))
        self.assertTrue(materialized.flags.c_contiguous)
        np.testing.assert_array_equal(materialized, source.astype(np.uint16))

    def test_cpu_page_serial_scheduler_writes_pages_without_thread_pool(self):
        volume = np.arange(3 * 5 * 4, dtype=np.uint16).reshape(3, 5, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "serial_top_shear.tif"
            logs = io.StringIO()
            with contextlib.redirect_stdout(logs):
                output_shape = _write_top_shear(
                    volume,
                    output,
                    dx=1.0,
                    dz=1.0,
                    angle=30.0,
                    flip=1,
                    z_chunk=1,
                    pyramid_max_downsample=1,
                )

            written = tifffile.imread(output)

        self.assertIn("mode=page_serial", logs.getvalue())
        self.assertNotIn("mode=parallel", logs.getvalue())
        self.assertEqual(written.shape, (output_shape[2], output_shape[0], output_shape[1]))

    def test_cpu_top_shear_ome_zarr_is_stored_as_zyx(self):
        volume = np.arange(3 * 5 * 4, dtype=np.uint16).reshape(3, 5, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "top_shear.ome.zarr"
            with contextlib.redirect_stdout(io.StringIO()):
                output_shape = _write_top_shear(
                    volume,
                    output,
                    dx=1.0,
                    dz=1.0,
                    angle=30.0,
                    flip=1,
                    z_chunk=1,
                    pyramid_max_downsample=1,
                )

            written = open_ome_zarr_array(output)
            zattrs = json.loads((output / ".zattrs").read_text(encoding="utf-8"))

        self.assertEqual(written.shape, (output_shape[1], output_shape[0], output_shape[2]))
        self.assertEqual(
            [axis["name"] for axis in zattrs["multiscales"][0]["axes"]],
            ["z", "y", "x"],
        )

    def test_clearex_affine_geometry_matches_clearex_bounds_for_reference_slab(self):
        geometry = _clearex_affine_geometry(
            source_shape_zyx=(500, 2024, 128),
            dx=0.168,
            dz=0.2,
            angle=45.0,
            flip=1,
        )

        self.assertEqual(geometry.output_shape_zyx, (1305, 2148, 128))
        self.assertEqual(geometry.applied_rotation_deg_xyz, (-45.0, 0.0, 0.0))
        self.assertAlmostEqual(geometry.shear_yz, 0.70710678118, places=6)
        self.assertAlmostEqual(geometry.matrix_xyz[1, 2], 1.20710678118, places=6)

    def test_clearex_affine_sampler_uses_half_voxel_image_domain(self):
        volume = np.asarray([[[10.0, 20.0]]], dtype=np.float32)

        self.assertAlmostEqual(
            _linear_sample_support_normalized(volume, zf=0.0, yf=0.0, xf=-0.5),
            10.0,
        )
        self.assertEqual(
            _linear_sample_support_normalized(volume, zf=0.0, yf=0.0, xf=-0.5001),
            0.0,
        )
        self.assertAlmostEqual(
            _linear_sample_support_normalized(volume, zf=0.0, yf=0.0, xf=1.5),
            20.0,
        )
        self.assertEqual(
            _linear_sample_support_normalized(volume, zf=0.0, yf=0.0, xf=1.5001),
            0.0,
        )

    def test_clearex_affine_gpu_source_x_tiles_include_interpolation_halo(self):
        self.assertEqual(_source_x_bounds_for_output_tile(0, 256, 3584), (0, 257))
        self.assertEqual(_source_x_bounds_for_output_tile(256, 512, 3584), (255, 513))
        self.assertEqual(_source_x_bounds_for_output_tile(3328, 3584, 3584), (3327, 3584))

    def test_cpu_clearex_affine_ome_zarr_is_stored_as_zyx_with_metadata(self):
        volume = np.arange(4 * 6 * 5, dtype=np.uint16).reshape(4, 6, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "clearex_affine.ome.zarr"
            with contextlib.redirect_stdout(io.StringIO()):
                output_shape = _write_clearex_affine(
                    volume,
                    output,
                    dx=1.0,
                    dz=1.0,
                    angle=30.0,
                    flip=1,
                    z_chunk=2,
                    pyramid_max_downsample=1,
                )

            written = open_ome_zarr_array(output)
            zattrs = json.loads((output / ".zattrs").read_text(encoding="utf-8"))

        self.assertEqual(written.shape, output_shape)
        self.assertEqual(
            [axis["name"] for axis in zattrs["multiscales"][0]["axes"]],
            ["z", "y", "x"],
        )
        self.assertIn("clearex_affine", zattrs)
        self.assertEqual(zattrs["clearex_affine"]["voxel_size_um_zyx"], [1.0, 1.0, 1.0])

    def test_cpu_clearex_affine_can_write_float32_for_clearex_parity(self):
        volume = np.arange(4 * 6 * 5, dtype=np.uint16).reshape(4, 6, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "clearex_affine_float.ome.zarr"
            with contextlib.redirect_stdout(io.StringIO()):
                _write_clearex_affine(
                    volume,
                    output,
                    dx=1.0,
                    dz=1.0,
                    angle=30.0,
                    flip=1,
                    z_chunk=2,
                    pyramid_max_downsample=1,
                    output_dtype="float32",
                )

            written = open_ome_zarr_array(output)

        self.assertEqual(written.dtype, np.dtype("float32"))

    def test_run_chunked_deskew_note_reports_ome_zarr_zyx_layout(self):
        volume = np.arange(3 * 5 * 4, dtype=np.uint16).reshape(3, 5, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            source = create_ome_zarr_array(
                input_dir / "sample.ome.zarr",
                shape=volume.shape,
                chunks=(1, 5, 4),
                dtype=volume.dtype,
                max_downsample=1,
            )
            source[:] = volume

            with contextlib.redirect_stdout(io.StringIO()):
                run_chunked_deskew(
                    image_path=str(input_dir),
                    cell_name="",
                    dx=1.0,
                    dz=1.0,
                    angle=30.0,
                    flip=1,
                    output_dir=str(output_dir),
                    deskew_backend="cpu_blocked",
                    deskew_geometry="top_view",
                    z_chunk=1,
                    deskew_prefetch=1,
                    pyramid_max_downsample=1,
                )

            note_text = (output_dir / "Top_shear" / "note.txt").read_text(encoding="utf-8")

        self.assertIn("ome_zarr_level0_zyx=", note_text)

    def test_multiscales_metadata_respects_pyramid_max_downsample(self):
        metadata = multiscales_metadata("deskewed", max_downsample=4)

        datasets = metadata["multiscales"][0]["datasets"]
        self.assertEqual([dataset["path"] for dataset in datasets], ["0", "1", "2"])
        self.assertEqual(
            [dataset["coordinateTransformations"][0]["scale"] for dataset in datasets],
            [[1, 1, 1], [1, 2, 2], [1, 4, 4]],
        )

    def test_ome_zarr_writer_compresses_uint16_and_float32_losslessly(self):
        from numcodecs import Blosc

        volumes = [
            np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5),
            np.linspace(-1.5, 2.5, 2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for volume in volumes:
                output = Path(tmpdir) / f"sample_{volume.dtype}.ome.zarr"
                zarr_array = create_ome_zarr_array(
                    output,
                    shape=volume.shape,
                    chunks=(1, 2, 5),
                    dtype=volume.dtype,
                    max_downsample=1,
                )
                zarr_array[:] = volume

                reopened = open_ome_zarr_array(output)

                self.assertIsNotNone(reopened.compressor)
                self.assertEqual(reopened.compressor.cname, "zstd")
                self.assertEqual(reopened.compressor.shuffle, Blosc.BITSHUFFLE)
                np.testing.assert_array_equal(reopened[:], volume)

    def test_pyramid_levels_use_lossless_zarr_compression(self):
        import zarr
        from numcodecs import Blosc

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "sample.ome.zarr"
            base = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
            zarr_array = create_ome_zarr_array(
                output,
                shape=base.shape,
                chunks=(2, 3, 4),
                dtype=base.dtype,
                max_downsample=2,
            )
            zarr_array[:] = base

            with contextlib.redirect_stdout(io.StringIO()):
                write_downsampled_pyramid(output, max_downsample=2)

            level1 = zarr.open(str(output / "1"), mode="r")

            self.assertIsNotNone(level1.compressor)
            self.assertEqual(level1.compressor.cname, "zstd")
            self.assertEqual(level1.compressor.shuffle, Blosc.BITSHUFFLE)
            np.testing.assert_array_equal(level1[:], base[:, ::2, ::2])

    def test_pyramid_writer_logs_level_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "sample.ome.zarr"
            zarr_array = create_ome_zarr_array(
                output,
                shape=(4, 6, 6),
                chunks=(2, 3, 3),
                dtype=np.dtype("uint16"),
                max_downsample=4,
            )
            zarr_array[:] = np.arange(4 * 6 * 6, dtype=np.uint16).reshape(4, 6, 6)

            logs = io.StringIO()
            with contextlib.redirect_stdout(logs):
                write_downsampled_pyramid(output, max_downsample=4)

        self.assertIn("Finished OME-Zarr pyramid level:", logs.getvalue())
        self.assertIn("chunks_written=", logs.getvalue())

    def test_zyx_pyramid_downsamples_y_x_and_preserves_z(self):
        import zarr

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "sample.ome.zarr"
            base = np.arange(4 * 6 * 8, dtype=np.uint16).reshape(4, 6, 8)
            zarr_array = create_ome_zarr_array(
                output,
                shape=base.shape,
                chunks=(2, 3, 4),
                dtype=base.dtype,
                max_downsample=2,
            )
            zarr_array[:] = base

            with contextlib.redirect_stdout(io.StringIO()):
                write_downsampled_pyramid(output, max_downsample=2)

            level1 = zarr.open(str(output / "1"), mode="r")
            level1_shape = level1.shape
            level1_values = level1[:]
            zattrs = json.loads((output / ".zattrs").read_text(encoding="utf-8"))

        self.assertEqual(level1_shape, (4, 3, 4))
        np.testing.assert_array_equal(level1_values, base[:, ::2, ::2])
        self.assertEqual(
            zattrs["multiscales"][0]["datasets"][1]["coordinateTransformations"][0]["scale"],
            [1, 2, 2],
        )

    def test_chunked_deskew_cli_accepts_pyramid_max_downsample(self):
        import chunked_deskew as chunked_module

        old_run_chunked_deskew = chunked_module.run_chunked_deskew
        calls = []

        def fake_run_chunked_deskew(**kwargs):
            calls.append(kwargs)

        chunked_module.run_chunked_deskew = fake_run_chunked_deskew
        try:
            chunked_module.main([
                "--image_path",
                "input.ome.zarr",
                "--dx",
                "0.168",
                "--dz",
                "0.2",
                "--angle",
                "45",
                "--flip",
                "1",
                "--output_dir",
                "output",
                "--pyramid_max_downsample",
                "4",
                "--deskew_geometry",
                "clearex_affine",
                "--deskew_output_dtype",
                "float32",
            ])
        finally:
            chunked_module.run_chunked_deskew = old_run_chunked_deskew

        self.assertEqual(calls[0]["pyramid_max_downsample"], 4)
        self.assertEqual(calls[0]["deskew_geometry"], "clearex_affine")
        self.assertEqual(calls[0]["deskew_output_dtype"], "float32")

    def test_nextflow_passes_deskew_geometry_parameter(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("deskew_geometry = 'top_view'", config_text)
        self.assertIn("--deskew_geometry ${params.deskew_geometry}", modules_text)

    def test_deskew_output_dtype_is_exposed_to_nextflow_and_astrocyte(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")
        package = yaml.safe_load((ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8"))
        schema = {entry["id"]: entry for entry in package["workflow_parameters"]}

        self.assertIn("deskew_output_dtype = 'uint16'", config_text)
        self.assertIn("deskew_output_dtype", schema)
        self.assertEqual(schema["deskew_output_dtype"]["type"], "select")
        self.assertEqual(schema["deskew_output_dtype"]["default"], "uint16")
        self.assertEqual(
            [choice[0] for choice in schema["deskew_output_dtype"]["choices"]],
            ["uint16", "float32"],
        )

def load_export_module():
    script_path = SCRIPTS / "export_ome_zarr_to_tiff.py"
    spec = importlib.util.spec_from_file_location("export_ome_zarr_to_tiff", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_normalize_module():
    script_path = SCRIPTS / "normalize_input_to_ome_zarr.py"
    spec = importlib.util.spec_from_file_location("normalize_input_to_ome_zarr", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExportDeskewOmeZarrToTiffTest(unittest.TestCase):
    def setUp(self):
        self.module = load_export_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_export_directory_writes_one_merged_tiff_for_all_top_shear_zarrs(self):
        input_dir = self.root / "Top_shear"
        output_dir = self.root / "deskewed_tiff"
        (input_dir / "b_sample.ome.zarr").mkdir(parents=True)
        (input_dir / "a_sample.ome.zarr").mkdir(parents=True)
        written = []

        volumes = {
            "a_sample.ome.zarr": np.full((2, 3, 4), 1, dtype=np.uint16),
            "b_sample.ome.zarr": np.full((1, 3, 4), 2, dtype=np.uint16),
        }

        def fake_open(path, mode="r"):
            return volumes[Path(path).name]

        with mock.patch.object(self.module, "open_ome_zarr_array", side_effect=fake_open), \
                mock.patch.object(
                    self.module.tifffile,
                    "imwrite",
                    side_effect=lambda path, array, **kwargs: written.append((Path(path), array, kwargs)),
                ):
            output = self.module.export_directory(input_dir, output_dir)

        self.assertEqual(output, output_dir / "deskewed_merged.tif")
        self.assertEqual(written[0][0], output_dir / "deskewed_merged.tif")
        self.assertEqual(written[0][1].shape, (3, 3, 4))
        np.testing.assert_array_equal(written[0][1][:2], volumes["a_sample.ome.zarr"])
        np.testing.assert_array_equal(written[0][1][2:], volumes["b_sample.ome.zarr"])
        self.assertTrue(written[0][2]["bigtiff"])

    def test_export_directory_rejects_unsupported_format(self):
        input_dir = self.root / "Top_shear"
        input_dir.mkdir()

        with self.assertRaisesRegex(ValueError, "Unsupported output format"):
            self.module.export_directory(input_dir, self.root / "out", output_format="czi")

    def test_export_directory_writes_one_ozx_archive_per_ome_zarr(self):
        input_dir = self.root / "Top_shear"
        output_dir = self.root / "deskewed_ozx"
        zarr_dir = input_dir / "sample.ome.zarr"
        (zarr_dir / "0").mkdir(parents=True)
        (zarr_dir / ".zgroup").write_text("{}\n", encoding="utf-8")
        (zarr_dir / "0" / ".zarray").write_text("{}\n", encoding="utf-8")

        outputs = self.module.export_directory(input_dir, output_dir, output_format="ozx")

        self.assertEqual(outputs, [output_dir / "sample.ozx"])
        with zipfile.ZipFile(output_dir / "sample.ozx") as archive:
            self.assertEqual(
                sorted(archive.namelist()),
                ["sample.ome.zarr/.zgroup", "sample.ome.zarr/0/.zarray"],
            )


class NormalizeOzxInputTest(unittest.TestCase):
    def setUp(self):
        self.module = load_normalize_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_normalize_directory_unpacks_ozx_file_as_ome_zarr_input(self):
        input_dir = self.root / "input"
        input_dir.mkdir()
        archive_path = input_dir / "sample.ozx"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("sample.ome.zarr/.zgroup", "{}\n")
            archive.writestr("sample.ome.zarr/.zattrs", "{}\n")
            archive.writestr("sample.ome.zarr/0/.zarray", "{}\n")

        outputs = self.module.normalize_directory(input_dir, self.root / "output")

        self.assertEqual(outputs, [self.root / "output" / "sample.ome.zarr"])
        self.assertTrue((self.root / "output" / "sample.ome.zarr" / ".zgroup").exists())
        self.assertTrue((self.root / "output" / "sample.ome.zarr" / "0" / ".zarray").exists())


if __name__ == "__main__":
    unittest.main()

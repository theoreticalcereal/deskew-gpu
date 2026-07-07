import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tifffile
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "workflow" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from chunked_deskew import _materialize_volume, _write_top_shear
from ome_zarr_io import create_ome_zarr_array, multiscales_metadata, write_downsampled_pyramid


class DeskewWiringTest(unittest.TestCase):
    def test_main_wires_deskew_without_deconvolution(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")

        self.assertIn("include { BUILD_DESKEW_CONTAINER } from './modules'", main_text)
        self.assertIn("include { STAGE_DESKEW_INPUT } from './modules'", main_text)
        self.assertIn("include { DESKEW } from './modules'", main_text)
        self.assertNotIn("DECON", main_text)

    def test_modules_keep_only_deskew_processes(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("process BUILD_DESKEW_CONTAINER", modules_text)
        self.assertIn("process STAGE_DESKEW_INPUT", modules_text)
        self.assertIn("process DESKEW", modules_text)
        self.assertNotIn("process DECON", modules_text)

    def test_deskew_publishes_terminal_output_without_copying_tree(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn('publishDir "${params.output_dir}", mode: \'move\'', modules_text)
        self.assertNotIn('publishDir "${params.output_dir}", mode: \'copy\'', modules_text)

    def test_config_exposes_optional_gpu_backend_not_psf_mode(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("deskew_backend = 'cpu_blocked'", config_text)
        self.assertIn("params.deskew_backend", config_text)
        self.assertIn("cuda", config_text)
        self.assertNotIn("psf_mode", config_text)

    def test_nextflow_passes_cpu_and_gpu_tuning_parameters(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("--deskew_backend ${params.deskew_backend}", modules_text)
        self.assertIn("--z_chunk ${params.z_chunk}", modules_text)
        self.assertIn("--deskew_prefetch ${params.deskew_prefetch}", modules_text)
        self.assertIn("--pyramid_max_downsample ${params.pyramid_max_downsample}", modules_text)
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

    def test_workflow_can_reuse_prebuilt_deskew_runtime(self):
        main_text = (ROOT / "workflow/main.nf").read_text(encoding="utf-8")
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("deskew_runtime_dir = ''", config_text)
        self.assertIn("params.deskew_runtime_dir", main_text)
        self.assertIn("BUILD_DESKEW_CONTAINER()", main_text)
        self.assertIn("file(params.deskew_runtime_dir, checkIfExists: true)", main_text)

    def test_processes_accept_runtime_root_or_direct_conda_env(self):
        modules_text = (ROOT / "workflow/modules.nf").read_text(encoding="utf-8")

        self.assertIn("${deskew_runtime}/deskew_env/bin", modules_text)
        self.assertIn("${deskew_runtime}/bin", modules_text)
        self.assertIn("CONDA_PREFIX", modules_text)

    def test_stage_deskew_input_uses_super_queue(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("withName: STAGE_DESKEW_INPUT", config_text)
        self.assertIn("queue = 'super'", config_text)

    def test_tai_ricky_params_are_declared_in_astrocyte_schema(self):
        package = yaml.safe_load((ROOT / "astrocyte_pkg.yml").read_text(encoding="utf-8"))
        schema_ids = {entry["id"] for entry in package["workflow_parameters"]}

        for params_name in (
            "tai_ricky_fused_skin_561.yml",
            "tai_ricky_fused_skin_561_cpu.yml",
        ):
            params = yaml.safe_load((ROOT / params_name).read_text(encoding="utf-8"))
            self.assertLessEqual(set(params), schema_ids)

    def test_x32_comparison_params_use_absolute_output_dirs(self):
        for params_name in (
            "tai_ricky_fused_skin_561_x32.yml",
            "tai_ricky_fused_skin_561_x32_cpu.yml",
        ):
            params = yaml.safe_load((ROOT / params_name).read_text(encoding="utf-8"))
            self.assertTrue(Path(params["output_dir"]).is_absolute(), params_name)

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

    def test_multiscales_metadata_respects_pyramid_max_downsample(self):
        metadata = multiscales_metadata("deskewed", max_downsample=4)

        datasets = metadata["multiscales"][0]["datasets"]
        self.assertEqual([dataset["path"] for dataset in datasets], ["0", "1", "2"])
        self.assertEqual(
            [dataset["coordinateTransformations"][0]["scale"] for dataset in datasets],
            [[1, 1, 1], [1, 2, 2], [1, 4, 4]],
        )

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
            ])
        finally:
            chunked_module.run_chunked_deskew = old_run_chunked_deskew

        self.assertEqual(calls[0]["pyramid_max_downsample"], 4)

    def test_deskew_runtime_includes_decon_dependencies(self):
        conda_text = (ROOT / "workflow/envs/deskew-conda.txt").read_text(encoding="utf-8")
        pip_text = (ROOT / "workflow/envs/deskew-pip-requirements.txt").read_text(encoding="utf-8")

        self.assertIn("cudadecon=0.7.0", conda_text)
        self.assertIn("pycudadecon=0.5.1", conda_text)
        self.assertIn("psfmodels", pip_text)


if __name__ == "__main__":
    unittest.main()

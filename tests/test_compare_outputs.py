from pathlib import Path
import contextlib
import io
import json
import sys
import tempfile
import unittest

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "workflow" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from compare_deskew_outputs import (
    FWHM_FACTOR,
    fit_line_profile,
    gaussian_intensity_stats,
    gaussian_profile,
    global_ssim,
    comparison_from_yaml_files,
    main,
    normalized_cross_correlation,
    parse_trim_zyx,
    sample_indices,
    compare_outputs,
)


class CompareDeskewOutputsTest(unittest.TestCase):
    def test_gaussian_intensity_stats_reports_fwhm_from_sigma(self):
        values = np.asarray([0.0, 2.0, 4.0], dtype=np.float32)

        stats = gaussian_intensity_stats(values)

        self.assertEqual(stats["count"], 3)
        self.assertAlmostEqual(stats["mean"], 2.0)
        expected_sigma = float(np.std(values.astype(np.float64)))
        self.assertAlmostEqual(stats["sigma"], expected_sigma)
        self.assertAlmostEqual(stats["fwhm"], expected_sigma * FWHM_FACTOR)

    def test_fit_line_profile_matches_clearex_fwhm_and_reports_r2(self):
        x_axis = np.arange(21, dtype=np.float64)
        sigma = 2.0
        lateral_pixel_size = 0.5
        profile = gaussian_profile(
            x_axis,
            amplitude=10.0,
            x_offset=10.0,
            sigma=sigma,
            y_offset=3.0,
        )

        fit = fit_line_profile(x_axis, profile, lateral_pixel_size=lateral_pixel_size)

        self.assertAlmostEqual(fit["fwhm"], FWHM_FACTOR * sigma * lateral_pixel_size)
        self.assertAlmostEqual(fit["sigma"], sigma)
        self.assertAlmostEqual(fit["r2"], 1.0)

    def test_normalized_cross_correlation_identical_is_one(self):
        values = np.arange(9, dtype=np.float32).reshape(3, 3)

        self.assertAlmostEqual(normalized_cross_correlation(values, values), 1.0)

    def test_global_ssim_identical_is_one(self):
        values = np.arange(9, dtype=np.float32).reshape(3, 3)

        self.assertAlmostEqual(global_ssim(values, values), 1.0)

    def test_sample_indices_include_endpoints(self):
        self.assertEqual(sample_indices(10, 4), [0, 3, 6, 9])

    def test_compare_outputs_writes_line_profile_metrics_to_json(self):
        import compare_deskew_outputs as compare_module

        x_axis = np.arange(9, dtype=np.float64)
        profile = gaussian_profile(x_axis, 100.0, 4.0, 1.5, 7.0)
        page = np.outer(profile, profile).astype(np.float32)
        reference = np.stack([page, page * 1.1], axis=0)
        candidate = reference.copy()
        old_open_volume = compare_module._open_volume

        def fake_open_volume(path, *, level="0", component=None, tpc=(0, 0, 0)):
            return reference if Path(path).name == "cpu" else candidate

        compare_module._open_volume = fake_open_volume
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_json = Path(tmpdir) / "metrics.json"

                summary = compare_outputs(
                    reference_path=Path("cpu"),
                    candidate_path=Path("gpu"),
                    output_json=output_json,
                    sample_count=1,
                )

                payload = json.loads(output_json.read_text(encoding="utf-8"))
        finally:
            compare_module._open_volume = old_open_volume

        self.assertEqual(summary["samples"], 1)
        self.assertEqual(payload["samples"], 1)
        self.assertIn("line_profiles", payload)
        self.assertEqual(len(payload["pages"][0]["line_profiles"]), 2)
        first_profile = payload["pages"][0]["line_profiles"][0]
        self.assertAlmostEqual(first_profile["ncc"], 1.0)
        self.assertAlmostEqual(first_profile["ssim"], 1.0)
        self.assertGreater(first_profile["cpu"]["r2"], 0.999)
        self.assertGreater(first_profile["gpu"]["r2"], 0.999)
        self.assertAlmostEqual(first_profile["fwhm_delta"], 0.0)

    def test_comparison_from_yaml_files_infers_outputs_and_pixel_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cpu_params = Path(tmpdir) / "cpu.yml"
            gpu_params = Path(tmpdir) / "gpu.yml"
            cpu_params.write_text(
                "\n".join([
                    "input:",
                    "  - /data/fused_skin_561.tif",
                    "output_dir: ./cpu_output",
                    "dx: 0.168",
                    "deskew_backend: cpu_blocked",
                ]),
                encoding="utf-8",
            )
            gpu_params.write_text(
                "\n".join([
                    "input:",
                    "  - /data/fused_skin_561.tif",
                    "output_dir: ./gpu_output",
                    "dx: 0.168",
                    "deskew_backend: gpu",
                ]),
                encoding="utf-8",
            )

            settings = comparison_from_yaml_files(cpu_params, gpu_params)

        self.assertEqual(
            settings["reference_path"],
            Path("cpu_output") / "Top_shear" / "fused_skin_561.ome.zarr",
        )
        self.assertEqual(
            settings["candidate_path"],
            Path("gpu_output") / "Top_shear" / "fused_skin_561.ome.zarr",
        )
        self.assertEqual(settings["lateral_pixel_size"], 0.168)

    def test_main_accepts_cpu_and_gpu_yaml_params(self):
        import compare_deskew_outputs as compare_module

        old_compare_outputs = compare_module.compare_outputs
        calls = []

        def fake_compare_outputs(**kwargs):
            calls.append(kwargs)
            return {"ok": True}

        compare_module.compare_outputs = fake_compare_outputs
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cpu_params = Path(tmpdir) / "cpu.yml"
                gpu_params = Path(tmpdir) / "gpu.yml"
                cpu_params.write_text(
                    "input: [/data/sample.tif]\noutput_dir: cpu_out\ndx: 0.2\n",
                    encoding="utf-8",
                )
                gpu_params.write_text(
                    "input: [/data/sample.tif]\noutput_dir: gpu_out\ndx: 0.2\n",
                    encoding="utf-8",
                )

                with contextlib.redirect_stdout(io.StringIO()):
                    main([
                        "--cpu_params",
                        str(cpu_params),
                        "--gpu_params",
                        str(gpu_params),
                        "--sample_count",
                        "3",
                    ])
        finally:
            compare_module.compare_outputs = old_compare_outputs

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["reference_path"], Path("cpu_out/Top_shear/sample.ome.zarr"))
        self.assertEqual(calls[0]["candidate_path"], Path("gpu_out/Top_shear/sample.ome.zarr"))
        self.assertEqual(calls[0]["sample_count"], 3)
        self.assertEqual(calls[0]["lateral_pixel_size"], 0.2)

    def test_compare_outputs_accepts_clearex_6d_reference_component(self):
        data = np.arange(1 * 1 * 1 * 2 * 9 * 9, dtype=np.float32).reshape(
            (1, 1, 1, 2, 9, 9)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            reference_path = tmp_path / "clearex_store.zarr"
            candidate_path = tmp_path / "candidate.ome.zarr"
            component = "clearex/runtime_cache/results/shear_transform/latest/data"

            root = zarr.open_group(str(reference_path), mode="w")
            reference = root.create_dataset(
                component,
                data=data,
                chunks=(1, 1, 1, 1, 9, 9),
                overwrite=True,
            )
            reference.attrs["axes"] = ["t", "p", "c", "z", "y", "x"]
            reference.attrs["voxel_size_um_zyx"] = [2.0, 0.5, 0.5]
            reference.attrs["affine_matrix_xyz"] = np.eye(3).tolist()

            candidate = zarr.open(
                str(candidate_path / "0"),
                mode="w",
                shape=data.shape[3:],
                chunks=(1, 9, 9),
                dtype=data.dtype,
            )
            candidate[:] = data[0, 0, 0]

            summary = compare_outputs(
                reference_path=reference_path,
                candidate_path=candidate_path,
                reference_component=component,
                sample_count=1,
            )

        self.assertEqual(summary["shape"], (2, 9, 9))
        self.assertEqual(summary["metadata"]["reference"]["component"], component)
        self.assertEqual(
            summary["metadata"]["reference"]["axes"],
            ["t", "p", "c", "z", "y", "x"],
        )
        self.assertEqual(summary["metadata"]["reference"]["voxel_size_um_zyx"], [2.0, 0.5, 0.5])
        self.assertEqual(summary["metadata"]["candidate"]["path"], str(candidate_path))

    def test_compare_outputs_can_trim_edges_before_sampling(self):
        import compare_deskew_outputs as compare_module

        reference = np.ones((6, 5, 4), dtype=np.float32)
        reference[0, :, :] = 100.0
        reference[-1, :, :] = 200.0
        candidate = reference.copy()
        old_open_volume = compare_module._open_volume

        def fake_open_volume(path, *, level="0", component=None, tpc=(0, 0, 0)):
            return reference if Path(path).name == "cpu" else candidate

        compare_module._open_volume = fake_open_volume
        try:
            summary = compare_outputs(
                reference_path=Path("cpu"),
                candidate_path=Path("gpu"),
                sample_count=4,
                trim_zyx=(1, 0, 0),
            )
        finally:
            compare_module._open_volume = old_open_volume

        self.assertEqual(summary["original_shape"], (6, 5, 4))
        self.assertEqual(summary["shape"], (4, 5, 4))
        self.assertEqual(summary["crop"]["zyx"], [[1, 5], [0, 5], [0, 4]])
        self.assertEqual([page["i"] for page in summary["pages"]], [1, 2, 3, 4])
        self.assertAlmostEqual(summary["brightness"]["reference_mean"], 1.0)

    def test_parse_trim_zyx_requires_three_nonnegative_values(self):
        self.assertEqual(parse_trim_zyx("32,0,1"), (32, 0, 1))
        with self.assertRaisesRegex(ValueError, "three comma-separated"):
            parse_trim_zyx("32,0")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            parse_trim_zyx("-1,0,0")

    def test_main_accepts_reference_component_cli(self):
        import compare_deskew_outputs as compare_module

        old_compare_outputs = compare_module.compare_outputs
        calls = []

        def fake_compare_outputs(**kwargs):
            calls.append(kwargs)
            return {"ok": True}

        compare_module.compare_outputs = fake_compare_outputs
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                main([
                    "--reference",
                    "clearex_store.zarr",
                    "--candidate",
                    "candidate.ome.zarr",
                    "--reference_component",
                    "clearex/runtime_cache/results/shear_transform/latest/data",
                    "--reference_tpc",
                    "1,2,3",
                    "--sample_count",
                    "4",
                    "--trim_zyx",
                    "32,0,0",
                ])
        finally:
            compare_module.compare_outputs = old_compare_outputs

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["reference_path"], Path("clearex_store.zarr"))
        self.assertEqual(calls[0]["candidate_path"], Path("candidate.ome.zarr"))
        self.assertEqual(
            calls[0]["reference_component"],
            "clearex/runtime_cache/results/shear_transform/latest/data",
        )
        self.assertEqual(calls[0]["reference_tpc"], (1, 2, 3))
        self.assertEqual(calls[0]["sample_count"], 4)
        self.assertEqual(calls[0]["trim_zyx"], (32, 0, 0))

    def test_open_volume_reports_missing_comparison_output(self):
        import compare_deskew_outputs as compare_module

        with self.assertRaisesRegex(FileNotFoundError, "Comparison input not found"):
            compare_module._open_volume(Path("missing/Top_shear/sample.ome.zarr"))


if __name__ == "__main__":
    unittest.main()

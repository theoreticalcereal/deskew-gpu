from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


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

    def test_config_exposes_deskew_backend_not_psf_mode(self):
        config_text = (ROOT / "workflow/configs/nextflow.config").read_text(encoding="utf-8")

        self.assertIn("deskew_backend = 'gpu'", config_text)
        self.assertNotIn("psf_mode", config_text)

    def test_deskew_runtime_includes_decon_dependencies(self):
        conda_text = (ROOT / "workflow/envs/deskew-conda.txt").read_text(encoding="utf-8")
        pip_text = (ROOT / "workflow/envs/deskew-pip-requirements.txt").read_text(encoding="utf-8")

        self.assertIn("cudadecon=0.7.0", conda_text)
        self.assertIn("pycudadecon=0.5.1", conda_text)
        self.assertIn("psfmodels", pip_text)


if __name__ == "__main__":
    unittest.main()

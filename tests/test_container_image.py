from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "workflow" / "images" / "deskew-gpu"
IMAGE = "git.biohpc.swmed.edu:5050/dean-lab/ctaslm2-deskew:0.1.0"


class DeskewContainerImageTests(unittest.TestCase):
    def test_image_context_builds_cuda_enabled_app_environment(self):
        dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
        environment = (IMAGE_ROOT / "environment.yml").read_text(encoding="utf-8")
        build_script = (IMAGE_ROOT / "build-deskew-image.sh").read_text(encoding="utf-8")

        self.assertIn("FROM docker.io/continuumio/miniconda3:", dockerfile)
        self.assertIn("libgomp1", dockerfile)
        self.assertIn("conda env create --name app --file /tmp/environment.yml", dockerfile)
        self.assertIn("ENV PATH=/opt/conda/envs/app/bin:/opt/conda/bin:$PATH", dockerfile)

        for dependency in (
            "python=3.10",
            "cudatoolkit=11.8",
            "cudadecon=0.7.0",
            "pycudadecon=0.5.1",
            "numpy",
            "numba",
            "tifffile=2025.5.10",
            "zarr=2.18.3",
            "numcodecs",
            "h5py",
            "aicsimageio",
            "nd2",
            "readlif",
        ):
            self.assertIn(dependency, environment)

        self.assertIn(IMAGE, build_script)
        self.assertNotIn("cuda/11.8.0", build_script)
        self.assertIn("singularity exec --nv", build_script)
        self.assertIn("pycudadecon", build_script)


if __name__ == "__main__":
    unittest.main()

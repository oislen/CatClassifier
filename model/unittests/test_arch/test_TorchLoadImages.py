import os
import sys
import tempfile
import unittest

from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from arch.load_image_v2 import TorchLoadImages

# ---------------------------------------------------------------------------
# Shared transforms and fixture helpers
# ---------------------------------------------------------------------------

_torch_transforms = transforms.Compose(
    [
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ]
)


def _make_jpeg(path: str, mode: str = "RGB", size: tuple = (10, 10)) -> str:
    """Create a minimal synthetic JPEG at *path* and return the path."""
    Image.new(mode, size, color=(128, 64, 32) if mode == "RGB" else 128).save(path, format="JPEG")
    return path


# ---------------------------------------------------------------------------
# Module-level fixtures — created once for the test module
# ---------------------------------------------------------------------------

_tmp_dir = tempfile.mkdtemp()
_fixture_paths = [
    _make_jpeg(os.path.join(_tmp_dir, "cat.0.jpg")),
    _make_jpeg(os.path.join(_tmp_dir, "dog.1.jpg")),
    _make_jpeg(os.path.join(_tmp_dir, "cat.2.jpg")),
]

_loader = TorchLoadImages(torch_transforms=_torch_transforms, n_workers=None)
_single_record = _loader.load_image(_fixture_paths[0])
_sequential_records = _loader.load_images(_fixture_paths)


class TestLoadImage(unittest.TestCase):
    """Unit tests for TorchLoadImages.load_image."""

    def setUp(self):
        self.record = _single_record

    # ------------------------------------------------------------------
    # Return type
    # ------------------------------------------------------------------

    def test_load_image_returns_dict(self):
        self.assertIsInstance(self.record, dict)

    # ------------------------------------------------------------------
    # images column removed
    # ------------------------------------------------------------------

    def test_load_image_record_has_no_images_key(self):
        """Regression guard: raw pixel array must not be stored in the record."""
        self.assertNotIn("images", self.record)

    # ------------------------------------------------------------------
    # ndims
    # ------------------------------------------------------------------

    def test_load_image_record_has_ndims_key(self):
        self.assertIn("ndims", self.record)

    def test_load_image_ndims_is_int(self):
        self.assertIsInstance(self.record["ndims"], int)

    def test_load_image_ndims_value_for_rgb(self):
        """An RGB JPEG must produce ndims == 3."""
        self.assertEqual(self.record["ndims"], 3)

    # ------------------------------------------------------------------
    # Other expected keys present
    # ------------------------------------------------------------------

    def test_load_image_has_filepaths_key(self):
        self.assertIn("filepaths", self.record)

    def test_load_image_has_filenames_key(self):
        self.assertIn("filenames", self.record)

    def test_load_image_has_image_tensors_key(self):
        self.assertIn("image_tensors", self.record)

    def test_load_image_has_category_tensors_key(self):
        self.assertIn("category_tensors", self.record)

    def test_load_image_has_torch_transform_error_key(self):
        self.assertIn("torch_transform_error", self.record)

    def test_load_image_transform_error_is_none_for_valid_rgb(self):
        self.assertIsNone(self.record["torch_transform_error"])


class TestLoadImages(unittest.TestCase):
    """Unit tests for TorchLoadImages.load_images (sequential and parallel)."""

    # ------------------------------------------------------------------
    # Sequential path
    # ------------------------------------------------------------------

    def test_load_images_sequential_returns_list(self):
        self.assertIsInstance(_sequential_records, list)

    def test_load_images_sequential_count(self):
        self.assertEqual(len(_sequential_records), len(_fixture_paths))

    def test_load_images_sequential_each_record_is_dict(self):
        for rec in _sequential_records:
            self.assertIsInstance(rec, dict)

    # ------------------------------------------------------------------
    # n_workers <= 1 guard — falls through to sequential
    # ------------------------------------------------------------------

    def test_load_images_n_workers_1_uses_sequential_path(self):
        """n_workers=1 must produce the same number of records as n_workers=None."""
        loader_1 = TorchLoadImages(torch_transforms=_torch_transforms, n_workers=1)
        records = loader_1.load_images(_fixture_paths)
        self.assertEqual(len(records), len(_fixture_paths))

    def test_load_images_n_workers_0_uses_sequential_path(self):
        """n_workers=0 must not raise and must return sequential results."""
        loader_0 = TorchLoadImages(torch_transforms=_torch_transforms, n_workers=0)
        records = loader_0.load_images(_fixture_paths)
        self.assertEqual(len(records), len(_fixture_paths))

    # ------------------------------------------------------------------
    # Parallel path
    # ------------------------------------------------------------------

    def test_load_images_parallel_count(self):
        """Parallel path must return the same number of records as sequential."""
        loader_p = TorchLoadImages(torch_transforms=_torch_transforms, n_workers=2)
        records = loader_p.load_images(_fixture_paths)
        self.assertEqual(len(records), len(_fixture_paths))

    def test_load_images_parallel_results_match_sequential_filenames(self):
        """Parallel results must contain the same filenames as sequential results."""
        loader_p = TorchLoadImages(torch_transforms=_torch_transforms, n_workers=2)
        parallel_records = loader_p.load_images(_fixture_paths)
        seq_filenames = sorted(r["filenames"] for r in _sequential_records)
        par_filenames = sorted(r["filenames"] for r in parallel_records)
        self.assertEqual(seq_filenames, par_filenames)

    def test_load_images_parallel_no_images_key(self):
        """Parallel records must also not contain the images key."""
        loader_p = TorchLoadImages(torch_transforms=_torch_transforms, n_workers=2)
        parallel_records = loader_p.load_images(_fixture_paths)
        for rec in parallel_records:
            self.assertNotIn("images", rec)

    def test_load_images_parallel_ndims_match_sequential(self):
        """ndims values from the parallel path must match the sequential path."""
        loader_p = TorchLoadImages(torch_transforms=_torch_transforms, n_workers=2)
        parallel_records = loader_p.load_images(_fixture_paths)
        seq_ndims = sorted(r["ndims"] for r in _sequential_records)
        par_ndims = sorted(r["ndims"] for r in parallel_records)
        self.assertEqual(seq_ndims, par_ndims)


if __name__ == "__main__":
    unittest.main()

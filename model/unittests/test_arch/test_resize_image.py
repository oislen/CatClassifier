import os
import sys
import unittest
import unittest.mock

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

with unittest.mock.patch("beartype.beartype", new=lambda f, **_kw: f):
    from arch.resize_image import resize_image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

np.random.seed(42)

# 8x8 RGB image
rgb_8x8 = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
# 8x8 single-channel greyscale image (3-D with 1 channel)
grey_8x8 = np.random.randint(0, 256, (8, 8, 1), dtype=np.uint8)

target_wh = (4, 4)   # resize to 4 wide, 4 tall

obs_resized_rgb        = resize_image(rgb_8x8,  reshape_wh=target_wh)
obs_resized_grey_3dim  = resize_image(grey_8x8, reshape_wh=target_wh, keep_3dim=True)
obs_resized_grey_2dim  = resize_image(grey_8x8, reshape_wh=target_wh, keep_3dim=False)

# Upscale test
obs_upscaled = resize_image(rgb_8x8, reshape_wh=(16, 16))

# Non-square target: 6 wide x 3 tall
obs_nonsquare = resize_image(rgb_8x8, reshape_wh=(6, 3))

# Identity resize: same dimensions as input
obs_identity = resize_image(rgb_8x8, reshape_wh=(8, 8))

# Nearest-neighbour interpolation
obs_nearest = resize_image(rgb_8x8, reshape_wh=(4, 4), interpolation=cv2.INTER_NEAREST)

# Float32 input
rgb_float = np.random.rand(8, 8, 3).astype(np.float32)
obs_float_resized = resize_image(rgb_float, reshape_wh=(4, 4))

# Tall non-square: resize a wide image to a tall narrow one
rgb_wide = np.random.randint(0, 256, (4, 16, 3), dtype=np.uint8)
obs_tall_narrow = resize_image(rgb_wide, reshape_wh=(2, 8))


class TestResizeImage(unittest.TestCase):
    """
    Unit tests for resize_image.

    Verifies that RGB and greyscale arrays are resized to the requested
    (width, height) target, that keep_3dim correctly controls output
    dimensions, and that various dtypes and interpolation modes work.
    """

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_type_rgb(self):
        self.assertIsInstance(obs_resized_rgb, np.ndarray)

    def test_type_grey(self):
        self.assertIsInstance(obs_resized_grey_3dim, np.ndarray)

    def test_type_float(self):
        self.assertIsInstance(obs_float_resized, np.ndarray)

    # ------------------------------------------------------------------
    # Shape tests (downscale)
    # ------------------------------------------------------------------

    def test_shape_rgb_downscale(self):
        """Output shape should be (target_h, target_w, channels)."""
        self.assertEqual(obs_resized_rgb.shape, (4, 4, 3))

    def test_shape_grey_keep_3dim_true(self):
        """Single-channel output should remain 3-D when keep_3dim=True."""
        self.assertEqual(obs_resized_grey_3dim.shape, (4, 4, 1))

    def test_shape_grey_keep_3dim_false(self):
        """Single-channel output should become 2-D when keep_3dim=False."""
        self.assertEqual(obs_resized_grey_2dim.shape, (4, 4))

    # ------------------------------------------------------------------
    # Shape tests (upscale)
    # ------------------------------------------------------------------

    def test_shape_rgb_upscale(self):
        self.assertEqual(obs_upscaled.shape, (16, 16, 3))

    # ------------------------------------------------------------------
    # Shape tests (non-square target)
    # ------------------------------------------------------------------

    def test_shape_nonsquare_target(self):
        """Non-square (w=6, h=3) target -> output shape (3, 6, 3)."""
        self.assertEqual(obs_nonsquare.shape, (3, 6, 3))

    def test_shape_tall_narrow(self):
        """Wide input resized to narrow tall target -> (8, 2, 3)."""
        self.assertEqual(obs_tall_narrow.shape, (8, 2, 3))

    # ------------------------------------------------------------------
    # Identity resize tests
    # ------------------------------------------------------------------

    def test_shape_identity(self):
        """Resizing to the same dimensions should preserve shape."""
        self.assertEqual(obs_identity.shape, rgb_8x8.shape)

    # ------------------------------------------------------------------
    # Dimension tests
    # ------------------------------------------------------------------

    def test_ndim_rgb(self):
        self.assertEqual(obs_resized_rgb.ndim, 3)

    def test_ndim_grey_3dim(self):
        self.assertEqual(obs_resized_grey_3dim.ndim, 3)

    def test_ndim_grey_2dim(self):
        self.assertEqual(obs_resized_grey_2dim.ndim, 2)

    def test_ndim_nonsquare(self):
        self.assertEqual(obs_nonsquare.ndim, 3)

    # ------------------------------------------------------------------
    # dtype tests
    # ------------------------------------------------------------------

    def test_dtype_rgb(self):
        self.assertEqual(obs_resized_rgb.dtype, rgb_8x8.dtype)

    def test_dtype_grey(self):
        self.assertEqual(obs_resized_grey_3dim.dtype, grey_8x8.dtype)

    def test_dtype_float32_preserved(self):
        self.assertEqual(obs_float_resized.dtype, np.float32)

    def test_dtype_upscale(self):
        self.assertEqual(obs_upscaled.dtype, rgb_8x8.dtype)

    # ------------------------------------------------------------------
    # Channel count preserved
    # ------------------------------------------------------------------

    def test_channel_count_rgb(self):
        self.assertEqual(obs_resized_rgb.shape[2], 3)

    def test_channel_count_grey(self):
        self.assertEqual(obs_resized_grey_3dim.shape[2], 1)

    def test_channel_count_upscale(self):
        self.assertEqual(obs_upscaled.shape[2], 3)

    # ------------------------------------------------------------------
    # Pixel value range preserved
    # ------------------------------------------------------------------

    def test_pixel_range_uint8(self):
        """All output pixels must be in [0, 255]."""
        self.assertTrue(int(obs_resized_rgb.min()) >= 0)
        self.assertTrue(int(obs_resized_rgb.max()) <= 255)

    def test_nearest_neighbour_dtype(self):
        """INTER_NEAREST should preserve dtype."""
        self.assertEqual(obs_nearest.dtype, rgb_8x8.dtype)

    def test_nearest_neighbour_shape(self):
        self.assertEqual(obs_nearest.shape, (4, 4, 3))

    # ------------------------------------------------------------------
    # keep_3dim consistency test
    # ------------------------------------------------------------------

    def test_keep_3dim_squeeze_consistent(self):
        """Squeezing the 3-dim grey result should equal the 2-dim grey result."""
        np.testing.assert_array_equal(
            obs_resized_grey_3dim[:, :, 0], obs_resized_grey_2dim
        )


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

with unittest.mock.patch("beartype.beartype", new=lambda f, **_kw: f):
    from arch.crop_image import crop_image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 10x10 RGB image with a unique value at every pixel for easy verification
rgb_10x10 = np.arange(300, dtype=np.uint8).reshape((10, 10, 3))
grey_10x10 = np.arange(100, dtype=np.uint8).reshape((10, 10))

# Standard symmetric crop: top-left (2, 2) -> bottom-right (6, 6)
pt1_wh = (2, 2)
pt2_wh = (6, 6)

obs_crop_rgb = crop_image(rgb_10x10, pt1_wh=pt1_wh, pt2_wh=pt2_wh)
obs_crop_grey = crop_image(grey_10x10, pt1_wh=pt1_wh, pt2_wh=pt2_wh)

# Expected crop: rows [2:6], cols [2:6]
exp_crop_rgb = rgb_10x10[2:6, 2:6, :]
exp_crop_grey = grey_10x10[2:6, 2:6]

# Non-square crop: different width and height extents
pt1_ns = (1, 3)  # (w=1, h=3)
pt2_ns = (4, 8)  # (w=4, h=8)
obs_crop_nonsquare_rgb = crop_image(rgb_10x10, pt1_wh=pt1_ns, pt2_wh=pt2_ns)
obs_crop_nonsquare_grey = crop_image(grey_10x10, pt1_wh=pt1_ns, pt2_wh=pt2_ns)
# function: wpts=(h1,h2)=(3,8), hpts=(w1,w2)=(1,4) -> rows [1:4], cols [3:8]
exp_crop_nonsquare_rgb = rgb_10x10[1:4, 3:8, :]
exp_crop_nonsquare_grey = grey_10x10[1:4, 3:8]

# Single-pixel crop: both points define a 1x1 region
pt1_sp = (3, 3)
pt2_sp = (4, 4)
obs_crop_single_rgb = crop_image(rgb_10x10, pt1_wh=pt1_sp, pt2_wh=pt2_sp)
obs_crop_single_grey = crop_image(grey_10x10, pt1_wh=pt1_sp, pt2_wh=pt2_sp)

# Full-image crop: points span the entire image
pt1_full = (0, 0)
pt2_full = (10, 10)
obs_crop_full_rgb = crop_image(rgb_10x10, pt1_wh=pt1_full, pt2_wh=pt2_full)
obs_crop_full_grey = crop_image(grey_10x10, pt1_wh=pt1_full, pt2_wh=pt2_full)

# Crop with list arguments (not tuples)
obs_crop_list = crop_image(rgb_10x10, pt1_wh=[2, 2], pt2_wh=[6, 6])


class TestCropImage(unittest.TestCase):
    """
    Unit tests for crop_image.

    Verifies that 3-D (RGB) and 2-D (greyscale) image arrays are correctly
    cropped to the region defined by two diagonal points, including non-square
    crops, single-pixel crops, full-image crops, and reversed-point symmetry.
    """

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_type_rgb(self):
        self.assertIsInstance(obs_crop_rgb, np.ndarray)

    def test_type_grey(self):
        self.assertIsInstance(obs_crop_grey, np.ndarray)

    def test_type_single_pixel_rgb(self):
        self.assertIsInstance(obs_crop_single_rgb, np.ndarray)

    # ------------------------------------------------------------------
    # Shape tests — standard crop
    # ------------------------------------------------------------------

    def test_shape_rgb(self):
        """Cropped RGB array should have shape (4, 4, 3)."""
        self.assertEqual(obs_crop_rgb.shape, (4, 4, 3))

    def test_shape_grey(self):
        """Cropped greyscale array should have shape (4, 4)."""
        self.assertEqual(obs_crop_grey.shape, (4, 4))

    def test_ndim_rgb(self):
        self.assertEqual(obs_crop_rgb.ndim, 3)

    def test_ndim_grey(self):
        self.assertEqual(obs_crop_grey.ndim, 2)

    # ------------------------------------------------------------------
    # Shape tests — non-square crop
    # ------------------------------------------------------------------

    def test_shape_nonsquare_rgb(self):
        """Non-square crop: rows [1:4] x cols [3:8] -> (3, 5, 3)."""
        self.assertEqual(obs_crop_nonsquare_rgb.shape, (3, 5, 3))

    def test_shape_nonsquare_grey(self):
        self.assertEqual(obs_crop_nonsquare_grey.shape, (3, 5))

    # ------------------------------------------------------------------
    # Shape tests — single-pixel crop
    # ------------------------------------------------------------------

    def test_shape_single_pixel_rgb(self):
        self.assertEqual(obs_crop_single_rgb.shape, (1, 1, 3))

    def test_shape_single_pixel_grey(self):
        self.assertEqual(obs_crop_single_grey.shape, (1, 1))

    # ------------------------------------------------------------------
    # Shape tests — full-image crop
    # ------------------------------------------------------------------

    def test_shape_full_image_rgb(self):
        self.assertEqual(obs_crop_full_rgb.shape, rgb_10x10.shape)

    def test_shape_full_image_grey(self):
        self.assertEqual(obs_crop_full_grey.shape, grey_10x10.shape)

    # ------------------------------------------------------------------
    # Value tests
    # ------------------------------------------------------------------

    def test_values_rgb(self):
        np.testing.assert_array_equal(obs_crop_rgb, exp_crop_rgb)

    def test_values_grey(self):
        np.testing.assert_array_equal(obs_crop_grey, exp_crop_grey)

    def test_values_nonsquare_rgb(self):
        np.testing.assert_array_equal(obs_crop_nonsquare_rgb, exp_crop_nonsquare_rgb)

    def test_values_nonsquare_grey(self):
        np.testing.assert_array_equal(obs_crop_nonsquare_grey, exp_crop_nonsquare_grey)

    def test_values_full_image_rgb(self):
        np.testing.assert_array_equal(obs_crop_full_rgb, rgb_10x10)

    def test_values_full_image_grey(self):
        np.testing.assert_array_equal(obs_crop_full_grey, grey_10x10)

    # ------------------------------------------------------------------
    # Channel value tests
    # ------------------------------------------------------------------

    def test_channel_values_preserved(self):
        """Each channel slice of the crop should equal the raw numpy slice."""
        for c in range(3):
            np.testing.assert_array_equal(obs_crop_rgb[:, :, c], exp_crop_rgb[:, :, c])

    # ------------------------------------------------------------------
    # dtype preservation
    # ------------------------------------------------------------------

    def test_dtype_rgb(self):
        self.assertEqual(obs_crop_rgb.dtype, rgb_10x10.dtype)

    def test_dtype_grey(self):
        self.assertEqual(obs_crop_grey.dtype, grey_10x10.dtype)

    def test_dtype_nonsquare(self):
        self.assertEqual(obs_crop_nonsquare_rgb.dtype, rgb_10x10.dtype)

    # ------------------------------------------------------------------
    # Reversed point order — same result
    # ------------------------------------------------------------------

    def test_reversed_points_rgb(self):
        """Swapping pt1 and pt2 should produce the same crop (min/max used)."""
        obs_reversed = crop_image(rgb_10x10, pt1_wh=pt2_wh, pt2_wh=pt1_wh)
        np.testing.assert_array_equal(obs_reversed, exp_crop_rgb)

    def test_reversed_points_grey(self):
        obs_reversed = crop_image(grey_10x10, pt1_wh=pt2_wh, pt2_wh=pt1_wh)
        np.testing.assert_array_equal(obs_reversed, exp_crop_grey)

    # ------------------------------------------------------------------
    # List input accepted (not just tuples)
    # ------------------------------------------------------------------

    def test_list_args_accepted(self):
        np.testing.assert_array_equal(obs_crop_list, exp_crop_rgb)


if __name__ == "__main__":
    unittest.main()

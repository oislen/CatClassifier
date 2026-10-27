import os
import sys
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

with unittest.mock.patch("beartype.beartype", new=lambda f, **_kw: f):
    from arch.greyscale_image import greyscale_image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Synthetic 4x4 RGB image with known channel values
rgb_4x4 = np.zeros((4, 4, 3), dtype=np.uint8)
rgb_4x4[:, :, 0] = 200  # red
rgb_4x4[:, :, 1] = 100  # green
rgb_4x4[:, :, 2] = 50  # blue

# Expected greyscale value (weights used by the function: R*0.2125 + G*0.7154 + B*0.0721)
# 200*0.2125 + 100*0.7154 + 50*0.0721 = 42.5 + 71.54 + 3.605 = 117.645 -> 117 (uint8 truncation)
exp_pixel_value = int(200 * 0.2125 + 100 * 0.7154 + 50 * 0.0721)

obs_grey_3dim = greyscale_image(rgb_4x4, keep_3dim=True)
obs_grey_2dim = greyscale_image(rgb_4x4, keep_3dim=False)

# Pure-red image: grey = R*0.2125
red_only = np.zeros((3, 3, 3), dtype=np.uint8)
red_only[:, :, 0] = 255
exp_red_grey = int(255 * 0.2125)
obs_red_grey = greyscale_image(red_only, keep_3dim=False)

# Pure-green image: grey = G*0.7154
green_only = np.zeros((3, 3, 3), dtype=np.uint8)
green_only[:, :, 1] = 255
exp_green_grey = int(255 * 0.7154)
obs_green_grey = greyscale_image(green_only, keep_3dim=False)

# Pure-blue image: grey = B*0.0721
blue_only = np.zeros((3, 3, 3), dtype=np.uint8)
blue_only[:, :, 2] = 255
exp_blue_grey = int(255 * 0.0721)
obs_blue_grey = greyscale_image(blue_only, keep_3dim=False)

# Black image -> all zeros
black = np.zeros((4, 4, 3), dtype=np.uint8)
obs_black_grey = greyscale_image(black, keep_3dim=False)

# White image -> all 255 (weights sum to 1.0 => floor(255) = 255)
white = np.ones((4, 4, 3), dtype=np.uint8) * 255
exp_white_grey = int(255 * (0.2125 + 0.7154 + 0.0721))
obs_white_grey = greyscale_image(white, keep_3dim=False)

# Non-square input: 3-tall x 6-wide
rgb_3x6 = np.zeros((3, 6, 3), dtype=np.uint8)
rgb_3x6[:, :, 1] = 128
obs_grey_3x6 = greyscale_image(rgb_3x6, keep_3dim=True)


class TestGreyscaleImage(unittest.TestCase):
    """
    Unit tests for greyscale_image.

    Verifies output shape, dtype, pixel values, the keep_3dim flag,
    pure-channel luminance weights, and edge-case inputs (black, white,
    non-square).
    """

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_type_3dim(self):
        self.assertIsInstance(obs_grey_3dim, np.ndarray)

    def test_type_2dim(self):
        self.assertIsInstance(obs_grey_2dim, np.ndarray)

    # ------------------------------------------------------------------
    # Shape tests
    # ------------------------------------------------------------------

    def test_shape_keep_3dim_true(self):
        """keep_3dim=True should produce shape (H, W, 1)."""
        self.assertEqual(obs_grey_3dim.shape, (4, 4, 1))

    def test_shape_keep_3dim_false(self):
        """keep_3dim=False should produce shape (H, W)."""
        self.assertEqual(obs_grey_2dim.shape, (4, 4))

    def test_ndim_keep_3dim_true(self):
        self.assertEqual(obs_grey_3dim.ndim, 3)

    def test_ndim_keep_3dim_false(self):
        self.assertEqual(obs_grey_2dim.ndim, 2)

    def test_shape_nonsquare(self):
        """Non-square input should yield (3, 6, 1) with keep_3dim=True."""
        self.assertEqual(obs_grey_3x6.shape, (3, 6, 1))

    # ------------------------------------------------------------------
    # dtype tests
    # ------------------------------------------------------------------

    def test_dtype_is_uint8_3dim(self):
        self.assertEqual(obs_grey_3dim.dtype, np.uint8)

    def test_dtype_is_uint8_2dim(self):
        self.assertEqual(obs_grey_2dim.dtype, np.uint8)

    def test_dtype_red_grey(self):
        self.assertEqual(obs_red_grey.dtype, np.uint8)

    # ------------------------------------------------------------------
    # Pixel value tests — mixed channels
    # ------------------------------------------------------------------

    def test_pixel_value_3dim(self):
        """Every pixel in the 3-dim output should equal the expected value."""
        self.assertTrue(np.all(obs_grey_3dim[:, :, 0] == exp_pixel_value))

    def test_pixel_value_2dim(self):
        """Every pixel in the 2-dim output should equal the expected value."""
        self.assertTrue(np.all(obs_grey_2dim == exp_pixel_value))

    # ------------------------------------------------------------------
    # Pure-channel luminance weight tests
    # ------------------------------------------------------------------

    def test_red_channel_weight(self):
        """Red-only image should produce floor(255 * 0.2125) for every pixel."""
        self.assertTrue(np.all(obs_red_grey == exp_red_grey))

    def test_green_channel_weight(self):
        """Green-only image should produce floor(255 * 0.7154) for every pixel."""
        self.assertTrue(np.all(obs_green_grey == exp_green_grey))

    def test_blue_channel_weight(self):
        """Blue-only image should produce floor(255 * 0.0721) for every pixel."""
        self.assertTrue(np.all(obs_blue_grey == exp_blue_grey))

    def test_green_weight_highest(self):
        """Green luminance weight is the largest; green grey > red grey > blue grey."""
        self.assertGreater(exp_green_grey, exp_red_grey)
        self.assertGreater(exp_red_grey, exp_blue_grey)

    # ------------------------------------------------------------------
    # Edge-case pixel value tests
    # ------------------------------------------------------------------

    def test_black_image_all_zeros(self):
        self.assertTrue(np.all(obs_black_grey == 0))

    def test_white_image_expected_value(self):
        self.assertTrue(np.all(obs_white_grey == exp_white_grey))

    def test_pixel_values_in_uint8_range(self):
        """All output pixels must be in [0, 255]."""
        self.assertTrue(np.all(obs_grey_2dim >= 0))
        self.assertTrue(np.all(obs_grey_2dim <= 255))

    # ------------------------------------------------------------------
    # Channel reduction test
    # ------------------------------------------------------------------

    def test_last_channel_size_is_1(self):
        """Third dimension must be exactly 1 when keep_3dim=True."""
        self.assertEqual(obs_grey_3dim.shape[2], 1)

    def test_3dim_and_2dim_values_consistent(self):
        """Squeezing the 3-dim result should match the 2-dim result exactly."""
        np.testing.assert_array_equal(obs_grey_3dim[:, :, 0], obs_grey_2dim)

    # ------------------------------------------------------------------
    # Spatial dimensions preserved
    # ------------------------------------------------------------------

    def test_spatial_dims_preserved_3dim(self):
        h, w = rgb_4x4.shape[:2]
        self.assertEqual(obs_grey_3dim.shape[0], h)
        self.assertEqual(obs_grey_3dim.shape[1], w)

    def test_spatial_dims_preserved_2dim(self):
        h, w = rgb_4x4.shape[:2]
        self.assertEqual(obs_grey_2dim.shape[0], h)
        self.assertEqual(obs_grey_2dim.shape[1], w)

    def test_nonsquare_spatial_dims_preserved(self):
        self.assertEqual(obs_grey_3x6.shape[0], 3)
        self.assertEqual(obs_grey_3x6.shape[1], 6)


if __name__ == "__main__":
    unittest.main()

import os
import sys
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# beartype in this environment rejects np.array as a type hint at decoration
# time.  Neutralise it before importing source modules so the decorator becomes
# a transparent pass-through during tests.
with unittest.mock.patch("beartype.beartype", new=lambda f, **_kw: f):
    from arch.pad_image import pad_image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# 3-channel (RGB) image smaller than the target pad size
rgb_array_4x4 = np.zeros((4, 4, 3), dtype=np.uint8)
rgb_array_4x4[:, :, 0] = 100  # set red channel to known value

# 2-D (greyscale, no channel dim) image
grey_array_4x4 = np.zeros((4, 4), dtype=np.uint8)
grey_array_4x4[0, 0] = 255

pad_shape_wh = (8, 8)  # (width, height) -> target is 8 x 8

obs_padded_rgb = pad_image(rgb_array_4x4, pad_shape_wh=pad_shape_wh)
obs_padded_grey = pad_image(grey_array_4x4, pad_shape_wh=pad_shape_wh)

# no-op: image already matches target
obs_padded_exact = pad_image(
    np.zeros((8, 8, 3), dtype=np.uint8), pad_shape_wh=pad_shape_wh
)

# Asymmetric padding: 3-wide x 2-tall image -> 10-wide x 7-tall target
rgb_3x2 = np.ones((2, 3, 3), dtype=np.uint8) * 50
obs_padded_asymmetric = pad_image(rgb_3x2, pad_shape_wh=(10, 7))

# Non-square input: 6-tall x 3-wide -> 8x8
rgb_6x3 = np.ones((6, 3, 3), dtype=np.uint8) * 77
obs_padded_nonsquare = pad_image(rgb_6x3, pad_shape_wh=(8, 8))

# float32 dtype
rgb_float = np.ones((4, 4, 3), dtype=np.float32) * 0.5
obs_padded_float = pad_image(rgb_float, pad_shape_wh=(6, 6))

# 4-channel image (RGBA)
rgba_4x4 = np.ones((4, 4, 4), dtype=np.uint8) * 200
obs_padded_rgba = pad_image(rgba_4x4, pad_shape_wh=(8, 8))


class TestPadImage(unittest.TestCase):
    """
    Unit tests for pad_image.

    Verifies that the function pads both 3-D (RGB) and 2-D (greyscale)
    numpy image arrays to the requested (width, height) target, preserving
    the original pixel values and adding zero-padding.
    """

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_type_rgb(self):
        self.assertIsInstance(obs_padded_rgb, np.ndarray)

    def test_type_grey(self):
        self.assertIsInstance(obs_padded_grey, np.ndarray)

    def test_type_float(self):
        self.assertIsInstance(obs_padded_float, np.ndarray)

    # ------------------------------------------------------------------
    # Shape tests
    # ------------------------------------------------------------------

    def test_shape_rgb(self):
        """Output shape should be (pad_h, pad_w, channels)."""
        self.assertEqual(obs_padded_rgb.shape, (8, 8, 3))

    def test_shape_grey(self):
        """Output shape should be (pad_h, pad_w) for 2-D input."""
        self.assertEqual(obs_padded_grey.shape, (8, 8))

    def test_shape_exact_match(self):
        """No-op when image already equals target size."""
        self.assertEqual(obs_padded_exact.shape, (8, 8, 3))

    def test_shape_asymmetric(self):
        """Asymmetric padding: 3-wide x 2-tall -> 10-wide x 7-tall."""
        self.assertEqual(obs_padded_asymmetric.shape, (7, 10, 3))

    def test_shape_nonsquare_input(self):
        """Non-square input padded to square target."""
        self.assertEqual(obs_padded_nonsquare.shape, (8, 8, 3))

    def test_shape_float(self):
        self.assertEqual(obs_padded_float.shape, (6, 6, 3))

    def test_shape_rgba(self):
        """4-channel (RGBA) input should preserve all 4 channels."""
        self.assertEqual(obs_padded_rgba.shape, (8, 8, 4))

    # ------------------------------------------------------------------
    # Dimension tests
    # ------------------------------------------------------------------

    def test_ndim_rgb(self):
        self.assertEqual(obs_padded_rgb.ndim, 3)

    def test_ndim_grey(self):
        self.assertEqual(obs_padded_grey.ndim, 2)

    def test_ndim_asymmetric(self):
        self.assertEqual(obs_padded_asymmetric.ndim, 3)

    # ------------------------------------------------------------------
    # Channel count preserved
    # ------------------------------------------------------------------

    def test_channel_count_rgb(self):
        self.assertEqual(obs_padded_rgb.shape[2], 3)

    def test_channel_count_rgba(self):
        self.assertEqual(obs_padded_rgba.shape[2], 4)

    # ------------------------------------------------------------------
    # Value preservation tests (original pixels unchanged)
    # ------------------------------------------------------------------

    def test_original_pixels_preserved_rgb(self):
        """Pixels in the original region must be unchanged."""
        np.testing.assert_array_equal(obs_padded_rgb[:4, :4, :], rgb_array_4x4)

    def test_original_pixels_preserved_grey(self):
        np.testing.assert_array_equal(obs_padded_grey[:4, :4], grey_array_4x4)

    def test_original_pixels_preserved_asymmetric(self):
        """Original 2x3 region must be unchanged inside the padded array."""
        np.testing.assert_array_equal(obs_padded_asymmetric[:2, :3, :], rgb_3x2)

    def test_original_pixels_preserved_nonsquare(self):
        np.testing.assert_array_equal(obs_padded_nonsquare[:6, :3, :], rgb_6x3)

    # ------------------------------------------------------------------
    # Padding value tests (zero-padded)
    # ------------------------------------------------------------------

    def test_padding_is_zero_rgb(self):
        """Padded region must be all zeros."""
        np.testing.assert_array_equal(obs_padded_rgb[4:, :, :], 0)
        np.testing.assert_array_equal(obs_padded_rgb[:, 4:, :], 0)

    def test_padding_is_zero_grey(self):
        np.testing.assert_array_equal(obs_padded_grey[4:, :], 0)
        np.testing.assert_array_equal(obs_padded_grey[:, 4:], 0)

    def test_padding_is_zero_asymmetric_rows(self):
        """Bottom-padded rows must all be zero."""
        np.testing.assert_array_equal(obs_padded_asymmetric[2:, :, :], 0)

    def test_padding_is_zero_asymmetric_cols(self):
        """Right-padded columns must all be zero."""
        np.testing.assert_array_equal(obs_padded_asymmetric[:, 3:, :], 0)

    def test_padding_is_zero_float(self):
        np.testing.assert_array_equal(obs_padded_float[4:, :, :], 0.0)
        np.testing.assert_array_equal(obs_padded_float[:, 4:, :], 0.0)

    # ------------------------------------------------------------------
    # dtype preservation
    # ------------------------------------------------------------------

    def test_dtype_preserved_rgb(self):
        self.assertEqual(obs_padded_rgb.dtype, rgb_array_4x4.dtype)

    def test_dtype_preserved_grey(self):
        self.assertEqual(obs_padded_grey.dtype, grey_array_4x4.dtype)

    def test_dtype_preserved_float32(self):
        self.assertEqual(obs_padded_float.dtype, np.float32)

    def test_dtype_preserved_rgba(self):
        self.assertEqual(obs_padded_rgba.dtype, rgba_4x4.dtype)

    # ------------------------------------------------------------------
    # Total pixel count tests
    # ------------------------------------------------------------------

    def test_total_pixel_count_rgb(self):
        self.assertEqual(obs_padded_rgb.size, 8 * 8 * 3)

    def test_total_pixel_count_grey(self):
        self.assertEqual(obs_padded_grey.size, 8 * 8)

    # ------------------------------------------------------------------
    # No-op value test (pixel values unchanged when already at target)
    # ------------------------------------------------------------------

    def test_noop_pixel_values_unchanged(self):
        src = np.arange(192, dtype=np.uint8).reshape((8, 8, 3))
        result = pad_image(src, pad_shape_wh=(8, 8))
        np.testing.assert_array_equal(result, src)


if __name__ == "__main__":
    unittest.main()

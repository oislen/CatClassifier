import glob
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from torch_models.ModelFit import ModelFit
from utilities.plot_model import plot_model_fit

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_model_fit = ModelFit(
    loss=[0.9, 0.7, 0.5],
    accuracy=[0.55, 0.70, 0.85],
    val_loss=[1.0, 0.8, 0.6],
    val_accuracy=[0.50, 0.65, 0.80],
)


class TestPlotModelFit(unittest.TestCase):
    """
    Unit tests for plot_model_fit().

    Verifies that the correct output filenames are created so that the
    mlflow.log_artifact calls in prg_torch_model.py remain correct if
    the filenames are ever changed.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # File creation tests
    # ------------------------------------------------------------------

    def test_plot_model_fit_creates_accuracy_file(self):
        plot_model_fit(_model_fit, output_fdir=self.tmp_dir, show_plot=False)
        expected = os.path.join(self.tmp_dir, "model_accuracy.png")
        self.assertTrue(os.path.exists(expected), msg=f"Expected {expected} to exist")

    def test_plot_model_fit_creates_loss_file(self):
        plot_model_fit(_model_fit, output_fdir=self.tmp_dir, show_plot=False)
        expected = os.path.join(self.tmp_dir, "model_loss.png")
        self.assertTrue(os.path.exists(expected), msg=f"Expected {expected} to exist")

    def test_plot_model_fit_creates_exactly_two_files(self):
        """Exactly 2 .png files should be created — no unexpected extras.

        tearDown uses shutil.rmtree to guarantee a clean temp directory,
        so leftover files from previous tests cannot cause false failures.
        """
        plot_model_fit(_model_fit, output_fdir=self.tmp_dir, show_plot=False)
        png_files = glob.glob(os.path.join(self.tmp_dir, "*.png"))
        self.assertEqual(
            len(png_files), 2, msg=f"Expected 2 .png files, found: {png_files}"
        )

    # ------------------------------------------------------------------
    # No-output tests
    # ------------------------------------------------------------------

    def test_plot_model_fit_no_output_when_fdir_is_none(self):
        """When output_fdir=None, no files are written and no exception raised."""
        # Should not raise
        plot_model_fit(_model_fit, output_fdir=None, show_plot=False)
        # Neither expected file should exist anywhere in the temp dir
        accuracy_path = os.path.join(self.tmp_dir, "model_accuracy.png")
        loss_path = os.path.join(self.tmp_dir, "model_loss.png")
        self.assertFalse(os.path.exists(accuracy_path))
        self.assertFalse(os.path.exists(loss_path))


if __name__ == "__main__":
    unittest.main()

import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from torch_models.export_onnx import export_onnx

# ---------------------------------------------------------------------------
# Minimal model used across all tests
# ---------------------------------------------------------------------------


class _TinyNet(nn.Module):
    """Minimal two-layer network for export_onnx testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc2(torch.relu(self.fc1(x)))


def _make_model():
    torch.manual_seed(0)
    return _TinyNet()


IMAGE_WIDTH = 8
IMAGE_HEIGHT = 8
NUM_CLASSES = 2


class TestExportOnnx(unittest.TestCase):
    """
    Unit tests for the export_onnx() utility.
    """

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        self._tmp.close()
        self.filepath = self._tmp.name
        self.model = _make_model()

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    # ------------------------------------------------------------------
    # Return value tests
    # ------------------------------------------------------------------

    def test_export_onnx_returns_string(self):
        msg = export_onnx(self.model, self.filepath, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.assertIsInstance(msg, str)

    def test_export_onnx_message_contains_filepath(self):
        msg = export_onnx(self.model, self.filepath, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.assertIn(self.filepath, msg)

    # ------------------------------------------------------------------
    # File tests
    # ------------------------------------------------------------------

    def test_export_onnx_creates_file(self):
        export_onnx(self.model, self.filepath, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.assertTrue(os.path.exists(self.filepath))

    def test_export_onnx_file_is_nonempty(self):
        export_onnx(self.model, self.filepath, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.assertGreater(os.path.getsize(self.filepath), 0)

    def test_export_onnx_file_is_valid_onnx(self):
        """File produced by export_onnx must be loadable by onnx.load()."""
        import onnx

        export_onnx(self.model, self.filepath, IMAGE_WIDTH, IMAGE_HEIGHT)
        model_proto = onnx.load(self.filepath)
        onnx.checker.check_model(model_proto)

    def test_export_onnx_overwrites_existing_file(self):
        """Calling export_onnx twice must silently overwrite without raising."""
        export_onnx(self.model, self.filepath, IMAGE_WIDTH, IMAGE_HEIGHT)
        size1 = os.path.getsize(self.filepath)
        export_onnx(self.model, self.filepath, IMAGE_WIDTH, IMAGE_HEIGHT)
        size2 = os.path.getsize(self.filepath)
        self.assertEqual(size1, size2)

    # ------------------------------------------------------------------
    # Output shape tests (via onnxruntime)
    # ------------------------------------------------------------------

    def test_export_onnx_output_shape(self):
        """ONNX model run on batch=1 input must produce shape (1, num_classes)."""
        import numpy as np
        import onnxruntime as ort

        export_onnx(self.model, self.filepath, IMAGE_WIDTH, IMAGE_HEIGHT)
        sess = ort.InferenceSession(self.filepath)
        dummy = np.random.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.float32)
        out = sess.run(None, {"input": dummy})
        self.assertEqual(out[0].shape, (1, NUM_CLASSES))

    def test_export_onnx_dynamic_batch_axis(self):
        """ONNX model must accept batch sizes other than 1 (dynamic batch axis)."""
        import numpy as np
        import onnxruntime as ort

        export_onnx(self.model, self.filepath, IMAGE_WIDTH, IMAGE_HEIGHT)
        sess = ort.InferenceSession(self.filepath)
        dummy = np.random.randn(4, 3, IMAGE_HEIGHT, IMAGE_WIDTH).astype(np.float32)
        out = sess.run(None, {"input": dummy})
        self.assertEqual(out[0].shape, (4, NUM_CLASSES))


if __name__ == "__main__":
    unittest.main()

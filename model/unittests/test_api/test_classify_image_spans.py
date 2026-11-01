"""
Unit tests for classify_image() span instrumentation in model/api.py.

Tests use unittest.mock to capture mlflow.start_span calls without
requiring a running MLflow server or real model weights.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, call, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Helpers — track which span names were opened
# ---------------------------------------------------------------------------

_OPENED_SPANS = []


def _make_span_mock(name, **kwargs):
    """Return a context manager mock that records the span name."""
    _OPENED_SPANS.append(name)
    span = MagicMock()
    span.__enter__ = MagicMock(return_value=span)
    span.__exit__ = MagicMock(return_value=False)
    return span


class TestClassifyImageSpans(unittest.TestCase):
    """
    Verifies that classify_image() opens the expected MLflow child spans.

    All heavy dependencies (model.load, TorchLoadImages, DataLoader, model.predict,
    mlflow.start_span) are mocked so the tests run without real weights or a server.
    """

    def _run_classify_image(self):
        """Patch dependencies and call classify_image(), returning the opened span names."""
        _OPENED_SPANS.clear()

        import numpy as np
        import pandas as pd

        # build a minimal test_df that classify_image expects
        fake_df = pd.DataFrame([{
            "filepaths": "/tmp/test.jpg",
            "filenames": "test.jpg",
            "images": None,
            "image_tensors": None,
            "ndims": 3,
            "torch_transform_error": False,
        }])

        fake_predict = np.array([[0.3, 0.7]])

        with patch("api.mlflow.start_span", side_effect=_make_span_mock):
            with patch("api.TorchLoadImages") as mock_loader_cls:
                mock_loader = MagicMock()
                mock_loader.load_images.return_value = fake_df.to_dict(orient="records")
                mock_loader_cls.return_value = mock_loader

                with patch("api.CustomDataset"):
                    with patch("api.DataLoader") as mock_dl_cls:
                        mock_dl = MagicMock()
                        mock_dl.__len__ = MagicMock(return_value=1)
                        mock_dl.__iter__ = MagicMock(return_value=iter([]))
                        mock_dl_cls.return_value = mock_dl

                        with patch("api.model_dict") as mock_model_dict:
                            mock_model = MagicMock()
                            mock_model.predict.return_value = fake_predict
                            mock_model_dict.__getitem__ = MagicMock(return_value=mock_model)
                            mock_model.to.return_value = mock_model

                            import api
                            api.classify_image(
                                image_filepath="/tmp/test.jpg",
                                model_id="alexnet_8_pretrained",
                                device_type="cpu",
                                original_filename="test.jpg",
                            )

        return list(_OPENED_SPANS)

    # ------------------------------------------------------------------
    # Individual span tests
    # ------------------------------------------------------------------

    def test_classify_image_creates_model_load_span(self):
        spans = self._run_classify_image()
        self.assertIn("model_load", spans)

    def test_classify_image_creates_image_loading_span(self):
        spans = self._run_classify_image()
        self.assertIn("image_loading", spans)

    def test_classify_image_creates_dataloader_span(self):
        spans = self._run_classify_image()
        self.assertIn("dataloader_creation", spans)

    def test_classify_image_creates_prediction_span(self):
        spans = self._run_classify_image()
        self.assertIn("prediction", spans)

    def test_classify_image_span_count(self):
        spans = self._run_classify_image()
        self.assertEqual(len(spans), 4, msg=f"Expected 4 spans, got: {spans}")

    def test_classify_image_loading_span_includes_original_filename(self):
        """The image_loading span must have original_filename in its set_inputs call."""
        _OPENED_SPANS.clear()

        import numpy as np
        import pandas as pd

        fake_df = pd.DataFrame([{
            "filepaths": "/tmp/test.jpg",
            "filenames": "test.jpg",
            "images": None,
            "image_tensors": None,
            "ndims": 3,
            "torch_transform_error": False,
        }])
        fake_predict = np.array([[0.3, 0.7]])

        captured_inputs = {}

        def span_side_effect(name, **kwargs):
            span = MagicMock()

            def set_inputs(inputs_dict):
                if name == "image_loading":
                    captured_inputs.update(inputs_dict)

            span.set_inputs = MagicMock(side_effect=set_inputs)
            span.set_outputs = MagicMock()
            span.set_attributes = MagicMock()
            span.__enter__ = MagicMock(return_value=span)
            span.__exit__ = MagicMock(return_value=False)
            return span

        with patch("api.mlflow.start_span", side_effect=span_side_effect):
            with patch("api.TorchLoadImages") as mock_loader_cls:
                mock_loader = MagicMock()
                mock_loader.load_images.return_value = fake_df.to_dict(orient="records")
                mock_loader_cls.return_value = mock_loader
                with patch("api.CustomDataset"):
                    with patch("api.DataLoader") as mock_dl_cls:
                        mock_dl = MagicMock()
                        mock_dl.__len__ = MagicMock(return_value=1)
                        mock_dl_cls.return_value = mock_dl
                        with patch("api.model_dict") as mock_model_dict:
                            mock_model = MagicMock()
                            mock_model.predict.return_value = fake_predict
                            mock_model_dict.__getitem__ = MagicMock(return_value=mock_model)
                            mock_model.to.return_value = mock_model
                            import api
                            api.classify_image(
                                image_filepath="/tmp/test.jpg",
                                model_id="alexnet_8_pretrained",
                                device_type="cpu",
                                original_filename="uploaded.jpg",
                            )

        self.assertIn(
            "original_filename", captured_inputs,
            msg=f"original_filename not found in image_loading span inputs: {captured_inputs}",
        )


if __name__ == "__main__":
    unittest.main()

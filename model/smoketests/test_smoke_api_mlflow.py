"""
Smoke tests for MLflow tracing in the Flask API (model/api.py).

Uses Flask's built-in test client, a file-based MLflow tracking store,
and mocks model.load() so no .pt file is required on disk.

Run manually from CatClassifier/model/:
    uv run python -m unittest smoketests.test_smoke_api_mlflow
"""

import io
import os
import platform
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cons
import mlflow

# ---------------------------------------------------------------------------
# Minimal synthetic JPEG bytes (1×1 white pixel)
# ---------------------------------------------------------------------------

_MINIMAL_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"!$%&&\xdc\x00\x00\x00\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
    b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
    b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x00\x00\x00\x00\xff\xd9"
)


class _ApiSmokeBase(unittest.TestCase):
    """
    Sets up an isolated file-based MLflow tracking store, configures the api
    module to use it, and provides a Flask test client with model.load mocked.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp()
        cls._tracking_uri = f"file://{cls._tmpdir}"

        # point the api module at our isolated store before importing
        mlflow.set_tracking_uri(cls._tracking_uri)
        mlflow.set_experiment(cons.mlflow_experiment_name)

        # patch model.load on all models in model_dict so no .pt file is needed
        import api
        for model_obj in api.model_dict.values():
            model_obj.load = MagicMock(return_value=None)

        # also patch the tracking URI in the api module itself
        api._client = mlflow.tracking.MlflowClient()
        mlflow.set_tracking_uri(cls._tracking_uri)

        cls._app = api.app
        cls._app.config["TESTING"] = True
        cls._client_flask = cls._app.test_client()

        # make a successful request to populate traces
        cls._make_request()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    @classmethod
    def _make_request(cls, model_id="alexnet_8_pretrained"):
        """POST a synthetic JPEG to /catclassifier and return the response."""
        data = {
            "image": (io.BytesIO(_MINIMAL_JPEG), "test.jpg"),
            "model_id": model_id,
            "device_type": "cpu",
        }
        return cls._client_flask.post(
            "/catclassifier",
            data=data,
            content_type="multipart/form-data",
            headers={"X-Request-ID": "smoke-test-001"},
        )

    def _get_traces(self):
        client = mlflow.tracking.MlflowClient()
        exp = mlflow.get_experiment_by_name(cons.mlflow_experiment_name)
        if exp is None:
            return []
        return client.search_traces(experiment_ids=[exp.experiment_id])


class TestSmokeApiTraces(_ApiSmokeBase):

    def test_smoke_api_trace_created(self):
        traces = self._get_traces()
        self.assertGreater(len(traces), 0, "No traces found after API request")

    def test_smoke_api_root_span_name(self):
        traces = self._get_traces()
        found = any(
            any(s.name == "api_request" for s in t.data.spans)
            for t in traces
        )
        self.assertTrue(found, "No span named 'api_request' found in traces")

    def test_smoke_api_child_spans_present(self):
        traces = self._get_traces()
        expected = {"model_load", "image_loading", "dataloader_creation", "prediction"}
        found_names = {
            s.name
            for t in traces
            for s in t.data.spans
        }
        missing = expected - found_names
        self.assertFalse(missing, f"Missing child spans: {missing}")

    def test_smoke_api_prediction_span_has_outputs(self):
        traces = self._get_traces()
        span = next(
            (s for t in traces for s in t.data.spans if s.name == "prediction"),
            None,
        )
        self.assertIsNotNone(span, "prediction span not found")
        self.assertIn(
            "category_name", span.outputs,
            msg=f"category_name not in prediction span outputs: {span.outputs}",
        )

    def test_smoke_api_root_span_has_inputs(self):
        traces = self._get_traces()
        span = next(
            (s for t in traces for s in t.data.spans if s.name == "api_request"),
            None,
        )
        self.assertIsNotNone(span, "api_request span not found")
        for key in ("model_id", "image_filename", "request_id"):
            self.assertIn(
                key, span.inputs,
                msg=f"Key '{key}' not found in api_request span inputs: {span.inputs}",
            )

    def test_smoke_api_root_span_has_full_result_outputs(self):
        traces = self._get_traces()
        span = next(
            (s for t in traces for s in t.data.spans if s.name == "api_request"),
            None,
        )
        self.assertIsNotNone(span, "api_request span not found")
        for key in ("status_code", "category_name", "probability"):
            self.assertIn(
                key, span.outputs,
                msg=f"Key '{key}' not found in api_request span outputs: {span.outputs}",
            )


class TestSmokeApiErrorPath(_ApiSmokeBase):

    def test_smoke_api_error_request_traced(self):
        """
        A POST with an invalid model_id should:
        - return HTTP 400
        - produce a trace with OK status (early return, not exception)
        - have outputs containing status_code: 400
        """
        before_traces = self._get_traces()
        before_count = len(before_traces)

        # send a request with a bad model_id
        resp = self._make_request(model_id="nonexistent_model_xyz")
        self.assertEqual(resp.status_code, 400)

        # give MLflow a moment to flush the trace
        time.sleep(1)

        after_traces = self._get_traces()
        self.assertGreater(
            len(after_traces), before_count,
            "No new trace recorded for the error request",
        )

        # find the most recent api_request span
        latest_traces = sorted(
            after_traces,
            key=lambda t: t.info.timestamp_ms,
            reverse=True,
        )
        error_span = next(
            (s for t in latest_traces for s in t.data.spans if s.name == "api_request"),
            None,
        )
        self.assertIsNotNone(error_span, "No api_request span found for error request")
        self.assertEqual(
            error_span.outputs.get("status_code"), 400,
            msg=f"Expected status_code 400 in outputs, got: {error_span.outputs}",
        )


if __name__ == "__main__":
    unittest.main()

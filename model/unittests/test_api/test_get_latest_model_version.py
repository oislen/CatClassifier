"""
Unit tests for _get_latest_model_version() in model/api.py.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import mlflow

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestGetLatestModelVersion(unittest.TestCase):
    """
    Covers _get_latest_model_version() using unittest.mock.patch on MlflowClient.
    The function is accessed from the api module.
    """

    def _call(self, model_id):
        import api
        return api._get_latest_model_version(model_id)

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_returns_string(self):
        result = self._call("alexnet_8_pretrained")
        self.assertIsInstance(result, str)

    # ------------------------------------------------------------------
    # Happy-path tests
    # ------------------------------------------------------------------

    def test_returns_version_when_model_exists(self):
        """Returns the version string when the registered model has versions."""
        mock_version = MagicMock()
        mock_version.version = "3"
        mock_rm = MagicMock()
        mock_rm.latest_versions = [mock_version]

        import api
        with patch.object(api._client, "get_registered_model", return_value=mock_rm):
            result = self._call("alexnet_8_pretrained")
        self.assertEqual(result, "3")

    # ------------------------------------------------------------------
    # Error-path tests
    # ------------------------------------------------------------------

    def test_returns_unknown_when_model_not_found(self):
        """MlflowException from get_registered_model must return 'unknown'."""
        import api
        with patch.object(
            api._client,
            "get_registered_model",
            side_effect=mlflow.exceptions.MlflowException("not found"),
        ):
            result = self._call("alexnet_8_pretrained")
        self.assertEqual(result, "unknown")

    def test_returns_unknown_when_no_versions(self):
        """Returns 'unknown' when latest_versions is an empty list."""
        mock_rm = MagicMock()
        mock_rm.latest_versions = []

        import api
        with patch.object(api._client, "get_registered_model", return_value=mock_rm):
            result = self._call("alexnet_8_pretrained")
        self.assertEqual(result, "unknown")

    def test_returns_unknown_for_unrecognised_model_id(self):
        """Returns 'unknown' when model_id is not in cons.mlflow_registered_model_names."""
        result = self._call("nonexistent_model_id")
        self.assertEqual(result, "unknown")


if __name__ == "__main__":
    unittest.main()

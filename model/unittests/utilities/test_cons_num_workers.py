import importlib
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cons as _cons_module


class TestConsNumWorkers(unittest.TestCase):
    """
    Unit tests for cons.num_workers.

    cons.num_workers is assigned at module import time from os.environ.
    Tests that verify the environment-variable override must patch os.environ
    and use importlib.reload(cons) to force re-evaluation, then restore the
    original environment in tearDown.
    """

    def _reload_cons(self):
        """Reload cons and return the fresh module object."""
        importlib.reload(_cons_module)
        return _cons_module

    # ------------------------------------------------------------------
    # Type and default value
    # ------------------------------------------------------------------

    def test_num_workers_is_int(self):
        self.assertIsInstance(_cons_module.num_workers, int)

    def test_num_workers_default_is_zero(self):
        """Safe default must be 0 when PARAM_NUM_WORKERS is not set."""
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PARAM_NUM_WORKERS", None)
            fresh = self._reload_cons()
            self.assertEqual(fresh.num_workers, 0)

    # ------------------------------------------------------------------
    # Environment variable override
    # ------------------------------------------------------------------

    def test_num_workers_env_override(self):
        """Setting PARAM_NUM_WORKERS=4 and reloading cons must yield num_workers == 4."""
        with unittest.mock.patch.dict(os.environ, {"PARAM_NUM_WORKERS": "4"}):
            fresh = self._reload_cons()
            self.assertEqual(fresh.num_workers, 4)

    def test_num_workers_env_override_zero(self):
        """Explicitly setting PARAM_NUM_WORKERS=0 must yield num_workers == 0."""
        with unittest.mock.patch.dict(os.environ, {"PARAM_NUM_WORKERS": "0"}):
            fresh = self._reload_cons()
            self.assertEqual(fresh.num_workers, 0)

    def test_num_workers_env_override_is_int(self):
        """num_workers must remain an int even when set via environment variable."""
        with unittest.mock.patch.dict(os.environ, {"PARAM_NUM_WORKERS": "2"}):
            fresh = self._reload_cons()
            self.assertIsInstance(fresh.num_workers, int)


if __name__ == "__main__":
    unittest.main()

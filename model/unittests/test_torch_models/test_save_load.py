import os
import sys
import tempfile
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from torch_models.save import save
from torch_models.load import load


# ---------------------------------------------------------------------------
# Minimal model used across all tests
# ---------------------------------------------------------------------------

class _TinyNet(nn.Module):
    """Minimal two-layer network for save/load testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def _make_model(seed=0):
    torch.manual_seed(seed)
    return _TinyNet()


class TestSave(unittest.TestCase):
    """
    Unit tests for the save() utility.

    Verifies return value, file creation, and non-empty file content.
    """

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        self._tmp.close()
        self.filepath = self._tmp.name
        self.model = _make_model()

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    # ------------------------------------------------------------------
    # Return value tests
    # ------------------------------------------------------------------

    def test_save_returns_string(self):
        msg = save(self.model, output_fpath=self.filepath)
        self.assertIsInstance(msg, str)

    def test_save_message_contains_filepath(self):
        msg = save(self.model, output_fpath=self.filepath)
        self.assertIn(self.filepath, msg)

    def test_save_message_contains_saved(self):
        msg = save(self.model, output_fpath=self.filepath)
        self.assertIn("Saved", msg)

    # ------------------------------------------------------------------
    # File tests
    # ------------------------------------------------------------------

    def test_save_creates_file(self):
        save(self.model, output_fpath=self.filepath)
        self.assertTrue(os.path.exists(self.filepath))

    def test_save_file_is_nonempty(self):
        save(self.model, output_fpath=self.filepath)
        self.assertGreater(os.path.getsize(self.filepath), 0)

    def test_save_overwrites_existing_file(self):
        """Saving twice should silently overwrite the first file."""
        save(self.model, output_fpath=self.filepath)
        size1 = os.path.getsize(self.filepath)
        save(self.model, output_fpath=self.filepath)
        size2 = os.path.getsize(self.filepath)
        # Same model so size should be identical
        self.assertEqual(size1, size2)

    # ------------------------------------------------------------------
    # State dict tests
    # ------------------------------------------------------------------

    def test_saved_file_loads_as_state_dict(self):
        """File written by save() must be loadable as a state dict."""
        save(self.model, output_fpath=self.filepath)
        state = torch.load(self.filepath, weights_only=True)
        self.assertIsInstance(state, dict)

    def test_state_dict_keys_match(self):
        save(self.model, output_fpath=self.filepath)
        state = torch.load(self.filepath, weights_only=True)
        self.assertEqual(set(state.keys()), set(self.model.state_dict().keys()))


class TestLoad(unittest.TestCase):
    """
    Unit tests for the load() utility.

    Verifies return value and that model weights are correctly restored.
    """

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        self._tmp.close()
        self.filepath = self._tmp.name
        self.model = _make_model(seed=1)
        # Save reference weights so load() has something to read
        torch.save(self.model.state_dict(), self.filepath)

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    # ------------------------------------------------------------------
    # Return value tests
    # ------------------------------------------------------------------

    def test_load_returns_string(self):
        msg = load(self.model, input_fpath=self.filepath)
        self.assertIsInstance(msg, str)

    def test_load_message_contains_filepath(self):
        msg = load(self.model, input_fpath=self.filepath)
        self.assertIn(self.filepath, msg)

    def test_load_message_contains_loaded(self):
        msg = load(self.model, input_fpath=self.filepath)
        self.assertIn("Loaded", msg)

    # ------------------------------------------------------------------
    # Eval mode test
    # ------------------------------------------------------------------

    def test_model_in_eval_mode_after_load(self):
        load(self.model, input_fpath=self.filepath)
        self.assertFalse(self.model.training)

    # ------------------------------------------------------------------
    # Weight round-trip tests
    # ------------------------------------------------------------------

    def test_weights_restored_correctly(self):
        """After load, fc1.weight should match what was saved."""
        original_fc1 = self.model.fc1.weight.data.clone()
        # Mutate model weights
        with torch.no_grad():
            self.model.fc1.weight.fill_(0.0)
        # Reload
        load(self.model, input_fpath=self.filepath)
        torch.testing.assert_close(self.model.fc1.weight.data, original_fc1)

    def test_fc2_weights_restored(self):
        original_fc2 = self.model.fc2.weight.data.clone()
        with torch.no_grad():
            self.model.fc2.weight.fill_(0.0)
        load(self.model, input_fpath=self.filepath)
        torch.testing.assert_close(self.model.fc2.weight.data, original_fc2)


class TestSaveLoadRoundTrip(unittest.TestCase):
    """
    Integration tests verifying save → load round-trip correctness.
    """

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        self._tmp.close()
        self.filepath = self._tmp.name

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    def test_roundtrip_fc1_weight(self):
        model_src = _make_model(seed=7)
        model_dst = _make_model(seed=99)   # different random init
        save(model_src, output_fpath=self.filepath)
        load(model_dst, input_fpath=self.filepath)
        torch.testing.assert_close(
            model_dst.fc1.weight.data, model_src.fc1.weight.data
        )

    def test_roundtrip_fc2_weight(self):
        model_src = _make_model(seed=7)
        model_dst = _make_model(seed=99)
        save(model_src, output_fpath=self.filepath)
        load(model_dst, input_fpath=self.filepath)
        torch.testing.assert_close(
            model_dst.fc2.weight.data, model_src.fc2.weight.data
        )

    def test_roundtrip_fc1_bias(self):
        model_src = _make_model(seed=7)
        model_dst = _make_model(seed=99)
        save(model_src, output_fpath=self.filepath)
        load(model_dst, input_fpath=self.filepath)
        torch.testing.assert_close(
            model_dst.fc1.bias.data, model_src.fc1.bias.data
        )

    def test_roundtrip_predictions_identical(self):
        """After save/load, both models must produce identical output."""
        torch.manual_seed(0)
        x = torch.randn(3, 4)
        model_src = _make_model(seed=7)
        model_dst = _make_model(seed=99)
        save(model_src, output_fpath=self.filepath)
        load(model_dst, input_fpath=self.filepath)
        model_src.eval()
        with torch.no_grad():
            out_src = model_src(x)
            out_dst = model_dst(x)
        torch.testing.assert_close(out_dst, out_src)

    def test_roundtrip_model_in_eval_mode(self):
        model_src = _make_model(seed=3)
        model_dst = _make_model(seed=5)
        save(model_src, output_fpath=self.filepath)
        load(model_dst, input_fpath=self.filepath)
        self.assertFalse(model_dst.training)


if __name__ == "__main__":
    unittest.main()

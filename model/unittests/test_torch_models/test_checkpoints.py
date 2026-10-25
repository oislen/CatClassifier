import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from torch_models.checkpoints import load_checkpoint, save_checkpoint

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

exp_checkpoint = {
    "epoch": 3,
    "loss": 0.42,
    "model_state": {"weight": torch.tensor([1.0, 2.0, 3.0])},
}

# Checkpoint with nested tensors and a list value
exp_checkpoint_complex = {
    "epoch": 7,
    "losses": [0.9, 0.7, 0.5],
    "accuracy": 0.85,
    "weights": {
        "layer1": torch.rand(4, 4),
        "layer2": torch.rand(2, 4),
    },
}

# Minimal checkpoint (scalars only)
exp_checkpoint_minimal = {"epoch": 0, "loss": 9.99}


class TestCheckpoints(unittest.TestCase):
    """
    Unit tests for save_checkpoint and load_checkpoint.

    Uses temporary files to avoid test artefacts. Verifies the round-trip:
    saved data is faithfully restored, including nested tensors, list values,
    overwriting an existing file, and a minimal scalar-only checkpoint.
    """

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        self._tmp.close()
        self.filepath = self._tmp.name

    def tearDown(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    # ------------------------------------------------------------------
    # save_checkpoint tests
    # ------------------------------------------------------------------

    def test_save_creates_file(self):
        save_checkpoint(state=exp_checkpoint, filepath=self.filepath)
        self.assertTrue(os.path.exists(self.filepath))

    def test_save_file_is_nonempty(self):
        save_checkpoint(state=exp_checkpoint, filepath=self.filepath)
        self.assertGreater(os.path.getsize(self.filepath), 0)

    def test_save_overwrites_existing_file(self):
        """Saving twice to the same path should silently overwrite."""
        save_checkpoint(state={"epoch": 1}, filepath=self.filepath)
        save_checkpoint(state={"epoch": 2}, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertEqual(obs["epoch"], 2)

    # ------------------------------------------------------------------
    # load_checkpoint tests
    # ------------------------------------------------------------------

    def test_load_returns_dict(self):
        save_checkpoint(state=exp_checkpoint, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertIsInstance(obs, dict)

    def test_load_keys(self):
        save_checkpoint(state=exp_checkpoint, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertEqual(list(obs.keys()), list(exp_checkpoint.keys()))

    def test_load_epoch_value(self):
        save_checkpoint(state=exp_checkpoint, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertEqual(obs["epoch"], exp_checkpoint["epoch"])

    def test_load_loss_value(self):
        save_checkpoint(state=exp_checkpoint, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertAlmostEqual(obs["loss"], exp_checkpoint["loss"], places=5)

    def test_load_tensor_values(self):
        save_checkpoint(state=exp_checkpoint, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        torch.testing.assert_close(
            obs["model_state"]["weight"],
            exp_checkpoint["model_state"]["weight"],
        )

    # ------------------------------------------------------------------
    # Round-trip tests
    # ------------------------------------------------------------------

    def test_roundtrip_full_object(self):
        save_checkpoint(state=exp_checkpoint, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertEqual(obs["epoch"], exp_checkpoint["epoch"])
        self.assertAlmostEqual(obs["loss"], exp_checkpoint["loss"], places=5)
        torch.testing.assert_close(
            obs["model_state"]["weight"],
            exp_checkpoint["model_state"]["weight"],
        )

    def test_roundtrip_complex_checkpoint_epoch(self):
        """Complex checkpoint with nested tensors and list values round-trips."""
        save_checkpoint(state=exp_checkpoint_complex, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertEqual(obs["epoch"], exp_checkpoint_complex["epoch"])

    def test_roundtrip_complex_checkpoint_losses_list(self):
        save_checkpoint(state=exp_checkpoint_complex, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertEqual(obs["losses"], exp_checkpoint_complex["losses"])

    def test_roundtrip_complex_checkpoint_accuracy(self):
        save_checkpoint(state=exp_checkpoint_complex, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertAlmostEqual(obs["accuracy"], exp_checkpoint_complex["accuracy"])

    def test_roundtrip_complex_layer1_tensor(self):
        save_checkpoint(state=exp_checkpoint_complex, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        torch.testing.assert_close(
            obs["weights"]["layer1"], exp_checkpoint_complex["weights"]["layer1"]
        )

    def test_roundtrip_complex_layer2_tensor(self):
        save_checkpoint(state=exp_checkpoint_complex, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        torch.testing.assert_close(
            obs["weights"]["layer2"], exp_checkpoint_complex["weights"]["layer2"]
        )

    def test_roundtrip_minimal_checkpoint(self):
        save_checkpoint(state=exp_checkpoint_minimal, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertEqual(obs["epoch"], 0)
        self.assertAlmostEqual(obs["loss"], 9.99, places=5)

    # ------------------------------------------------------------------
    # Key count tests
    # ------------------------------------------------------------------

    def test_key_count_standard(self):
        save_checkpoint(state=exp_checkpoint, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertEqual(len(obs), len(exp_checkpoint))

    def test_key_count_minimal(self):
        save_checkpoint(state=exp_checkpoint_minimal, filepath=self.filepath)
        obs = load_checkpoint(filepath=self.filepath)
        self.assertEqual(len(obs), len(exp_checkpoint_minimal))


if __name__ == "__main__":
    unittest.main()

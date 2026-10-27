import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from torch_models.ModelFit import ModelFit

# ---------------------------------------------------------------------------
# Fixtures — standard 3-epoch training run
# ---------------------------------------------------------------------------

exp_loss = [0.9, 0.7, 0.5]
exp_accuracy = [0.55, 0.70, 0.85]
exp_val_loss = [1.0, 0.8, 0.6]
exp_val_accuracy = [0.50, 0.65, 0.80]

obs_model_fit = ModelFit(
    loss=exp_loss,
    accuracy=exp_accuracy,
    val_loss=exp_val_loss,
    val_accuracy=exp_val_accuracy,
)

exp_history_keys = ["loss", "accuracy", "val_loss", "val_accuracy"]

# Single-epoch ModelFit
single_model_fit = ModelFit(
    loss=[0.95],
    accuracy=[0.52],
    val_loss=[1.10],
    val_accuracy=[0.48],
)

# Empty-list ModelFit (edge case: no epochs recorded yet)
empty_model_fit = ModelFit(loss=[], accuracy=[], val_loss=[], val_accuracy=[])

# Longer run (10 epochs) for index-access tests
long_loss = [1.0 - i * 0.05 for i in range(10)]
long_accuracy = [0.5 + i * 0.04 for i in range(10)]
long_val_loss = [1.1 - i * 0.05 for i in range(10)]
long_val_acc = [0.45 + i * 0.04 for i in range(10)]
long_model_fit = ModelFit(
    loss=long_loss,
    accuracy=long_accuracy,
    val_loss=long_val_loss,
    val_accuracy=long_val_acc,
)


class TestModelFit(unittest.TestCase):
    """
    Unit tests for ModelFit.

    Verifies that the history dict is constructed correctly with the right
    keys, value types, lengths, exact values, and for edge cases such as
    empty lists, single-epoch runs, and multi-epoch index access.
    """

    def setUp(self):
        self.model_fit = obs_model_fit
        self.exp_history_keys = exp_history_keys

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_history_type(self):
        self.assertIsInstance(self.model_fit.history, dict)

    def test_loss_type(self):
        self.assertIsInstance(self.model_fit.history["loss"], list)

    def test_accuracy_type(self):
        self.assertIsInstance(self.model_fit.history["accuracy"], list)

    def test_val_loss_type(self):
        self.assertIsInstance(self.model_fit.history["val_loss"], list)

    def test_val_accuracy_type(self):
        self.assertIsInstance(self.model_fit.history["val_accuracy"], list)

    # ------------------------------------------------------------------
    # Key tests
    # ------------------------------------------------------------------

    def test_history_keys(self):
        self.assertEqual(list(self.model_fit.history.keys()), self.exp_history_keys)

    def test_history_key_count(self):
        self.assertEqual(len(self.model_fit.history), len(self.exp_history_keys))

    def test_history_contains_loss_key(self):
        self.assertIn("loss", self.model_fit.history)

    def test_history_contains_accuracy_key(self):
        self.assertIn("accuracy", self.model_fit.history)

    def test_history_contains_val_loss_key(self):
        self.assertIn("val_loss", self.model_fit.history)

    def test_history_contains_val_accuracy_key(self):
        self.assertIn("val_accuracy", self.model_fit.history)

    # ------------------------------------------------------------------
    # Length tests
    # ------------------------------------------------------------------

    def test_loss_length(self):
        self.assertEqual(len(self.model_fit.history["loss"]), len(exp_loss))

    def test_accuracy_length(self):
        self.assertEqual(len(self.model_fit.history["accuracy"]), len(exp_accuracy))

    def test_val_loss_length(self):
        self.assertEqual(len(self.model_fit.history["val_loss"]), len(exp_val_loss))

    def test_val_accuracy_length(self):
        self.assertEqual(
            len(self.model_fit.history["val_accuracy"]), len(exp_val_accuracy)
        )

    def test_all_series_same_length(self):
        h = self.model_fit.history
        lengths = [len(h[k]) for k in exp_history_keys]
        self.assertEqual(len(set(lengths)), 1)

    # ------------------------------------------------------------------
    # Value tests
    # ------------------------------------------------------------------

    def test_loss_values(self):
        self.assertEqual(self.model_fit.history["loss"], exp_loss)

    def test_accuracy_values(self):
        self.assertEqual(self.model_fit.history["accuracy"], exp_accuracy)

    def test_val_loss_values(self):
        self.assertEqual(self.model_fit.history["val_loss"], exp_val_loss)

    def test_val_accuracy_values(self):
        self.assertEqual(self.model_fit.history["val_accuracy"], exp_val_accuracy)

    # ------------------------------------------------------------------
    # Index access tests
    # ------------------------------------------------------------------

    def test_first_epoch_loss(self):
        self.assertAlmostEqual(self.model_fit.history["loss"][0], exp_loss[0])

    def test_last_epoch_loss(self):
        self.assertAlmostEqual(self.model_fit.history["loss"][-1], exp_loss[-1])

    def test_first_epoch_accuracy(self):
        self.assertAlmostEqual(self.model_fit.history["accuracy"][0], exp_accuracy[0])

    def test_last_epoch_val_accuracy(self):
        self.assertAlmostEqual(
            self.model_fit.history["val_accuracy"][-1], exp_val_accuracy[-1]
        )

    # ------------------------------------------------------------------
    # Object equality test
    # ------------------------------------------------------------------

    def test_history_object(self):
        exp_history = {
            "loss": exp_loss,
            "accuracy": exp_accuracy,
            "val_loss": exp_val_loss,
            "val_accuracy": exp_val_accuracy,
        }
        self.assertEqual(self.model_fit.history, exp_history)

    # ------------------------------------------------------------------
    # Single-epoch tests
    # ------------------------------------------------------------------

    def test_single_epoch_loss_length(self):
        self.assertEqual(len(single_model_fit.history["loss"]), 1)

    def test_single_epoch_loss_value(self):
        self.assertAlmostEqual(single_model_fit.history["loss"][0], 0.95)

    def test_single_epoch_keys(self):
        self.assertEqual(list(single_model_fit.history.keys()), exp_history_keys)

    # ------------------------------------------------------------------
    # Empty-list edge-case tests
    # ------------------------------------------------------------------

    def test_empty_history_type(self):
        self.assertIsInstance(empty_model_fit.history, dict)

    def test_empty_history_keys(self):
        self.assertEqual(list(empty_model_fit.history.keys()), exp_history_keys)

    def test_empty_loss_is_list(self):
        self.assertIsInstance(empty_model_fit.history["loss"], list)

    def test_empty_loss_length(self):
        self.assertEqual(len(empty_model_fit.history["loss"]), 0)

    # ------------------------------------------------------------------
    # Long-run index tests
    # ------------------------------------------------------------------

    def test_long_run_length(self):
        self.assertEqual(len(long_model_fit.history["loss"]), 10)

    def test_long_run_first_loss(self):
        self.assertAlmostEqual(long_model_fit.history["loss"][0], long_loss[0])

    def test_long_run_last_loss(self):
        self.assertAlmostEqual(long_model_fit.history["loss"][-1], long_loss[-1])

    def test_long_run_loss_decreasing(self):
        """Verify test fixture itself: loss should decrease over 10 epochs."""
        losses = long_model_fit.history["loss"]
        self.assertTrue(all(losses[i] > losses[i + 1] for i in range(len(losses) - 1)))


if __name__ == "__main__":
    unittest.main()

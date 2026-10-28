import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utilities.evaluate import compute_eval_metrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Perfect predictions
y_true_perfect = ["cat", "dog", "cat", "dog", "cat"]
y_pred_perfect = ["cat", "dog", "cat", "dog", "cat"]

# All wrong predictions
y_true_wrong = ["cat", "dog", "cat", "dog"]
y_pred_wrong = ["dog", "cat", "dog", "cat"]

# Known mixed case: 3/5 correct, tp=1, fp=1, fn=1 for 'dog' class
# true: cat dog cat dog cat
# pred: cat dog dog cat cat  -> acc=3/5, precision=0.5, recall=0.5, f1=0.5
y_true_mixed = ["cat", "dog", "cat", "dog", "cat"]
y_pred_mixed = ["cat", "dog", "dog", "cat", "cat"]


class TestComputeEvalMetrics(unittest.TestCase):
    """
    Unit tests for compute_eval_metrics().
    """

    def setUp(self):
        self.result = compute_eval_metrics(y_true_perfect, y_pred_perfect)
        self.mixed = compute_eval_metrics(y_true_mixed, y_pred_mixed)

    # ------------------------------------------------------------------
    # Type tests
    # ------------------------------------------------------------------

    def test_returns_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_metric_values_are_floats(self):
        for v in self.result.values():
            self.assertIsInstance(v, float)

    # ------------------------------------------------------------------
    # Key tests
    # ------------------------------------------------------------------

    def test_dict_contains_accuracy_key(self):
        self.assertIn("test_accuracy", self.result)

    def test_dict_contains_precision_key(self):
        self.assertIn("test_precision", self.result)

    def test_dict_contains_recall_key(self):
        self.assertIn("test_recall", self.result)

    def test_dict_contains_f1_key(self):
        self.assertIn("test_f1", self.result)

    def test_key_count(self):
        self.assertEqual(len(self.result), 4)

    # ------------------------------------------------------------------
    # Boundary value tests
    # ------------------------------------------------------------------

    def test_perfect_predictions_accuracy(self):
        self.assertAlmostEqual(self.result["test_accuracy"], 1.0)

    def test_perfect_predictions_f1(self):
        self.assertAlmostEqual(self.result["test_f1"], 1.0)

    def test_all_wrong_predictions_accuracy(self):
        result = compute_eval_metrics(y_true_wrong, y_pred_wrong)
        self.assertAlmostEqual(result["test_accuracy"], 0.0)

    # ------------------------------------------------------------------
    # Range tests
    # ------------------------------------------------------------------

    def test_metric_values_in_range(self):
        for v in self.result.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    # ------------------------------------------------------------------
    # Known mixed value tests
    # ------------------------------------------------------------------

    def test_known_mixed_accuracy(self):
        # 3 out of 5 correct
        self.assertAlmostEqual(self.mixed["test_accuracy"], 3 / 5)

    def test_known_mixed_f1(self):
        # tp=1, fp=1, fn=1 for pos_label='dog' -> precision=0.5, recall=0.5, f1=0.5
        self.assertAlmostEqual(self.mixed["test_f1"], 0.5)


if __name__ == "__main__":
    unittest.main()

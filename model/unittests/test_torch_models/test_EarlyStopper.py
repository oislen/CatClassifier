import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from torch_models.EarlyStopper import EarlyStopper


class TestEarlyStopper(unittest.TestCase):
    """
    Unit tests for EarlyStopper.

    Verifies initial state, improvement detection, counter increments,
    patience threshold, min_delta boundary, equal-loss behaviour,
    return type, and long plateau scenarios.
    """

    # ------------------------------------------------------------------
    # Initial state tests
    # ------------------------------------------------------------------

    def test_initial_counter(self):
        es = EarlyStopper(patience=2, min_delta=0)
        self.assertEqual(es.counter, 0)

    def test_initial_min_validation_loss(self):
        es = EarlyStopper(patience=2, min_delta=0)
        self.assertEqual(es.min_validation_loss, float("inf"))

    def test_initial_patience(self):
        es = EarlyStopper(patience=5, min_delta=0)
        self.assertEqual(es.patience, 5)

    def test_initial_min_delta(self):
        es = EarlyStopper(patience=1, min_delta=0.1)
        self.assertAlmostEqual(es.min_delta, 0.1)

    # ------------------------------------------------------------------
    # Return type tests
    # ------------------------------------------------------------------

    def test_early_stop_returns_bool_on_improvement(self):
        es = EarlyStopper(patience=1, min_delta=0)
        result = es.early_stop(0.5)
        self.assertIsInstance(result, bool)

    def test_early_stop_returns_bool_on_no_improvement(self):
        es = EarlyStopper(patience=2, min_delta=0)
        es.early_stop(1.0)
        result = es.early_stop(1.5)
        self.assertIsInstance(result, bool)

    def test_early_stop_returns_false_not_none(self):
        es = EarlyStopper(patience=1, min_delta=0)
        result = es.early_stop(0.5)
        self.assertFalse(result)
        self.assertIsNotNone(result)

    # ------------------------------------------------------------------
    # Improvement (loss decreases) — should NOT stop
    # ------------------------------------------------------------------

    def test_no_stop_on_first_call(self):
        """Any finite first loss is always an improvement over inf."""
        es = EarlyStopper(patience=1, min_delta=0)
        result = es.early_stop(0.5)
        self.assertFalse(result)

    def test_no_stop_on_improvement(self):
        es = EarlyStopper(patience=1, min_delta=0)
        result = es.early_stop(0.5)
        self.assertFalse(result)

    def test_min_validation_loss_updated_on_improvement(self):
        es = EarlyStopper(patience=1, min_delta=0)
        es.early_stop(0.5)
        self.assertAlmostEqual(es.min_validation_loss, 0.5)

    def test_min_validation_loss_updates_to_lower_value(self):
        es = EarlyStopper(patience=2, min_delta=0)
        es.early_stop(1.0)
        es.early_stop(0.3)
        self.assertAlmostEqual(es.min_validation_loss, 0.3)

    def test_counter_reset_on_improvement(self):
        es = EarlyStopper(patience=2, min_delta=0)
        es.early_stop(1.0)  # improvement: sets min to 1.0
        es.early_stop(2.0)  # no improvement: counter -> 1
        es.early_stop(0.5)  # improvement: counter should reset to 0
        self.assertEqual(es.counter, 0)

    def test_no_stop_after_late_improvement(self):
        """Stopping should be prevented when a new minimum appears."""
        es = EarlyStopper(patience=2, min_delta=0)
        es.early_stop(1.0)
        es.early_stop(1.5)  # counter = 1
        result = es.early_stop(0.1)  # improvement -> counter reset
        self.assertFalse(result)
        self.assertEqual(es.counter, 0)

    # ------------------------------------------------------------------
    # Equal loss — should NOT increment counter (not strictly greater)
    # ------------------------------------------------------------------

    def test_equal_loss_does_not_increment_counter(self):
        """Loss equal to min_validation_loss is neither < nor > so counter stays."""
        es = EarlyStopper(patience=1, min_delta=0)
        es.early_stop(1.0)  # min = 1.0
        es.early_stop(1.0)  # exactly equal: no increment
        self.assertEqual(es.counter, 0)

    def test_equal_loss_does_not_stop(self):
        es = EarlyStopper(patience=1, min_delta=0)
        es.early_stop(1.0)
        result = es.early_stop(1.0)
        self.assertFalse(result)

    # ------------------------------------------------------------------
    # No improvement — counter increments
    # ------------------------------------------------------------------

    def test_counter_increments_on_no_improvement(self):
        es = EarlyStopper(patience=3, min_delta=0)
        es.early_stop(1.0)  # sets min to 1.0
        es.early_stop(1.5)  # no improvement -> counter = 1
        self.assertEqual(es.counter, 1)

    def test_counter_increments_twice(self):
        es = EarlyStopper(patience=3, min_delta=0)
        es.early_stop(1.0)
        es.early_stop(1.5)
        es.early_stop(1.6)
        self.assertEqual(es.counter, 2)

    def test_no_stop_before_patience_exceeded(self):
        es = EarlyStopper(patience=2, min_delta=0)
        es.early_stop(1.0)
        result = es.early_stop(1.5)  # counter = 1, patience = 2 -> no stop
        self.assertFalse(result)

    # ------------------------------------------------------------------
    # Patience exceeded — should stop
    # ------------------------------------------------------------------

    def test_stop_when_patience_exceeded(self):
        es = EarlyStopper(patience=1, min_delta=0)
        es.early_stop(1.0)  # min = 1.0
        result = es.early_stop(1.5)  # counter = 1 >= patience=1 -> True
        self.assertTrue(result)

    def test_stop_after_multiple_no_improvements(self):
        es = EarlyStopper(patience=3, min_delta=0)
        es.early_stop(1.0)
        es.early_stop(1.1)  # counter = 1
        es.early_stop(1.2)  # counter = 2
        result = es.early_stop(1.3)  # counter = 3 >= patience=3 -> True
        self.assertTrue(result)

    def test_continues_returning_true_after_threshold(self):
        """Once patience is exceeded, subsequent calls should also return True."""
        es = EarlyStopper(patience=1, min_delta=0)
        es.early_stop(1.0)
        es.early_stop(1.5)  # triggers stop
        result_2 = es.early_stop(1.6)
        self.assertTrue(result_2)

    # ------------------------------------------------------------------
    # min_delta tests
    # ------------------------------------------------------------------

    def test_no_stop_when_loss_within_min_delta(self):
        """Loss that exceeds min by less than min_delta should not increment counter."""
        es = EarlyStopper(patience=1, min_delta=1.0)
        es.early_stop(1.0)  # min = 1.0
        result = es.early_stop(1.5)  # 1.5 <= 1.0 + 1.0 -> no increment
        self.assertFalse(result)

    def test_stop_when_loss_exceeds_min_delta(self):
        es = EarlyStopper(patience=1, min_delta=0.1)
        es.early_stop(1.0)  # min = 1.0
        result = es.early_stop(1.5)  # 1.5 > 1.0 + 0.1 -> counter=1 >= patience=1
        self.assertTrue(result)

    def test_min_delta_boundary_exactly_equal(self):
        """Loss exactly at min + min_delta should not increment (not strictly greater)."""
        es = EarlyStopper(patience=1, min_delta=0.5)
        es.early_stop(1.0)
        result = es.early_stop(1.5)  # 1.5 == 1.0 + 0.5 -> not >
        self.assertFalse(result)

    # ------------------------------------------------------------------
    # Long plateau tests
    # ------------------------------------------------------------------

    def test_long_plateau_triggers_stop(self):
        """A sustained plateau over many calls must eventually stop."""
        es = EarlyStopper(patience=5, min_delta=0)
        es.early_stop(1.0)  # min = 1.0
        for _ in range(5):
            es.early_stop(1.0 + 1e-9)  # tiny increase > 0 but within no-delta
        # counter should have incremented 5 times
        self.assertGreaterEqual(es.counter, 5)

    def test_patience_1_stops_immediately_on_first_bad(self):
        es = EarlyStopper(patience=1, min_delta=0)
        es.early_stop(0.5)
        result = es.early_stop(0.6)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()

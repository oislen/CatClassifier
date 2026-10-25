import os
import sys
import time
import unittest

sys.path.append(os.path.join(os.getcwd(), "model"))

from utilities.TimeIt import TimeIt

# ---------------------------------------------------------------------------
# Fixtures — a single shared TimeIt instance with two logged steps
# ---------------------------------------------------------------------------

parent_key_1 = "Stage1"
sub_key_1    = "Load"
parent_key_2 = "Stage2"
sub_key_2    = None          # sub_key is optional

exp_log_keys = ["parent_key", "sub_key", "step_time", "cumulative_time"]

obs_timeit = TimeIt()
obs_timeit.log_time(parent_key=parent_key_1, sub_key=sub_key_1)
time.sleep(0.01)             # small sleep so step_time > 0
obs_timeit.log_time(parent_key=parent_key_2, sub_key=sub_key_2)


class TestTimeIt(unittest.TestCase):
    """
    Comprehensive unit tests for TimeIt.

    Verifies log structure, key names, value types, step timing, cumulative
    timing, rounding behaviour, the reset() method, log growth across many
    steps, and independence of separate instances.
    """

    def setUp(self):
        self.timeit = obs_timeit
        self.exp_log_keys = exp_log_keys

    # ------------------------------------------------------------------
    # Log structure tests
    # ------------------------------------------------------------------

    def test_log_is_list(self):
        self.assertIsInstance(self.timeit.log, list)

    def test_log_length(self):
        """Two log_time calls should produce two entries."""
        self.assertEqual(len(self.timeit.log), 2)

    def test_log_entry_type(self):
        self.assertIsInstance(self.timeit.log[0], dict)

    def test_log_entry_keys_type(self):
        self.assertEqual(type(list(self.timeit.log[0].keys())), type(self.exp_log_keys))

    def test_log_entry_keys_len(self):
        self.assertEqual(len(self.timeit.log[0]), len(self.exp_log_keys))

    def test_log_entry_keys_values(self):
        self.assertEqual(list(self.timeit.log[0].keys()), self.exp_log_keys)

    def test_all_entries_have_same_keys(self):
        for entry in self.timeit.log:
            self.assertEqual(list(entry.keys()), self.exp_log_keys)

    # ------------------------------------------------------------------
    # parent_key / sub_key content tests
    # ------------------------------------------------------------------

    def test_parent_key_first_entry(self):
        self.assertEqual(self.timeit.log[0]["parent_key"], parent_key_1)

    def test_sub_key_first_entry(self):
        self.assertEqual(self.timeit.log[0]["sub_key"], sub_key_1)

    def test_parent_key_second_entry(self):
        self.assertEqual(self.timeit.log[1]["parent_key"], parent_key_2)

    def test_sub_key_second_entry_is_none(self):
        """sub_key defaults to None and must be stored as None."""
        self.assertIsNone(self.timeit.log[1]["sub_key"])

    def test_parent_key_is_string(self):
        self.assertIsInstance(self.timeit.log[0]["parent_key"], str)

    # ------------------------------------------------------------------
    # Timing value types
    # ------------------------------------------------------------------

    def test_step_time_type(self):
        self.assertIsInstance(self.timeit.log[0]["step_time"], float)

    def test_cumulative_time_type(self):
        self.assertIsInstance(self.timeit.log[0]["cumulative_time"], float)

    # ------------------------------------------------------------------
    # Timing value constraints
    # ------------------------------------------------------------------

    def test_step_time_is_positive(self):
        for entry in self.timeit.log:
            self.assertGreater(entry["step_time"], 0)

    def test_cumulative_time_is_positive(self):
        for entry in self.timeit.log:
            self.assertGreater(entry["cumulative_time"], 0)

    def test_cumulative_time_increases(self):
        """Each cumulative_time must be >= the previous one."""
        times = [e["cumulative_time"] for e in self.timeit.log]
        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i], times[i - 1])

    def test_cumulative_equals_sum_of_step_times(self):
        """cumulative_time of last entry must equal sum of all step_times."""
        total_steps = sum(e["step_time"] for e in self.timeit.log)
        last_cumulative = self.timeit.log[-1]["cumulative_time"]
        self.assertAlmostEqual(last_cumulative, total_steps, places=4)

    def test_step_time_rounded_to_5dp(self):
        """step_time must be rounded to exactly 5 decimal places."""
        for entry in self.timeit.log:
            val = entry["step_time"]
            self.assertEqual(val, round(val, 5))

    def test_cumulative_time_rounded_to_5dp(self):
        """cumulative_time must be rounded to exactly 5 decimal places."""
        for entry in self.timeit.log:
            val = entry["cumulative_time"]
            self.assertEqual(val, round(val, 5))

    # ------------------------------------------------------------------
    # Log growth tests
    # ------------------------------------------------------------------

    def test_log_grows_with_each_call(self):
        """Each log_time call must append exactly one new entry."""
        t = TimeIt()
        for i in range(5):
            t.log_time(parent_key=f"step_{i}")
            self.assertEqual(len(t.log), i + 1)

    def test_many_steps_cumulative_ordering(self):
        """With 10 steps, cumulative_time must be strictly non-decreasing."""
        t = TimeIt()
        for i in range(10):
            t.log_time(parent_key=f"step_{i}")
        times = [e["cumulative_time"] for e in t.log]
        for i in range(1, len(times)):
            self.assertGreaterEqual(times[i], times[i - 1])

    def test_many_steps_sum_equals_last_cumulative(self):
        t = TimeIt()
        for i in range(8):
            t.log_time(parent_key=f"k{i}")
        total = sum(e["step_time"] for e in t.log)
        self.assertAlmostEqual(t.log[-1]["cumulative_time"], total, places=4)

    # ------------------------------------------------------------------
    # reset() tests
    # ------------------------------------------------------------------

    def test_reset_clears_log(self):
        t = TimeIt()
        t.log_time(parent_key="A", sub_key="B")
        t.reset()
        self.assertEqual(len(t.log), 0)

    def test_reset_zeroes_cumulative_time(self):
        t = TimeIt()
        t.log_time(parent_key="A")
        t.reset()
        self.assertEqual(t.cumulative_time, 0.0)

    def test_reset_clears_step_time(self):
        t = TimeIt()
        t.log_time(parent_key="A")
        t.reset()
        self.assertIsNone(t.step_time)

    def test_reset_clears_previous_time(self):
        t = TimeIt()
        t.log_time(parent_key="A")
        t.reset()
        self.assertIsNone(t.previous_time)

    def test_can_log_after_reset(self):
        t = TimeIt()
        t.log_time(parent_key="A")
        t.reset()
        t.log_time(parent_key="B", sub_key="C")
        self.assertEqual(len(t.log), 1)
        self.assertEqual(t.log[0]["parent_key"], "B")

    def test_multiple_resets(self):
        """Resetting multiple times should always return to clean state."""
        t = TimeIt()
        for _ in range(3):
            t.log_time(parent_key="X")
            t.reset()
            self.assertEqual(len(t.log), 0)
            self.assertIsNone(t.step_time)
            self.assertEqual(t.cumulative_time, 0.0)

    def test_cumulative_time_restarts_after_reset(self):
        """Cumulative time after reset must equal only the post-reset step_time."""
        t = TimeIt()
        t.log_time(parent_key="pre")
        t.reset()
        t.log_time(parent_key="post")
        post_entry = t.log[0]
        # After a reset, cumulative_time restarts from 0, so it must equal
        # exactly the single step_time recorded after the reset.
        self.assertAlmostEqual(
            post_entry["cumulative_time"], post_entry["step_time"], places=4
        )

    # ------------------------------------------------------------------
    # Instance independence test
    # ------------------------------------------------------------------

    def test_two_instances_are_independent(self):
        """Logging into one TimeIt must not affect another."""
        t1 = TimeIt()
        t2 = TimeIt()
        t1.log_time(parent_key="t1_only")
        self.assertEqual(len(t1.log), 1)
        self.assertEqual(len(t2.log), 0)


if __name__ == "__main__":
    unittest.main()

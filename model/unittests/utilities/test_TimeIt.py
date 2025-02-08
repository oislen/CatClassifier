import unittest
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "models"))

import cons
from utilities.TimeIt import TimeIt

parentKey="Test Parent Key"
subKey="Test Sub Key"
exp_log_keys = ["parentKey", "subKey", "stepTime", "cumulativeTime"]
obs_timeit = TimeIt()
obs_timeit.logTime(parentKey=parentKey, subKey=subKey)
obs_log_keys = list(obs_timeit.log[0].keys())

class Test_TimeIt(unittest.TestCase):
    """"""

    def setUp(self):
        self.exp_log_keys = exp_log_keys
        self.obs_log_keys = obs_log_keys

    def test_type(self):
        self.assertEqual(type(self.obs_log_keys), type(self.exp_log_keys))

    def test_len(self):
        self.assertEqual(len(self.obs_log_keys), len(self.exp_log_keys))

    def test_values(self):
        self.assertEqual(self.obs_log_keys, self.exp_log_keys)

if __name__ == "__main__":
    unittest.main()

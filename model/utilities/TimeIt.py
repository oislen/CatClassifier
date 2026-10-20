import logging
from time import time

from beartype import beartype


class TimeIt:
    def __init__(self):
        """An object for timing code execution times, records time step and cumulative time."""
        self.log = []
        self.current_time = time()
        self.previous_time = None
        self.step_time = None
        self.cumulative_time = 0.0

    @beartype
    def log_time(self, parent_key: str, sub_key: str | None = None):
        """Sets a timestamp for code execution step

        Parameters
        ----------
        paraentKey : str
            The parent key to label the code execution step
        sub_key : str
            The sub key to label the code execution step, default is None
        """
        self.previous_time = self.current_time
        self.current_time = time()
        self.step_time = round((self.current_time - self.previous_time), 5)
        self.cumulative_time = round(self.step_time + self.cumulative_time, 5)
        log_entry = {
            "parent_key": parent_key,
            "sub_key": sub_key,
            "step_time": self.step_time,
            "cumulative_time": self.cumulative_time,
        }
        logging.info(log_entry)
        self.log.append(log_entry)

    def reset(self):
        """Resets the timing object"""
        self.log = []
        self.current_time = time()
        self.previous_time = None
        self.step_time = None
        self.cumulative_time = 0.0

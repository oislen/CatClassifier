import logging
from beartype import beartype
from typing import Union
from time import time

class TimeIt():

    def __init__(self):
        """An object for timing code execution times, records time step and cumulative time.
        """
        self.log = []
        self.currentTime = time()
        self.previousTime = None
        self.stepTime = None
        self.cumulativeTime = 0.0
    
    @beartype
    def logTime(
        self,
        parentKey:str,
        subKey:Union[str,None]=None
        ):
        """Sets a timestamp for code execution step

        Parameters
        ----------
        paraentKey : str
            The parent key to label the code execution step
        subKey : str
            The sub key to label the code execution step, default is None
        """
        self.previousTime = self.currentTime
        self.currentTime = time()
        self.stepTime = round((self.currentTime - self.previousTime), 5)
        self.cumulativeTime = round(self.stepTime + self.cumulativeTime, 5)
        logEntry = {"parentKey":parentKey, "subKey":subKey, "stepTime":self.stepTime, "cumulativeTime":self.cumulativeTime}
        logging.info(logEntry)
        self.log.append(logEntry)

    def reset(self):
        """Resets the timing object
        """
        self.log = []
        self.currentTime = time()
        self.previousTime = None
        self.stepTime = None
        self.cumulativeTime = 0.0
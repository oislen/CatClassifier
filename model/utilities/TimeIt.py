import logging
from time import time

class TimeIt():
    """
    """

    def __init__(self):
        """
        """
        self.log = []
        self.currentTime = time()
        self.previousTime = None
        self.cumulativeTime = 0.0
    
    def logTime(self, parentKey, subKey=None):
        """
        """
        self.previousTime = self.currentTime
        self.currentTime = time()
        self.cumulativeTime = (self.currentTime - self.previousTime) + self.cumulativeTime
        logEntry = {"parentKey":parentKey, "subKey":subKey, "cumulativeTime":self.cumulativeTime}
        logging.info(logEntry)
        self.log.append(logEntry)
    
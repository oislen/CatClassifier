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
        self.stepTime = None
        self.cumulativeTime = 0.0
    
    def logTime(self, parentKey, subKey=None):
        """
        """
        self.previousTime = self.currentTime
        self.currentTime = time()
        self.stepTime = round((self.currentTime - self.previousTime), 5)
        self.cumulativeTime = round(self.stepTime + self.cumulativeTime, 5)
        logEntry = {"parentKey":parentKey, "subKey":subKey, "stepTime":self.stepTime, "cumulativeTime":self.cumulativeTime}
        logging.info(logEntry)
        self.log.append(logEntry)
    
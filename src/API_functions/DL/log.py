import logging
import sys
from contextlib import contextmanager
import wandb
import pandas as pd
import pprint
import copy

class LoggerManager:
    """Manages logging configuration and states"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.handler = logging.StreamHandler(sys.stdout)
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self._enabled = True

    def enable(self):
        """Enable logging"""
        self.logger.setLevel(logging.INFO)
        self._enabled = True

    def disable(self):
        """Disable logging"""
        self.logger.setLevel(logging.CRITICAL + 1)  # Set to higher than CRITICAL to disable all logging
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if logging is enabled"""
        return self._enabled

    @contextmanager
    def disabled(self):
        """Context manager to temporarily disable logging"""
        previous_state = self._enabled
        self.disable()
        try:
            yield
        finally:
            if previous_state:
                self.enable()

    def info(self, msg: str):
        """Log info message if enabled"""
        if self._enabled:
            self.logger.info(msg)

    def error(self, msg: str):
        """Log error message if enabled"""
        if self._enabled:
            self.logger.error(msg)


# Singleton instance
logger_manager = LoggerManager()


class DataLogger:
    """Handles data logging to various outputs"""
    def __init__(self, log_method: str):
        self.log_method = log_method
        self.data_frame = pd.DataFrame()

    def log(self, data: dict):
        if self.log_method == 'wandb':
            log_wandb(data)
        elif self.log_method == 'csv':
            log2DataFrame(data, self.data_frame)
        elif self.log_method == 'on_screen':
            pprint.pprint(data)
        elif self.log_method == 'all':
            log_wandb(data)
            log2DataFrame(data, self.data_frame)
            pprint.pprint(data)
        else:
            raise ValueError('Invalid log method, replace with wandb, csv or on_screen or all')

    def get_data_frame(self):
        return self.data_frame


# Using pandas to store the increasing data
def log2DataFrame(data: dict, data_frame: pd.DataFrame):
    # To put all scalars in data dict in one first row
    df = pd.DataFrame(data, index=[0])        
    data_frame = pd.concat([data_frame, df], axis=0)
    return data_frame


# Log the data to the wandb
def log_wandb(data: dict):
    # To avoid a log attribute named the 'epoch'
    data1 = copy.copy(data)
    del data1['epoch']
    wandb.log(data1)
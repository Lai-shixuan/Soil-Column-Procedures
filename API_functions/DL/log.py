import wandb
import pandas as pd
import pprint
import copy


# A class to contain a series of logs methods, option to log to wandb or csv or both. When initialing the class, the user can specify the log method.
# And then the use can call the log method to log the data using the specified methods.
class Logger:
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

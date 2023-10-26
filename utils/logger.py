"""Implements the logger for saving results."""

from typing import Any, Dict

import os
import json
import numpy as np
import torch


class Logger(object):
    """Implements the custom logger for saving results.
    
    Attributes:
        logs: logger object.
        path: logging path.
        seed: random seed.
        total: total number of steps.
        filename: filename for storage.
    """
    def __init__(self, log_dir: str, env_name: str, seed: int, steps: int):
        """Initializes the logger object."""
        self.logs = {
            'returns' : []
        }
        self.path = log_dir+'/'+str(env_name)+'/'
        self.seed = str(seed)
        self.total = steps
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.filename = self.path+self.seed

    def log(self, stats: Dict[str, Any]):
        """Logs the statistics of training."""
        for key in stats.keys():
            if key not in self.logs.keys():
                self.logs[key] = []
            self.logs[key].append(stats[key])
        with open(self.filename+'.json', 'w') as fp:
            json.dump(self.logs, fp)

    def get_filename(self):
        """Returns the logging filename."""
        return self.path

def compute_monte_carlo(array: np.ndarray, terminals: np.ndarray,
                        discount: np.ndarray):
    """COmputes monte carlo returns."""
    new_array = []
    disc_val = 0
    for val, is_terminal in zip(reversed(array), reversed(terminals)):
        if is_terminal:
            disc_val = 0
        disc_val = val + (discount * disc_val)
        new_array.insert(0, disc_val)
    new_array = torch.tensor(new_array, dtype=torch.float32).to(device)
    new_array = (new_array - new_array.mean()) / (new_array.std() + 1e-7)
    return new_array

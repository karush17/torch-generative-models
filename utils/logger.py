import numpy as np
import torch
import json
import os
import collections

class Logger(object):
    def __init__(self, log_dir, env_name, seed, steps):
        self.logs = {
            'returns' : []
        }
        self.path = log_dir+'/'+str(env_name)+'/'
        self.seed = str(seed)
        self.total = steps
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.filename = self.path+self.seed
    
    def log(self, stats):
        for key in stats.keys():
            if key not in self.logs.keys():
                self.logs[key] = []
            self.logs[key].append(stats[key])
        with open(self.filename+'.json', 'w') as fp:
            json.dump(self.logs, fp)
    
    def get_filename(self):
        return self.path
    
def compute_monte_carlo(array, terminals, discount):
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

import numpy as np
class Arm:
    def __init__(self, true_p):
        self.true_p = true_p
        self.reset()
    def reset(self):
        self.impressions = 0
        self.actions = 0
    def get_state(self):
        return self.impressions,self.actions
    def get_rate(self):
        return self.actions / self.impressions if self.impressions >0 else 0.
    def pull(self):
        self.impressions+=1
        res = 1 if np.random.random() < self.true_p else 0
        self.actions+=res
        return res
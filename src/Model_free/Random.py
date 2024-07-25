import numpy as np


class Random_Policy:

    def __init__(self, u_max, u_min):
        self.u_max = u_max
        self.u_min = u_min

    def get_action(self, x):
        return self.u_min + (self.u_max - self.u_min) * np.random.random()

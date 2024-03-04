import numpy as np

class Random:

    def __init__(self, u_max = 5, u_min = -5):
        self.u_max = u_max
        self.u_min = u_min

    def get_action(self, x):
        return self.u_min + (self.u_max-self.u_min)*np.random.random()
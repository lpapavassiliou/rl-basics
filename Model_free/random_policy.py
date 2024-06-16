import numpy as np

class Random:

    def __init__(self, u_max_abs = 2):
        self.u_max_abs = u_max_abs

    def get_action(self, x):
        return -self.u_max_abs + 2*self.u_max_abs*np.random.random()
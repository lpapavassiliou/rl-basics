from environment import Environment
import matplotlib.pyplot as plt
import numpy as np

world = Environment(theta_init=np.pi/2)
world.set_reference([np.pi/3, 0])

for i in range(200000):
    action = 0 #2*(np.random.random()-0.5)
    world.step(action)
    world.visualize(action)


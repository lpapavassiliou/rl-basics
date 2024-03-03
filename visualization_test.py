import numpy as np

from environment import Environment

world = Environment(theta_init=np.pi/2)
world.set_reference([np.pi/3, 0])

while True:
    action = 0 #2*(np.random.random()-0.5)
    _, _, done = world.step(action)
    world.visualize(action)

    if done:
        break


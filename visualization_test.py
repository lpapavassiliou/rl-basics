import numpy as np

from environment import Environment

from Model_free import Random

world = Environment(theta_init=np.pi/2)
policy = Random(u_max=world.mparams.u_max, u_min=world.mparams.u_min)

world.set_reference([np.pi/3, 0])

state = world.get_state()
while True:
    action = policy.get_action(state)
    state, _, done = world.step(action)
    world.visualize(action)

    if done:
        break


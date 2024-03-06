import numpy as np

from Utils import Environment
from Model_free import Random, TD_learning

# Set the random seed
np.random.seed(41)

world = Environment(theta_init=np.pi/2)
world.set_reference([np.pi/3, 0])

# policy = Random(u_max=world.mparams.u_max, u_min=world.mparams.u_min)
policy = TD_learning(world, load_V=True)
# policy.plot_V()
policy.train()
# policy.plot_V()

state = world.get_state()
while True:
    action = policy.get_action(state, training=False)
    state, _, done = world.step(action)
    world.visualize(action)
    if done:
        break
    print(f"Steps: {world.steps_taken}/{world.lparams.max_steps}")


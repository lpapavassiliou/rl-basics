import gymnasium as gym
import numpy as np

# from Model_free import Random, TD_learning, Q_learning, NN_Q_learning
from Model_free import TD_learning, Q_learning

# Set the random seed for reproducibility
np.random.seed(41)

# Initialize the environment
env = gym.make("Pendulum-v1")

# Initialize the policy
# Uncomment the desired policy
# policy = Random(u_max=env.mparams.u_max_abs, u_min=-env.mparams.u_max_abs)
policy = TD_learning(env, load_V=False)
# policy = Q_learning(env, load_Q=False)
# policy = NN_Q_learning(env, load_Q=False)

# Train the policy
policy.learn()

# Close the environment when done
env.close()


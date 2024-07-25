import gymnasium as gym
import numpy as np

from Model_free import *

# Set the random seed for reproducibility
np.random.seed(41)

# Initialize the environment
env = gym.make("Pendulum-v1")

# Initialize the policies
policies = [
    Random_Policy(u_max=env.action_space.high, u_min=-env.action_space.low),
    Tabular_V_Policy(env),
    Tabular_Q_Policy(env),
    DQN_Policy(env),
    SAC_Policy(env)
]

index = 4
policy = policies[index]

# Train the policy
policy.learn()

# Close the environment when done
env.close()


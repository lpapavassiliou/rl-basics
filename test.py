import gymnasium as gym
import numpy as np

# from Model_free import Random, TD_learning, Q_learning, NN_Q_learning
from Model_free import TD_learning, Q_learning

# Initialize the environment
env = gym.make("Pendulum-v1", render_mode="human")

# Initialize the policy
# Uncomment the desired policy
# policy = Random(u_max=env.mparams.u_max_abs, u_min=-env.mparams.u_max_abs)
policy = TD_learning(env, load_V=True)
# policy = Q_learning(env, load_Q=True)
# policy = NN_Q_learning(env, load_Q=True)

# Reset the environment
x, _ = env.reset()

# Testing the policy
while True:
    # Get action from the policy
    u = policy.get_action(x)
    
    # Step the environment
    x, reward, done, truncated, _ = env.step(u)
    
    # Check if the episode is done
    if done or truncated:
        break

# Close the environment when done
env.close()

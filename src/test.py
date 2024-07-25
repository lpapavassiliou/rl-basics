import gymnasium as gym

from Model_free import *

# Initialize the environment
env = gym.make("Pendulum-v1", render_mode="human")

# Initialize the policies
policies = [
    Random_Policy(u_max=env.action_space.high, u_min=-env.action_space.low),
    Tabular_V_Policy(env, load_V=True),
    Tabular_Q_Policy(env, load_Q=True),
    DQN_Policy(env, load_Q=True),
    SAC_Policy(env, load_PI=True),
]

index = 4
policy = policies[index]

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

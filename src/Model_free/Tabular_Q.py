import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class Tabular_Q_Policy:
    """
    Tabular Q-Learning Policy for training an agent using Q-learning on a discrete state and action space.

    Args:
        env: The environment to train the agent on.
        load_Q: Whether to load a pre-trained Q-table.

    Methods:
        - get_bin(x): Returns the bin index for a given state x.
        - get_action(x, deterministic=True): Returns the action for a given state x.
        - get_action_index(x, deterministic=True): Returns the index of the action for a given state x.
        - learn(): Trains the agent using Q-learning.
        - update_Q(x, action_index, x_next, reward): Updates the Q-table based on the observed transition.
    """

    def __init__(self, env: gym.Env, load_Q: bool = False) -> None:
        self.env = env

        self.hyperparameters = {
            "episodes": 150000,
            "state_bins": [21, 21, 65],
            "action_bins": 17,
            "discount_factor": 0.95,
            "l_r": 0.1,
            "eps": 1.0,
            "eps_decay_value": 0.999,
            "eps_min": 0.1,
        }

        # compute action bins
        du = (env.action_space.high - env.action_space.low) / (self.hyperparameters["action_bins"] - 1)
        self.action_space = {}
        for k in range(self.hyperparameters["action_bins"]):
            self.action_space[k] = [env.action_space.low[0] + k * du[0]]

        if load_Q:
            self.Q = np.load("src/Model_free/output/Tabular_Q.npy")
        else:
            self.Q = np.random.uniform(
                low=-2,
                high=-0,
                size=(self.hyperparameters["state_bins"] + [self.hyperparameters["action_bins"]]),
            )

        self.dx = (env.observation_space.high - env.observation_space.low) / [
            n - 1 for n in self.hyperparameters["state_bins"]
        ]

    def get_bin(self, x: np.ndarray) -> tuple[int, int, int]:
        """
        Get the index of the discrete bin for a given state.

        Args:
            x (np.ndarray): The state for which to get the bin index.

        Returns:
            tuple[int, int, int]: The bin index for the given state.
        """
        ds = (x - self.env.observation_space.low) / self.dx
        return tuple(ds.astype(np.int32))

    def get_action(self, x: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get the action for the given state.

        Args:
            x (np.ndarray): The state for which to get the action.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to True.

        Returns:
            np.ndarray: The action for the given state.
        """
        return self.action_space[self.get_action_index(x, deterministic)]

    def get_action_index(self, x: np.ndarray, deterministic: bool = True) -> int:
        """
        Get the index of the action to take for a given state.

        Args:
            x (np.ndarray): The state for which to get the action.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to True.

        Returns:
            int: The index of the action to take.
        """
        if not deterministic and np.random.rand() < self.hyperparameters["eps"]:  # exploration
            return np.random.randint(0, self.hyperparameters["action_bins"])
        x_bin = self.get_bin(x)
        return np.argmax(self.Q[x_bin])

    def learn(self) -> None:
        """
        Trains the agent for a specified number of episodes.

        This function iterates over a specified number of episodes and performs the following steps for each episode:
        1. Resets the environment and initializes the reward for the episode.
        2. Executes the agent in the environment until the episode is done or truncated.
        3. Updates the Q-value table using the obtained state, action, next state, and reward.
        4. Decays the epsilon value for the epsilon-greedy exploration strategy.
        5. If the iteration is a multiple of 100, computes the average reward for the last 100 episodes and plots the discounted cumulative rewards.
        6. Saves the Q-value table to a file.
        """
        rewards = []
        average_rewards = []
        for it in tqdm(range(self.hyperparameters["episodes"]), desc="Training Progress"):
            x, _ = self.env.reset()
            rewards.append(0.0)
            while True:
                action_index = self.get_action_index(x, deterministic=False)
                x_next, reward, done, truncated, _ = self.env.step(self.action_space[action_index])
                self.update_Q(x, action_index, x_next, reward)
                rewards[-1] += reward
                if done or truncated:
                    break
                x = x_next.copy()
            if self.hyperparameters["eps"] > self.hyperparameters["eps_min"]:
                self.hyperparameters["eps"] *= self.hyperparameters["eps_decay_value"]
            if it % 100 == 0:
                average_rewards.append(np.mean(np.array(rewards)))
                # Plot the discounted cumulative rewards
                plt.plot(
                    np.arange(len(average_rewards), dtype=np.int64),
                    average_rewards,
                    color="blue",
                )
                plt.xlabel("Iteration")
                plt.ylabel("Cumulative Rewards")
                plt.title("Cumulative Rewards")
                plt.grid(True)
                plt.draw()
                plt.pause(0.1)
                rewards = []

        print("Saving Q")
        np.save("src/Model_free/output/Tabular_Q.npy", self.Q)

    def update_Q(self, x: np.ndarray, action_index: int, x_next: np.ndarray, reward: float) -> None:
        """
        Updates the Q-value table for a given state-action pair using the obtained next state, reward, and learning rate.

        Parameters:
            x (np.ndarray): The current state.
            action_index (int): The index of the action taken.
            x_next (np.ndarray): The next state.
            reward (float): The reward obtained from the current state and action.
        """
        x_bin = self.get_bin(x)
        x_next_bin = self.get_bin(x_next)
        if x_next_bin[0] == 0 and x_next_bin[1] == 1:
            self.Q[x_bin + (action_index,)] = 0.0
        else:
            q_max = np.max(self.Q[x_next_bin])
            q_expected = reward + self.hyperparameters["discount_factor"] * q_max
            self.Q[x_bin + (action_index,)] = (1 - self.hyperparameters["l_r"]) * self.Q[
                x_bin + (action_index,)
            ] + self.hyperparameters["l_r"] * q_expected

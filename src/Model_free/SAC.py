import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm
import matplotlib.pyplot as plt

from Utils import Buffer


class Critic(nn.Module):
    """
    Critic network for the Soft Actor-Critic (SAC) algorithm.

    Args:
        input_dim (int): Dimension of the input state.
        output_dim (int): Dimension of the output.
        l_r (float): Learning rate for the optimizer.

    Attributes:
        network (nn.Sequential): Neural network layers.
        optimizer (optim.Adam): Adam optimizer for training.

    Methods:
        forward(x, a): Forward pass through the network.

    Returns:
        torch.Tensor: Output tensor from the network.
    """

    def __init__(self, input_dim: int, output_dim: int, l_r: float) -> None:
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=l_r)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor to the network.
            a (torch.Tensor): Input tensor for the action.

        Returns:
            torch.Tensor: Output tensor from the network.
        """
        return self.network(torch.cat([x, a], dim=-1))


class Actor(nn.Module):
    """
    Neural network actor model for the Soft Actor-Critic (SAC) algorithm.

    Args:
        input_dim (int): Dimension of the input state space.
        output_dim (int): Dimension of the output action space.
        l_r (float): Learning rate for the optimizer.

    Attributes:
        layer (nn.Sequential): Neural network layers for feature extraction.
        fc_mean (nn.Linear): Fully connected layer for outputting the mean of the action distribution.
        fc_log_std (nn.Linear): Fully connected layer for outputting the log standard deviation of the action distribution.
        optimizer (optim.Adam): Adam optimizer for training the actor network.

    Methods:
        forward(x): Forward pass through the actor network.
        get_distribution(mean, std): Get the Normal distribution based on mean and standard deviation.
        sample(x): Sample an action from the distribution and calculate the log probability.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: tuple containing the sampled action and its log probability.
    """

    def __init__(self, input_dim: int, output_dim: int, l_r: float) -> None:
        super(Actor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(64, output_dim)
        self.fc_log_std = nn.Linear(64, output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=l_r)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the actor network.

        Args:
            x (torch.Tensor): Input tensor to the actor network.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple containing the mean and log standard deviation of the action distribution.
        """
        x = self.layer(x)
        mean = self.fc_mean(x)
        log_std = torch.clamp(self.fc_log_std(x), -20, 2)
        return mean, log_std

    def get_distribution(self, mean: torch.Tensor, std: torch.Tensor) -> Normal:
        """
        Returns a Normal distribution with the specified mean and standard deviation.

        Parameters:
        mean (torch.Tensor): The mean of the distribution.
        std (torch.Tensor): The standard deviation of the distribution.

        Returns:
        Normal: A Normal distribution object.
        """
        return Normal(mean, std)

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the distribution and calculate the log probability.

        Args:
            x (torch.Tensor): Input tensor to the actor network.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: tuple containing the sampled action and its log probability.
        """
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        distribution = self.get_distribution(mean, std)
        x = distribution.rsample()
        return x, distribution.log_prob(x)


class SAC_Policy:
    """
    SAC_Policy class for implementing the Soft Actor-Critic (SAC) algorithm.

    Args:
        env(gym.Env): The environment for the agent to interact with.
        load_PI (bool): Flag to load pre-trained policy (default is False).

    Attributes:
        hyperparameters (dict): Dictionary containing various hyperparameters for training.
        buffer (Buffer): Buffer object for storing and sampling experiences.
        actor (Actor): Actor network for generating actions.
        critic (Critic): Critic network for estimating Q-values.
        critic_target (Critic): Target Critic network for soft updates.
        alpha (torch.Tensor): Temperature parameter for entropy regularization.
        alpha_optimizer (optim.Adam): Adam optimizer for updating alpha.
        action_scale (torch.Tensor): Scaling factor for actions.
        action_bias (torch.Tensor): Bias factor for actions.
        cum_rewards (list): List to store cumulative rewards during training.
        average_rewards (list): List to store average rewards over episodes.

    Methods:
        learn(): Main method to train the SAC policy.
        collect_rollout(): Collects experiences by interacting with the environment.
        train(): Performs training steps for the actor, critic, and alpha.
        get_action(x): Get an action based on the input state.
        get_action_and_log_prob(x): Get action and log probability for a state.
        clip_actor(sample, log_prob): Clip the action and calculate log probability.
        get_target(x_next, r, done): Calculate the target Q-value for training.
        plot_rewards(): Plot the cumulative rewards over iterations.
    """

    def __init__(self, env: gym.Env, load_PI: bool = False) -> None:
        self.env = env

        self.hyperparameters = {
            "episodes": 5000,
            "n_steps": 200,
            "epochs": 10,
            "batch_size": 32,
            "buffer_size": 5000 * 200,
            "discount_factor": 0.98,
            "init_alpha": torch.log(torch.tensor(0.01)),
            "l_r": 1e-3,
            "tau": 0.01,
            "eps": 1.0,
            "eps_decay_value": 0.999,
            "eps_min": 1e-1,
        }

        self.buffer = Buffer(buffer_size=self.hyperparameters["buffer_size"])

        if load_PI:
            self.actor = Actor(
                input_dim=env.observation_space.shape[0],
                output_dim=env.action_space.shape[0],
                l_r=self.hyperparameters["l_r"],
            )
            self.actor.load_state_dict(torch.load("src/Model_free/output/SAC_NN.pth"))
        else:
            self.actor = Actor(
                input_dim=env.observation_space.shape[0],
                output_dim=1,
                l_r=self.hyperparameters["l_r"],
            )
            self.critic = Critic(
                input_dim=env.observation_space.shape[0] + env.action_space.shape[0],
                output_dim=1,
                l_r=self.hyperparameters["l_r"],
            )
            self.critic_target = Critic(
                input_dim=env.observation_space.shape[0] + env.action_space.shape[0],
                output_dim=1,
                l_r=self.hyperparameters["l_r"],
            )
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.alpha = self.hyperparameters["init_alpha"]
            self.alpha.requires_grad = True
            self.alpha_optimizer = optim.Adam([self.alpha], lr=5 * self.hyperparameters["l_r"])
            self.target_entropy = -env.action_space.shape[0]

        self.action_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0)
        self.action_bias = torch.tensor((self.env.action_space.high + self.env.action_space.low) / 2.0)

        self.cum_rewards = []
        self.average_rewards = []

    def learn(self) -> None:
        """
        Train the SAC policy over a specified number of episodes.

        This method performs the following steps:
        1. Collects rollouts by interacting with the environment.
        2. Trains the actor, critic, and alpha networks using the collected experiences.
        3. Updates the exploration parameter epsilon after each episode.
        4. Periodically plots the cumulative rewards.
        5. Saves the actor's state dictionary to a file after training is complete.
        """
        for episode in tqdm(range(self.hyperparameters["episodes"]), desc="Training Progress"):
            self.collect_rollout()
            self.train()
            if not episode % 100:
                self.plot_rewards()
            if self.hyperparameters["eps"] > self.hyperparameters["eps_min"]:
                self.hyperparameters["eps"] *= self.hyperparameters["eps_decay_value"]

        print("Saving Q")
        torch.save(self.actor.state_dict(), "src/Model_free/output/SAC_NN.pth")

    def collect_rollout(self) -> None:
        """
        Collects a rollout by executing the agent in the environment.

        This function iterates over a rollout by executing the agent in the environment.
        It collects the state, action, next state, reward, and done values.
        The collected values are added to the buffer.
        The function also appends the cumulative rewards to the `cum_rewards` list.
        """

        x, _ = self.env.reset()
        rewards = 0
        while True:
            action = self.get_action(x)
            x_next, reward, done, truncated, _ = self.env.step(action)
            self.buffer.add(x, action, x_next, reward, done)
            rewards += reward
            if done or truncated:
                break
            x = x_next.copy()
        self.cum_rewards.append(rewards)

    def train(self) -> None:
        """
        Trains the SAC policy over a specified number of epochs.

        This method performs the following steps:
        1. Samples a batch of states, actions, next states, rewards, and dones from the buffer.
        2. Computes the target values for the Q-network using the next states, rewards, and dones.
        3. Computes the critic loss using the smooth L1 loss function.
        4. Trains the critic network using the computed critic loss.
        5. Samples an action from the actor network.
        6. Computes the log probability of the action.
        7. Clips the actor's action and log probability.
        8. Computes the entropy.
        9. Computes the actor loss.
        10. Trains the actor network using the computed actor loss.
        11. Computes the alpha loss.
        12. Trains the alpha network using the computed alpha loss.
        13. Updates the Critic-target network using a soft update strategy.
        """
        for _ in range(self.hyperparameters["epochs"]):
            states, actions, next_states, rewards, dones = self.buffer.sample(
                self.hyperparameters["batch_size"], "torch"
            )
            td_target = self.get_target(next_states, rewards, dones)

            #### Critic train ####
            critic_loss = F.smooth_l1_loss(self.critic(states, actions), td_target)
            self.critic.optimizer.zero_grad()
            critic_loss.mean().backward()
            self.critic.optimizer.step()

            #### Actor train ####
            sample, log_prob = self.actor.sample(states)
            action, log_prob = self.clip_actor(sample, log_prob)

            entropy = -self.alpha.exp() * log_prob
            actor_loss = -(self.critic(states, action) + entropy)
            self.actor.optimizer.zero_grad()
            actor_loss.mean().backward()
            self.actor.optimizer.step()

            #### Alpha train ####
            sample, log_prob = self.actor.sample(states)
            _, log_prob = self.clip_actor(
                sample, log_prob
            )  # need to repeat it because self.actor.optimizer.zero_grad() removes log_prob gradient graph
            alpha_loss = -(self.alpha.exp() * (log_prob + self.target_entropy))
            self.alpha_optimizer.zero_grad()
            alpha_loss.mean().backward()
            self.alpha_optimizer.step()

            #### Q target soft-update ####
            for param_target, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                param_target.data.copy_(
                    param_target.data * (1.0 - self.hyperparameters["tau"]) + param.data * self.hyperparameters["tau"]
                )

    def get_action(self, x: np.ndarray) -> np.ndarray:
        """
        Get an action based on the input state.

        Parameters:
            x (np.ndarray): The input state for which to get an action.

        Returns:
            np.ndarray: The action corresponding to the input state.
        """
        action, _ = self.get_action_and_log_prob(torch.tensor(x))
        return action.numpy()

    def get_action_and_log_prob(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get an action and its log probability based on the input state.

        Args:
            x (torch.Tensor): The input state for which to get an action and its log probability.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the action and its log probability.
        """
        with torch.no_grad():
            sample, log_prob = self.actor.sample(x)
            action, log_prob = self.clip_actor(sample, log_prob)
        return action, log_prob

    def clip_actor(self, sample: torch.Tensor, log_prob: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Clips the given sample and calculates the corresponding action and log probability.

        Args:
            sample (torch.Tensor): The input sample to be clipped.
            log_prob (torch.Tensor): The log probability of the input sample.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the clipped action and its log probability.
        """
        sample = torch.tanh(sample)
        action = self.action_scale * sample + self.action_bias
        log_prob -= torch.sum(
            torch.log(self.action_scale * (1 - sample.pow(2)) + 1e-6),
            dim=-1,
            keepdim=True,
        )
        return action, log_prob

    def get_target(self, x_next: torch.Tensor, r: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """
        Calculates the target value for each state in the next time step.

        Args:
            x_next (torch.Tensor): The next state. Shape: (batch_size, state_dim)
            r (torch.Tensor): The reward for each state. Shape: (batch_size,)
            done (torch.Tensor): A boolean tensor indicating whether each state is done. Shape: (batch_size,)

        Returns:
            torch.Tensor: The target value for each state. Shape: (batch_size, 1)
        """
        with torch.no_grad():
            sample, log_prob = self.actor.sample(x_next)
            action, log_prob = self.clip_actor(sample, log_prob)
            entropy = -self.alpha.exp() * log_prob
            q_target = self.critic_target(x_next, action)
            target = r.unsqueeze(1) + (~done.unsqueeze(1)) * self.hyperparameters["discount_factor"] * (
                q_target + entropy
            )
        return target

    def plot_rewards(self) -> None:
        """
        This function plots the average rewards for each episode.
        """
        self.average_rewards.append(np.mean(np.array(self.cum_rewards)))
        # Plot the discounted cumulative rewards
        plt.plot(
            np.arange(len(self.average_rewards), dtype=np.int64),
            self.average_rewards,
            color="blue",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Rewards")
        plt.title("Cumulative Rewards")
        plt.grid(True)
        plt.draw()
        plt.pause(0.1)
        self.cum_rewards = []

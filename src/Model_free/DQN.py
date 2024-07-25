import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from Utils import Buffer


class NN_Q(nn.Module):
    def __init__(self, input_dim, output_dim, l_r):
        super(NN_Q, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=l_r)

    def forward(self, x):
        return self.network(x)


class DQN_Policy:
    def __init__(self, env, load_Q=False):
        self.env = env

        self.hyperparameters = {
            "episodes": 10000,
            "n_steps": 200,
            "epochs": 10,
            "batch_size": 32,
            "buffer_size": 10000 * 200,
            "action_bins": 17,
            "discount_factor": 0.98,
            "l_r": 1e-3,
            "tau": 0.01,
            "eps": 1.0,
            "eps_decay_value": 0.999,
            "eps_min": 1e-1,
        }

        self.buffer = Buffer(buffer_size=self.hyperparameters["buffer_size"])

        # compute action bins
        du = (env.action_space.high - env.action_space.low) / (self.hyperparameters["action_bins"] - 1)
        self.action_space = {}
        for k in range(self.hyperparameters["action_bins"]):
            self.action_space[k] = [env.action_space.low[0] + k * du[0]]

        if load_Q:
            self.Q = NN_Q(
                input_dim=env.observation_space.shape[0],
                output_dim=self.hyperparameters["action_bins"],
                l_r=self.hyperparameters["l_r"],
            )
            self.Q.load_state_dict(torch.load("src/Model_free/output/DQN_NN.pth"))
        else:
            self.Q = NN_Q(
                input_dim=env.observation_space.shape[0],
                output_dim=self.hyperparameters["action_bins"],
                l_r=self.hyperparameters["l_r"],
            )
            self.Q_target = NN_Q(
                input_dim=env.observation_space.shape[0],
                output_dim=self.hyperparameters["action_bins"],
                l_r=self.hyperparameters["l_r"],
            )
            self.Q_target.load_state_dict(self.Q.state_dict())

        self.cum_rewards = []
        self.average_rewards = []

    def learn(self):
        for episode in tqdm(range(self.hyperparameters["episodes"]), desc="Training Progress"):
            self.collect_rollout()
            self.train()
            if not episode % 100:
                self.plot_rewards()
            if self.hyperparameters["eps"] > self.hyperparameters["eps_min"]:
                self.hyperparameters["eps"] *= self.hyperparameters["eps_decay_value"]

        print("Saving Q")
        torch.save(self.Q.state_dict(), "src/Model_free/output/DQN_NN.pth")

    def collect_rollout(self):
        x, _ = self.env.reset()
        rewards = 0
        while True:
            action_index = self.get_action_index(x, deterministic=False)
            x_next, reward, done, truncated, _ = self.env.step(self.action_space[action_index])
            self.buffer.add(x, action_index, x_next, reward, done)
            rewards += reward
            if done or truncated:
                break
            x = x_next.copy()
        self.cum_rewards.append(rewards)

    def train(self):
        for _ in range(self.hyperparameters["epochs"]):
            states, action_indeces, next_states, rewards, dones = self.buffer.sample(
                self.hyperparameters["batch_size"], "torch"
            )
            td_target = self.get_target(next_states, rewards, dones)

            #### Q train ####
            q_values = self.Q(states).gather(1, action_indeces)
            q_loss = F.smooth_l1_loss(q_values, td_target)
            self.Q.optimizer.zero_grad()
            q_loss.mean().backward()
            self.Q.optimizer.step()

            #### Q target soft-update ####
            for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
                param_target.data.copy_(
                    param_target.data * (1.0 - self.hyperparameters["tau"]) + param.data * self.hyperparameters["tau"]
                )

    def get_action_index(self, x, deterministic=True):
        if not deterministic and np.random.rand() < self.hyperparameters["eps"]:  # exploration
            return np.random.choice(self.hyperparameters["action_bins"])

        with torch.no_grad():
            return torch.argmax(self.Q(torch.tensor(x))).item()

    def get_action(self, x, deterministic=True):
        return self.action_space[self.get_action_index(x, deterministic)]

    def get_target(self, x_next, r, done):
        with torch.no_grad():
            q_target = self.Q_target(x_next).max(1)[0]
            target = r + (~done) * self.hyperparameters["discount_factor"] * q_target
        return target.unsqueeze(1)

    def plot_rewards(self):
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

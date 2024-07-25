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
    def __init__(self, input_dim, output_dim, l_r):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=l_r)

    def forward(self, x, a):
        return self.network(torch.cat([x, a], dim=-1))


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, l_r):
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

    def forward(self, x):
        x = self.layer(x)
        mean = self.fc_mean(x)
        log_std = torch.clamp(self.fc_log_std(x), -20, 2)
        return mean, log_std

    def get_distribution(self, mean, std):
        return Normal(mean, std)

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        distribution = self.get_distribution(mean, std)
        x = distribution.rsample()
        return x, distribution.log_prob(x)


class SAC_Policy:
    def __init__(self, env, load_PI=False):
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

    def learn(self):
        for episode in tqdm(range(self.hyperparameters["episodes"]), desc="Training Progress"):
            self.collect_rollout()
            self.train()
            if not episode % 100:
                self.plot_rewards()
            if self.hyperparameters["eps"] > self.hyperparameters["eps_min"]:
                self.hyperparameters["eps"] *= self.hyperparameters["eps_decay_value"]

        print("Saving Q")
        torch.save(self.actor.state_dict(), "src/Model_free/output/SAC_NN.pth")

    def collect_rollout(self):
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

    def train(self):
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

    def get_action(self, x):
        action, _ = self.get_action_and_log_prob(torch.tensor(x))
        return action.numpy()

    def get_action_and_log_prob(self, x):
        with torch.no_grad():
            sample, log_prob = self.actor.sample(x)
            action, log_prob = self.clip_actor(sample, log_prob)
        return action, log_prob

    def clip_actor(self, sample, log_prob):
        sample = torch.tanh(sample)
        action = self.action_scale * sample + self.action_bias
        log_prob -= torch.sum(
            torch.log(self.action_scale * (1 - sample.pow(2)) + 1e-6),
            dim=-1,
            keepdim=True,
        )
        return action, log_prob

    def get_target(self, x_next, r, done):
        with torch.no_grad():
            sample, log_prob = self.actor.sample(x_next)
            action, log_prob = self.clip_actor(sample, log_prob)
            entropy = -self.alpha.exp() * log_prob
            q_target = self.critic_target(x_next, action)
            target = r.unsqueeze(1) + (~done.unsqueeze(1)) * self.hyperparameters["discount_factor"] * (
                q_target + entropy
            )
        return target

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

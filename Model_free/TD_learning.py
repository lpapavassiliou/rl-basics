import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class TD_learning:

    def __init__(self, env, load_V=False):
        self.env = env

        self.hyperparameters = {
            "episodes": 30000,
            "state_bins": [21, 21, 65],
            "action_bins": 17,
            "discount_factor": 0.95,
            "l_r": 0.1,
            "eps": 1.,
            "eps_decay_value": 0.999,
            "eps_min": 0.1
        }

        # compute action bins
        du = (env.action_space.high - env.action_space.low)/(self.hyperparameters["action_bins"]-1)
        self.action_space = np.zeros(self.hyperparameters["action_bins"])
        for k in range(self.hyperparameters["action_bins"]):
            self.action_space[k] = env.action_space.low[0] + k*du[0]

        if load_V:
            self.V = np.load('Model_free/output/V_learning.npy')
        else:
            self.V = np.random.uniform(low=-2, high=0, size=self.hyperparameters["state_bins"])
        
        self.dx = (env.observation_space.high - env.observation_space.low) / [n-1 for n in self.hyperparameters["state_bins"]]

    def get_bin(self, x):
        ds = (x - self.env.observation_space.low) / self.dx
        return ds.astype(np.int32)

    def get_action(self, x, deterministic=True):
        if not deterministic and np.random.rand() < self.hyperparameters["eps"]: #exploration
            return self.env.action_space.sample()
        x_next, reward = self.predict(x)
        next_V = self.V[tuple(self.get_bin(x_next).T)]
        td = reward + self.hyperparameters["discount_factor"] * next_V
        return [self.action_space[np.argmax(td)]]

    def learn(self):
        rewards = []
        average_rewards = []
        for it in tqdm(range(self.hyperparameters["episodes"]), desc="Training Progress"):
            x, _ = self.env.reset()
            rewards.append(0.)
            while True:
                action = self.get_action(x, deterministic=False)
                x_next, reward, done, truncated, _ = self.env.step(action)
                self.update_V(x, x_next, reward)
                rewards[-1] += reward
                if done or truncated:
                    break
                x = x_next.copy()
                if self.hyperparameters["eps"] > self.hyperparameters["eps_min"]:
                    self.hyperparameters["eps"] *= self.hyperparameters["eps_decay_value"]
            if it % 1000 == 0:
                average_rewards.append(np.mean(np.array(rewards)))
                # Plot the discounted cumulative rewards
                plt.plot(np.arange(len(average_rewards), dtype=np.int64), average_rewards, color='blue')
                plt.xlabel('Iteration')
                plt.ylabel('Cumulative Rewards')
                plt.title('Cumulative Rewards')
                plt.grid(True)
                plt.draw()
                plt.pause(0.1)
                rewards = []

        print("Saving V")
        np.save('Model_free/output/V_learning.npy', self.V)
    
    def update_V(self, x, x_next, reward):
        x_bin = tuple(self.get_bin(x))
        x_next_bin = tuple(self.get_bin(x_next))
        if x_next_bin[0] == 0 and x_next_bin[1] == 1:
            self.V[x_bin] = 0.
        else:
            v_next = self.V[x_next_bin]
            v_expected = reward + self.hyperparameters["discount_factor"]*v_next
            self.V[x_bin] = (1-self.hyperparameters["l_r"])*self.V[x_bin] + self.hyperparameters["l_r"]*v_expected

    def predict(self, x):
        cos, sin, theta_dot = x

        theta = np.clip(np.arctan2(sin, cos), -np.pi, np.pi)
        reward = -(theta ** 2 + 0.1 * theta_dot**2 + 0.001 * (self.action_space**2))

        new_theta_dot = theta_dot + (3 * self.env.unwrapped.g / (2 * self.env.unwrapped.l) * sin + 3.0 / (self.env.unwrapped.m * self.env.unwrapped.l**2) * self.action_space) * self.env.unwrapped.dt
        new_theta_dot = np.clip(new_theta_dot, -self.env.unwrapped.max_speed, self.env.unwrapped.max_speed)
        new_theta = theta + new_theta_dot * self.env.unwrapped.dt


        return np.column_stack((np.cos(new_theta), np.sin(new_theta), new_theta_dot)), reward
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Tabular_Q_Policy:

    def __init__(self, env, load_Q = False):
        self.env = env

        self.hyperparameters = {
            "episodes": 150000,
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
        self.action_space = {}
        for k in range(self.hyperparameters["action_bins"]):
            self.action_space[k] = [env.action_space.low[0] + k*du[0]]

        if load_Q:
            self.Q = np.load('Model_free/output/Tabular_Q.npy')
        else:
            self.Q = np.random.uniform(low=-2, high=-0, size=(self.hyperparameters["state_bins"] + [self.hyperparameters["action_bins"]]))
        
        self.dx = (env.observation_space.high - env.observation_space.low) / [n-1 for n in self.hyperparameters["state_bins"]]

    def get_bin(self, x):
        ds = (x - self.env.observation_space.low) / self.dx
        return tuple(ds.astype(np.int32))
    
    def get_action(self, x, deterministic=True):
        return self.action_space[self.get_action_index(x, deterministic)]
    
    def get_action_index(self, x, deterministic=True):
        if not deterministic and np.random.rand() < self.hyperparameters["eps"]: #exploration
            return np.random.randint(0, self.hyperparameters["action_bins"])
        x_bin = self.get_bin(x)
        return np.argmax(self.Q[x_bin])
    
    def learn(self):
        rewards = []
        average_rewards = []
        for it in tqdm(range(self.hyperparameters["episodes"]), desc="Training Progress"):
            x, _ = self.env.reset()
            rewards.append(0.)
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
                plt.plot(np.arange(len(average_rewards), dtype=np.int64), average_rewards, color='blue')
                plt.xlabel('Iteration')
                plt.ylabel('Cumulative Rewards')
                plt.title('Cumulative Rewards')
                plt.grid(True)
                plt.draw()
                plt.pause(0.1)
                rewards = []

        print("Saving Q")
        np.save('Model_free/output/Tabular_Q.npy', self.Q)
    
    def update_Q(self, x, action_index, x_next, reward):
        x_bin = self.get_bin(x)
        x_next_bin = self.get_bin(x_next)
        if x_next_bin[0] == 0 and x_next_bin[1] == 1:
            self.Q[x_bin + (action_index, )] = 0.
        else:
            q_max = np.max(self.Q[x_next_bin])
            q_expected = reward + self.hyperparameters["discount_factor"]*q_max
            self.Q[x_bin + (action_index, )] = (1-self.hyperparameters["l_r"])*self.Q[x_bin + (action_index, )] + self.hyperparameters["l_r"]*q_expected
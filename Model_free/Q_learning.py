import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import pickle

from Utils import Environment

STATS_EVERY = 100

class Q_learning:

    def __init__(self, world: Environment, load_Q = False):
        self.world = copy(world)
        self.u_max_abs = world.mparams.u_max_abs

        self.hyperparamters = {
            "train_episodes": 150000,
            "max_steps": self.world.lparams.max_steps,
            "theta_dot_max_abs": 8,
            "encoded_state_bins": (21, 21, 65),
            "action_bins": 17,
            "discount_factor": 0.95,
            "momentum": 0.1,
            "exploration_factor": 0.1
        }

        # compute state bins
        self.U = np.zeros(self.hyperparamters["action_bins"])
        du = 2*self.u_max_abs/(self.U.shape[0]-1)
        for k in range(self.U.shape[0]):
            self.U[k] = -self.u_max_abs + k*du

        if load_Q:
            self.Q = np.load('Model_free/output/Q_learning_Q.npy')
        else:
            self.Q = np.random.random(self.hyperparamters["encoded_state_bins"] + (self.hyperparamters["action_bins"], ))-2
        
        self.bin_ref, _ = self.find_bins(self.world.get_encoded_reference())

    def find_bins(self, x_encoded):
        dcos = 2/(self.hyperparamters["encoded_state_bins"][0]-1)
        dsin = 2/(self.hyperparamters["encoded_state_bins"][1]-1)
        dtheta_dot = 2*self.hyperparamters["theta_dot_max_abs"]/(self.hyperparamters["encoded_state_bins"][2]-1)
        dX_encoded = np.array([dcos, dsin, dtheta_dot])
        X_encoded_min = np.array([-1., -1., -self.hyperparamters["theta_dot_max_abs"]])

        bin_indeces = np.round((x_encoded-X_encoded_min) / dX_encoded).astype(np.int32)

        if len(x_encoded.shape) == 1:
            bin_indeces[0] = np.clip(bin_indeces[0], 0, self.hyperparamters["encoded_state_bins"][0]-1)
            bin_indeces[1] = np.clip(bin_indeces[1], 0, self.hyperparamters["encoded_state_bins"][1]-1)
            bin_indeces[2] = np.clip(bin_indeces[2], 0, self.hyperparamters["encoded_state_bins"][2]-1)
            bin_encoded_states = X_encoded_min + bin_indeces*dX_encoded
            bin_indeces = tuple(bin_indeces)
        else:
            bin_indeces = np.round((x_encoded-X_encoded_min) / dX_encoded).astype(np.int32)
            bin_indeces[:, 0] = np.clip(bin_indeces[:, 0], 0, self.hyperparamters["encoded_state_bins"][0]-1)
            bin_indeces[:, 1] = np.clip(bin_indeces[:, 1], 0, self.hyperparamters["encoded_state_bins"][1]-1)
            bin_indeces[:, 2] = np.clip(bin_indeces[:, 2], 0, self.hyperparamters["encoded_state_bins"][2]-1)
            bin_encoded_states = X_encoded_min + bin_indeces*dX_encoded
            bin_indeces = list(map(tuple, bin_indeces.reshape(-1, 3)))
        return bin_indeces, bin_encoded_states
    
    def train(self):
        _, x_encoded = self.world.reset()
        self.world.lparams.max_steps = self.hyperparamters["max_steps"]
        action = 0
        it = 0
        episode_reward = 0
        episode_rewards = []
        aggr_episode_rewards = {'it': [], 'avg': [], 'ep': []}
        print(f"Training episode {it+1}/{self.hyperparamters['train_episodes']}")
        while it < self.hyperparamters["train_episodes"]:
            action, action_index = self.get_action_in_training(x_encoded)
            _, x_encoded_next, reward, done = self.world.step(action)
            self.update_Q(x_encoded, x_encoded_next, action_index, reward)
            episode_reward += reward
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                it += 1
                if not it % STATS_EVERY:
                    aggr_episode_rewards['it'].append(it)
                    aggr_episode_rewards['avg'].append(sum(episode_rewards[-STATS_EVERY:]) / STATS_EVERY)
                action = 0
                _, x_encoded = self.world.reset()
                if it < self.hyperparamters["train_episodes"]:
                    print(f"Training episode {it+1}/{self.hyperparamters['train_episodes']}")
            x_encoded = x_encoded_next.copy()
        aggr_episode_rewards['ep'] = episode_rewards
        print("Saving Q")
        np.save('Model_free/output/Q_learning_Q.npy', self.Q)
        print("Saving statistics")
        with open('Model_free/statistics/Q_learning_statistics.pkl', 'wb') as f:
            pickle.dump(aggr_episode_rewards, f)

    def get_action_in_training(self, x_encoded):        
        if np.random.rand() < self.hyperparamters["exploration_factor"]: #exploration
            action_index = np.random.choice(self.hyperparamters["action_bins"])
        else: #exploitation
            bin_index, _ = self.find_bins(x_encoded)
            #return action with higher Q value
            action_index = np.argmax(self.Q[bin_index])
        return self.U[action_index], action_index
    
    def get_action(self, x):
        bin_index, _ = self.find_bins(self.world.encode_state(x))
        #return action with higher Q value
        action_index = np.argmax(self.Q[bin_index])
        return self.U[action_index]
    
    def update_Q(self, x_encoded, x_encoded_next, action_index, reward):
        bin_indeces, bin_encoded_states = self.find_bins(np.append([x_encoded], [x_encoded_next], axis=0))
        if bin_indeces[0] == self.bin_ref:
            print(self.world.decode_state(x_encoded), self.world.x_ref, bin_indeces[0], self.world.decode_state(bin_encoded_states[0, :]))
            self.Q[bin_indeces[0] + (action_index, )] = 0
            print("ref reached in training !!!!!!!!!!")
        else:
            Q_new = reward + self.hyperparamters["discount_factor"]*np.max(self.Q[bin_indeces[1]])
            self.Q[bin_indeces[0] + (action_index, )] = (1-self.hyperparamters["momentum"])*self.Q[bin_indeces[0] + (action_index, )] + self.hyperparamters["momentum"]*Q_new

    def plot_statistics(self): 
        with open('Model_free/statistics/Q_learning_statistics.pkl', 'rb') as f:
            aggr_episode_rewards = pickle.load(f)
        plt.plot(aggr_episode_rewards['it'], aggr_episode_rewards['avg'], label="average rewards")
        plt.plot(STATS_EVERY + np.arange(len(aggr_episode_rewards['ep'])-STATS_EVERY), aggr_episode_rewards['ep'][STATS_EVERY:], label="all rewards", alpha=0.5)
        plt.legend(loc=4)
        plt.show()
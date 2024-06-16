import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import copy
import pickle

from Utils import Environment

STATS_EVERY = 100

class TD_learning:

    def __init__(self, world: Environment, load_V = False):
        self.world = copy(world)
        self.u_max_abs = world.mparams.u_max_abs

        self.hyperparamters = {
            "train_episodes": 200,
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

        if load_V:
            self.V = np.load('Model_free/output/td_learning_V.npy')
        else:
            self.V = np.random.random(self.hyperparamters["encoded_state_bins"])-2
        
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
        x, x_encoded = self.world.reset()
        self.world.lparams.max_steps = self.hyperparamters["max_steps"]
        action = 0
        it = 0
        episode_reward = 0
        episode_rewards = []
        aggr_episode_rewards = {'it': [], 'avg': [], 'ep': []}
        for it in tqdm(range(self.hyperparamters["train_episodes"]), desc="Training Progress"):
            action = self.get_action_in_training(x)
            x, x_encoded_next, reward, done = self.world.step(action)
            self.update_V(x_encoded, x_encoded_next, reward)
            episode_reward += reward
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                it += 1
                if not it % STATS_EVERY:
                    aggr_episode_rewards['it'].append(it)
                    aggr_episode_rewards['avg'].append(sum(episode_rewards[-STATS_EVERY:]) / STATS_EVERY)
                action = 0
                x, x_encoded = self.world.reset()
                if it < self.hyperparamters["train_episodes"]:
                    print(f"Training episode {it+1}/{self.hyperparamters['train_episodes']}")   
            x_encoded = x_encoded_next.copy()
        aggr_episode_rewards['ep'] = episode_rewards
        print("Saving V")
        np.save('Model_free/output/td_learning_V.npy', self.V)
        print("Saving statistics")
        with open('Model_free/statistics/TD_learning_statistics.pkl', 'wb') as f:
            pickle.dump(aggr_episode_rewards, f)

    def get_action_in_training(self, x):        
        if np.random.rand() < self.hyperparamters["exploration_factor"]: #exploration
            return np.random.choice(self.U)
        else: #exploitation
            possible_encoded_states = []
            for u in self.U:
                x_next = self.world.compute_next_state(x, u)
                possible_encoded_states.append(self.world.encode_state(x_next))
            possible_encoded_states = np.array(possible_encoded_states)

            bin_indeces, bin_encoded_states = self.find_bins(possible_encoded_states)
            if all(t == bin_indeces[0] for t in bin_indeces):
                return np.random.choice(self.U)

            #compute td target for each bin state
            td_targets = []
            for i, x in enumerate(self.world.decode_state(bin_encoded_states)):
                reward = self.world.reward(x, self.U[i])
                td_target = reward + self.hyperparamters["discount_factor"]*self.V[bin_indeces[i]]
                td_targets.append(td_target)
            td_targets = np.array(td_targets)
            
            #return action with higher td target
            return self.U[np.argmax(td_targets)]
        
    def get_action(self, x):        
        possible_encoded_states = []
        for u in self.U:
            x_encoded_next = self.world.compute_next_encoded_state(self.world.encode_state(x), u)
            possible_encoded_states.append(x_encoded_next)
        possible_encoded_states = np.array(possible_encoded_states)

        bin_indeces, bin_encoded_states = self.find_bins(possible_encoded_states)
        if all(t == bin_indeces[0] for t in bin_indeces):
            return np.random.choice(self.U)

        #compute td target for each bin state
        td_targets = []
        for i, x in enumerate(self.world.decode_state(bin_encoded_states)):
            reward = self.world.reward(x, self.U[i])
            td_target = reward + self.hyperparamters["discount_factor"]*self.V[bin_indeces[i]]
            td_targets.append(td_target)
        td_targets = np.array(td_targets)
        
        #return action with higher td target
        return self.U[np.argmax(td_targets)]
    
    def update_V(self, x_encoded, x_encoded_next, reward):
        bin_indeces, _ = self.find_bins(np.append([x_encoded], [x_encoded_next], axis=0))
        if bin_indeces[0] == self.bin_ref:
            print(x_encoded, self.world.x_ref, bin_indeces[0])
            self.V[bin_indeces[0]] = 0
            print("ref reached in training !!!!!!!!!!")
        else:
            V_new = reward + self.hyperparamters["discount_factor"]*self.V[bin_indeces[1]]
            self.V[bin_indeces[0]] = (1-self.hyperparamters["momentum"])*self.V[bin_indeces[0]] + self.hyperparamters["momentum"]*V_new
    
    def plot_V(self):
        if len(self.V.shape) == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Create grid of theta and theta_dot values
            theta, theta_dot = np.meshgrid(range(self.V.shape[0]), range(self.V.shape[1]))

            # Plot the 3D surface
            ax.plot_surface(theta, theta_dot, self.V, cmap='viridis')

            # Set labels and title
            ax.set_xlabel('theta')
            ax.set_ylabel('theta_dot')
            ax.set_zlabel('Value')
            ax.set_title('Value Function Map')

            # Show the plot
            plt.show()
        else:
            print("Plot enabled only for 2D value function !!!!!!!!!")

    def plot_statistics(self): 
        with open('Model_free/statistics/TD_learning_statistics.pkl', 'rb') as f:
            aggr_episode_rewards = pickle.load(f)
        plt.plot(aggr_episode_rewards['it'], aggr_episode_rewards['avg'], label="average rewards")
        plt.plot(STATS_EVERY + np.arange(len(aggr_episode_rewards['ep'])-STATS_EVERY), aggr_episode_rewards['ep'][STATS_EVERY:], label="all rewards", alpha=0.5)
        plt.legend(loc=4)
        plt.show()
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

from Utils import Environment

class TD_learning:

    def __init__(self, world: Environment, load_V = False):
        self.world = copy(world)

        self.hyperparamters = {
            "train_iterations": 150000,
            "max_steps": self.world.lparams.max_steps,
            "theta_dot_max_abs": 8,
            "action_max_abs": 2,
            "state_bins": (21, 21, 65),
            "action_bins": 17,
            "discount_factor": 0.95,
            "momentum": 0.1,
            "exploration_factor": 0.1
        }

        # compute state bins
        self.U = np.zeros(self.hyperparamters["action_bins"])
        du = 2*self.hyperparamters["action_max_abs"]/(self.U.shape[0]-1)
        for k in range(self.U.shape[0]):
            self.U[k] = -self.hyperparamters["action_max_abs"] + k*du

        if load_V:
            self.V = np.load('Model_free/output/td_learning_V.npy')
        else:
            self.V = np.random.random(self.hyperparamters["state_bins"])-1
        
        bin_ref, _ = self.find_bins(self.world.reference_state)
        self.bin_ref = bin_ref[0]

    def encode(self, states):
        new_states = np.insert(states, 0, 0, axis=1)
        new_states[:, 0] = np.cos(states[:, 0])
        new_states[:, 1] = np.sin(states[:, 1])
        return new_states
    
    def decode(self, states):
        new_states = states[:, 1:]
        new_states[:, 0] = np.arctan2(states[:, 1], states[:, 0])
        return new_states

    def find_bins(self, states):
        if len(states.shape) == 1:
            states = states.reshape((1, -1))
        states = self.encode(states)
        dcos = 2/(self.hyperparamters["state_bins"][0]-1)
        dsin = 2/(self.hyperparamters["state_bins"][1]-1)
        dtheta_dot = 2*self.hyperparamters["theta_dot_max_abs"]/(self.hyperparamters["state_bins"][2]-1)
        dX = np.array([dcos, dsin, dtheta_dot])
        X_min = np.array([-1., -1., -self.hyperparamters["theta_dot_max_abs"]])

        bin_indeces = np.round((states-X_min) / dX).astype(np.int32)
        bin_indeces[:, 0] = np.clip(bin_indeces[:, 0], 0, self.hyperparamters["state_bins"][0]-1)
        bin_indeces[:, 1] = np.clip(bin_indeces[:, 1], 0, self.hyperparamters["state_bins"][1]-1)
        bin_indeces[:, 2] = np.clip(bin_indeces[:, 2], 0, self.hyperparamters["state_bins"][2]-1)
        bin_states = self.decode(X_min + bin_indeces*dX)
        bin_indeces = list(map(tuple, bin_indeces.reshape(-1, 3)))
        return bin_indeces, bin_states
    
    def train(self):
        past_state = self.world.get_state()
        self.world.lparams.max_steps = self.hyperparamters["max_steps"]
        action = 0
        it = 0
        print(f"Training iteration {it+1}/{self.hyperparamters['train_iterations']}")
        while it < self.hyperparamters["train_iterations"]:
            action = self.get_action(past_state)
            new_state, reward, done = self.world.step(action)
            reward = self.update_reward(reward)
            self.update_V(past_state, new_state, reward)
            if done:
                it += 1
                if it < self.hyperparamters["train_iterations"]:
                    print(f"Training iteration {it+1}/{self.hyperparamters['train_iterations']}")
                    action = 0
                    past_state = self.world.reset()
            past_state = new_state.copy()
        print("Saving V")
        np.save('Model_free/output/td_learning_V.npy', self.V)

    def get_action(self, state, training=True):        
        if training and np.random.rand() < self.hyperparamters["exploration_factor"]: #exploration
            return np.random.choice(self.U)
        else: #exploitation
            possible_states = []
            for u in self.U:
                next_state = self.world.compute_next_state(state, u)
                possible_states.append(next_state)
            possible_states = np.array(possible_states)

            bin_indeces, bin_states = self.find_bins(possible_states)
            if all(t == bin_indeces[0] for t in bin_indeces):
                return np.random.choice(self.U)

            #compute td target for each bin state
            td_targets = []
            for i, state in enumerate(bin_states):
                reward = self.world.reward(state, self.U[i])
                td_target = reward + self.hyperparamters["discount_factor"]*self.V[bin_indeces[i]]
                td_targets.append(td_target)
            td_targets = np.array(td_targets)
            
            #return action with higher td target
            return self.U[np.argmax(td_targets)]
    
    def update_reward(self, reward):
        return reward
    
    def update_V(self, past_state, new_state, reward):
        states = np.append([past_state], [new_state], axis=0)
        bin_indeces, _ = self.find_bins(states)
        if bin_indeces[0] == self.bin_ref:
            print(past_state, self.world.reference_state, bin_indeces[0])
            self.V[bin_indeces[0]] = 0
            print("ref reached in training !!!!!!!!!!")
        else:
            V_new = reward + self.hyperparamters["discount_factor"]*self.V[bin_indeces[1]]
            self.V[bin_indeces[0]] = (1-self.hyperparamters["momentum"])*self.V[bin_indeces[0]] + self.hyperparamters["momentum"]*V_new
    
    # def plot_V(self):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Create grid of theta and theta_dot values
    #     theta, theta_dot = np.meshgrid(range(self.V.shape[0]), range(self.V.shape[1]))

    #     # Plot the 3D surface
    #     ax.plot_surface(theta, theta_dot, self.V, cmap='viridis')

    #     # Set labels and title
    #     ax.set_xlabel('theta')
    #     ax.set_ylabel('theta_dot')
    #     ax.set_zlabel('Value')
    #     ax.set_title('Value Function Map')

    #     # Show the plot
    #     plt.show()
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from copy import copy
# from tqdm import tqdm

# from Utils import Environment, Buffer

# class NN_Q(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(NN_Q, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, input_dim//2),
#             nn.ReLU(),
#             nn.Linear(input_dim//2, input_dim//4),
#             nn.ReLU(),
#             nn.Linear(input_dim//4, output_dim)
#         )

#     def forward(self, x):
#         return self.network(x)

# class NN_Q_learning:
#     def __init__(self, world: Environment, load_Q=False):
#         self.world = copy(world)
#         self.u_max_abs = world.mparams.u_max_abs

#         self.hyperparameters = {
#             "total_timesteps": 1000,
#             "n_steps": world.lparams.max_steps,
#             "epochs": 10,
#             "batch_size": 100, 
#             "action_bins": 10,
#             "discount_factor": 0.99,
#             "eps": 1.0,
#             "l_r": 1e-3
#         }

#         self.buffer = Buffer(buffer_size=self.hyperparameters["n_steps"])

#         # compute action bins
#         self.U = torch.zeros(self.hyperparameters["action_bins"])
#         du = 2 * self.u_max_abs / (self.U.shape[0] - 1)
#         for k in range(self.U.shape[0]):
#             self.U[k] = -self.u_max_abs + k * du

#         if load_Q:
#             self.Q = NN_Q(input_dim=3 + 1, output_dim=1)  # input_dim=3 (state) + 1 (action)
#             self.Q.load_state_dict(torch.load('Model_free/output/Q_learning_NN.pth'))
#         else:
#             self.Q = NN_Q(input_dim=3 + 1, output_dim=1)  # input_dim=3 (state) + 1 (action)
        
#         self.optimizer = optim.Adam(self.Q.parameters(), lr=self.hyperparameters["l_r"])
#         self.criterion = nn.MSELoss()
    
#     def learn(self):
#         for _ in tqdm(range(self.hyperparameters["total_timesteps"]), desc="Training Progress"):
#             self.buffer.reset()
#             self.collect_rollout()
#             self.train()
#             self.buffer.plot_rewards()
#             self.hyperparameters["eps"] *= 0.99

#         print("Saving Q")
#         torch.save(self.Q.state_dict(), 'Model_free/output/Q_learning_NN.pth')
    
#     def collect_rollout(self):
#         x = self.world.reset()
#         for _ in range(self.hyperparameters["n_steps"]):
#             action_index = self.get_action_index(x, deterministic=False)
#             x_next, reward, _ = self.world.step(self.U[action_index])
#             self.buffer.add(x, action_index, reward)
#             x = x_next.copy()
#         self.buffer.add_last_state(x)

#     def train(self):
#         for _ in range(self.hyperparameters["epochs"]):
#             for data in self.buffer.get(self.hyperparameters["batch_size"]):
#                 states = torch.tensor(data.states, dtype=torch.float32)
#                 action_indeces = torch.tensor(data.actions, dtype=torch.long)
#                 next_states = torch.tensor(data.next_states, dtype=torch.float32)
#                 rewards = torch.tensor(data.rewards, dtype=torch.float32)

#                 states_actions = torch.cat((states, self.U[action_indeces]), dim=1)
#                 q_value = self.Q(states_actions)

#                 # Expand next_states to be compatible with all actions
#                 next_states_expanded = next_states.unsqueeze(1).expand(-1, self.hyperparameters["action_bins"], -1)
#                 actions_expanded = torch.tensor(self.U).unsqueeze(0).unsqueeze(2).expand(self.hyperparameters["batch_size"], -1, -1)
                
#                 # Concatenate states and actions
#                 next_state_actions = torch.cat((next_states_expanded, actions_expanded), dim=2)
                
#                 # Compute Q-values for all state-action pairs
#                 next_q_values = self.Q(next_state_actions).squeeze(2)
                
#                 # Get the maximum Q-value for each next_state
#                 next_q_values_max, _ = next_q_values.max(dim=1)
                
#                 expected_q_value = rewards + self.hyperparameters["discount_factor"] * next_q_values_max

#                 loss = self.criterion(q_value.squeeze(1), expected_q_value)

#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()

#     def get_action_index(self, x, deterministic=True):
#         if not deterministic and np.random.rand() < self.hyperparameters["eps"]:  # exploration
#             return np.random.choice(self.hyperparameters["action_bins"])
        
#         state = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
#         state_expanded = state.expand(self.hyperparameters["action_bins"], -1)
#         actions_expanded = torch.tensor(self.U).unsqueeze(1)
#         state_actions = torch.cat((state_expanded, actions_expanded), dim=1)
#         return torch.argmax(self.Q(state_actions)
# )
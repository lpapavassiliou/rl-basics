import numpy as np
import torch
import matplotlib.pyplot as plt


class Buffer:

    def __init__(self, buffer_size, n_x=3, n_u=1):
        self.buffer_size = buffer_size
        self.n_x = n_x
        self.n_u = n_u

        self.pos = 0
        self.reset()
        self.cum_rewards = []
        self.average_rewards = []
        plt.figure(figsize=(10, 6))

    def reset(self):
        self.states = np.zeros((self.buffer_size, self.n_x), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_u), dtype=np.int64)
        self.next_states = np.zeros((self.buffer_size, self.n_x), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.bool_)
        self.pos = 0

    def size(self):
        """
        :return: The current size of the buffer
        """
        return min(self.pos, self.buffer_size)

    def sample(self, batch_size, type):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        batch_inds = np.random.randint(0, self.size(), size=batch_size)
        return self._get_samples(batch_inds, type)

    def _get_samples(self, batch_inds, type):
        if type == "torch":
            data = (
                torch.tensor(self.states[batch_inds], dtype=torch.float32),
                torch.tensor(self.actions[batch_inds], dtype=torch.int64),
                torch.tensor(self.next_states[batch_inds], dtype=torch.float32),
                torch.tensor(self.rewards[batch_inds], dtype=torch.float32),
                torch.tensor(self.dones[batch_inds], dtype=torch.bool),
            )
        else:
            data = (
                self.states[batch_inds],
                self.actions[batch_inds],
                self.next_states[batch_inds],
                self.rewards[batch_inds],
                self.dones[batch_inds],
            )
        return data

    def add(self, state, action, next_state, reward, done):
        self.states[self.pos] = np.array(state)
        self.actions[self.pos] = np.array(action)
        self.next_states[self.pos] = np.array(next_state)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.pos += 1

    def get(self, batch_size, type):
        indices = np.random.permutation(self.buffer_size)

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size], type)
            start_idx += batch_size

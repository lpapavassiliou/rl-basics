import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Generator


class Buffer:
    """
    Buffer class for storing and sampling experiences.

    Args:
        buffer_size (int): The maximum size of the buffer.
        n_x (int): The dimension of the state space (default is 3).
        n_u (int): The dimension of the action space (default is 1).

    Attributes:
        pos (int): Current position in the buffer.
        states (np.ndarray): Array to store states.
        actions (np.ndarray): Array to store actions.
        next_states (np.ndarray): Array to store next states.
        rewards (np.ndarray): Array to store rewards.
        dones (np.ndarray): Array to store done flags.
        cum_rewards (list): List to store cumulative rewards.
        average_rewards (list): List to store average rewards.
        plt (matplotlib.pyplot): Matplotlib plotting object.

    Methods:
        reset(): Reset the buffer to initial state.
        size(): Get the current size of the buffer.
        sample(batch_size, type): Sample elements from the buffer.
        add(state, action, next_state, reward, done): Add an experience to the buffer.
        get(batch_size, type): Generator to yield batches of samples.
    """

    def __init__(self, buffer_size: int, n_x: int = 3, n_u: int = 1) -> None:
        self.buffer_size = buffer_size
        self.n_x = n_x
        self.n_u = n_u

        self.pos = 0
        self.reset()
        self.cum_rewards = []
        self.average_rewards = []
        plt.figure(figsize=(10, 6))

    def reset(self) -> None:
        """
        Resets the buffer to its initial state.
        """
        self.states = np.zeros((self.buffer_size, self.n_x), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_u), dtype=np.int64)
        self.next_states = np.zeros((self.buffer_size, self.n_x), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size), dtype=np.bool_)
        self.pos = 0

    def size(self) -> int:
        """
        Get the current size of the buffer.

        Returns:
            int: The current size of the buffer. This is the minimum of the current position and the buffer size.
        """

        return min(self.pos, self.buffer_size)

    def sample(self, batch_size: int, type: str) -> tuple[torch.Tensor, ...]:
        """
        Sample elements from the buffer.

        Args:
            batch_size (int): The number of elements to sample.
            type (str): The type of samples to return. Can be "torch" or "numpy".

        Returns:
            tuple[torch.Tensor, ...]: A tuple of tensors containing the sampled elements.
                The tensors are of type torch.Tensor if type is "torch", otherwise they are of type numpy.ndarray.
                The tuple contains the following tensors:
                - states: A tensor of shape (batch_size, n_x) containing the sampled states.
                - actions: A tensor of shape (batch_size, n_u) containing the sampled actions.
                - next_states: A tensor of shape (batch_size, n_x) containing the sampled next states.
                - rewards: A tensor of shape (batch_size) containing the sampled rewards.
                - dones: A tensor of shape (batch_size) containing the sampled done flags.
        """

        batch_inds = np.random.randint(0, self.size(), size=batch_size)
        return self._get_samples(batch_inds, type)

    def _get_samples(self, batch_inds: np.ndarray, type: str) -> tuple[torch.Tensor, ...]:
        """
        Get samples from the buffer.

        Args:
            batch_inds (np.ndarray): The indices of the samples to retrieve from the buffer.
            type (str): The type of samples to return. Can be "torch" or "numpy".

        Returns:
            tuple[torch.Tensor, ...]: A tuple containing the sampled elements.
                - states (torch.Tensor): A tensor of shape (batch_size, n_x) containing the sampled states.
                - actions (torch.Tensor): A tensor of shape (batch_size, n_u) containing the sampled actions.
                - next_states (torch.Tensor): A tensor of shape (batch_size, n_x) containing the sampled next states.
                - rewards (torch.Tensor): A tensor of shape (batch_size) containing the sampled rewards.
                - dones (torch.Tensor): A tensor of shape (batch_size) containing the sampled done flags.
        """
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

    def add(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, reward: np.float32, done: np.bool_
    ) -> None:
        """
        Adds a new experience to the buffer.

        Args:
            state (np.ndarray): The state of the environment.
            action (np.ndarray): The action taken in the environment.
            next_state (np.ndarray): The next state of the environment.
            reward (np.float32): The reward obtained from the environment.
            done (np.bool_): Whether the episode is done.
        """
        self.states[self.pos] = np.array(state)
        self.actions[self.pos] = np.array(action)
        self.next_states[self.pos] = np.array(next_state)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.pos += 1

    def get(self, batch_size: int, type: str) -> Generator[tuple[torch.Tensor, ...]]:
        """
        Returns a generator that yields batches of samples from the buffer.

        Args:
            batch_size (int): The number of samples to yield in each iteration.
            type (str): The type of samples to return. Can be "torch" or "numpy".

        Yields:
            tuple[torch.Tensor, ...]: A tuple containing the sampled elements.
                - states (torch.Tensor): A tensor of shape (batch_size, n_x) containing the sampled states.
                - actions (torch.Tensor): A tensor of shape (batch_size, n_u) containing the sampled actions.
                - next_states (torch.Tensor): A tensor of shape (batch_size, n_x) containing the sampled next states.
                - rewards (torch.Tensor): A tensor of shape (batch_size) containing the sampled rewards.
                - dones (torch.Tensor): A tensor of shape (batch_size) containing the sampled done flags.
        """
        indices = np.random.permutation(self.buffer_size)

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size], type)
            start_idx += batch_size

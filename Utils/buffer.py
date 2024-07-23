import numpy as np
import matplotlib.pyplot as plt

class RolloutBufferSamples:

    def __init__(self, states, actions, next_states, rewards):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.rewards = rewards

class Buffer:

    def __init__(
        self,
        buffer_size,
        n_x = 3,
        n_u = 1
    ):
        self.buffer_size = buffer_size
        self.n_x = n_x
        self.n_u = n_u

        self.pos = 0
        self.reset()
        self.cum_rewards = []
        self.average_rewards = []
        plt.figure(figsize=(10, 6))

    def reset(self):
        self.states = np.zeros((self.buffer_size+1, self.n_x), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_u), dtype=np.int64)
        self.rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.pos = 0

    def size(self):
        """
        :return: The current size of the buffer
        """
        return np.min(self.pos, self.buffer_size)

    def sample(self, batch_size):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        batch_inds = np.random.randint(0, self.size(), size=batch_size)
        return self._get_samples(batch_inds)
    
    def _get_samples(self, batch_inds):
        data = (
            self.states[batch_inds],
            self.actions[batch_inds],
            self.states[batch_inds+1],
            self.rewards[batch_inds],
        )
        return RolloutBufferSamples(*data)

    def add(
        self,
        state,
        action,
        reward,
    ):  
        self.states[self.pos] = np.array(state)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.pos += 1

    def add_last_state(self, state):
        self.states[-1] = np.array(state)

    def get(self, batch_size):
        indices = np.random.permutation(self.buffer_size)

        start_idx = 0
        while start_idx < self.buffer_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def plot_rewards(self):
        self.cum_rewards.append(np.sum(self.rewards))
        if len(self.cum_rewards == 1000):
            self.average_rewards.append(np.mean(np.array(self.cum_rewards)))
            # Plot the discounted cumulative rewards
            plt.plot(np.arange(len(self.average_rewards), dtype=np.int64), self.average_rewards, color='blue')
            plt.xlabel('Iteration')
            plt.ylabel('Cumulative Rewards')
            plt.title('Cumulative Rewards')
            plt.grid(True)
            plt.draw()
            plt.pause(0.1)
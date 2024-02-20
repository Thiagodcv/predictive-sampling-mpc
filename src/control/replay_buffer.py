import random
import collections

import numpy as np
import torch

# n_ = normalized, d_ = difference
Transition = collections.namedtuple('Transition', ('n_state', 'n_action', 'd_n_state'))
device = 'cpu'


class ReplayBuffer:

    def __init__(self, state_dim, action_dim, max_size=10000):
        self.data = collections.deque([], maxlen=max_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_mean, self.state_var = np.zeros(self.state_dim), np.ones(self.state_dim)
        self.action_mean, self.action_var = np.zeros(self.action_dim), np.ones(self.action_dim)

    def push(self, state, action, next_state):
        """
        Push (s, a, s') tuple to replay memory where 'state', 'action', and 'next_state' are not yet normalized.
        """
        # Update means and variances
        self.state_mean = (self.__len__()*self.state_mean + state) / (self.__len__() + 1)
        self.action_mean = (self.__len__()*self.action_mean + action) / (self.__len__() + 1)
        self.state_var = ((self.__len__() - 1)*self.state_var +
                          np.square(state - self.state_mean))/(self.__len__() - 1)
        self.action_var = ((self.__len__() - 1)*self.action_var +
                           np.square(action - self.action_mean))/(self.__len__() - 1)

        # Normalize data
        n_state = np.diagflat(np.reciprocal(self.state_var)) @ (state - self.state_mean)
        n_action = np.diagflat(np.reciprocal(self.action_var)) @ (action - self.action_mean)
        n_next_state = np.diagflat(np.reciprocal(self.state_var)) @ (next_state - self.state_mean)
        d_n_state = n_next_state - n_state

        # Add zero-mean gaussian noise to tuple variables
        n_state += self.sample_state_gaussian()
        n_action += self.sample_action_gaussian()
        n_next_state += self.sample_state_gaussian()

        # Push normalized data into replay buffer
        transition = Transition(n_state, n_action, d_n_state)
        self.data.append(transition)

    def sample_state_gaussian(self):
        return np.random.multivariate_normal(mean=np.zeros(len(self.state_dim)),
                                             cov=0.01*np.eye(len(self.state_dim)),
                                             size=len(self.state_dim))

    def sample_action_gaussian(self):
        return np.random.multivariate_normal(mean=np.zeros(len(self.action_dim)),
                                             cov=0.01*np.eye(len(self.action_dim)),
                                             size=len(self.action_dim))

    def sample(self, batch_size):
        transitions = random.sample(self.data, batch_size)

        batch = Transition(*zip(*transitions))
        n_state = torch.tensor(batch.n_state, device=device, dtype=torch.float32)
        n_action = torch.tensor(batch.n_action, device=device, dtype=torch.float32)
        d_n_state = torch.tensor(batch.d_n_state, device=device, dtype=torch.float32)
        return n_state, n_action, d_n_state

    def __len__(self):
        return len(self.data)

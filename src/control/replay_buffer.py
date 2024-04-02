import random
import collections

import numpy as np

Transition = collections.namedtuple('Transition', ('state', 'action', 'next_state'))


class ReplayBuffer:

    def __init__(self, state_dim, action_dim, max_size=10000, normalize=False):
        self.rand_data = collections.deque([], maxlen=max_size)
        self.rl_data = collections.deque([], maxlen=max_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_mean, self.state_var = np.zeros(self.state_dim), np.ones(self.state_dim)
        self.action_mean, self.action_var = np.zeros(self.action_dim), np.ones(self.action_dim)
        self.normalize = normalize

    def push(self, state, action, next_state, rl):
        """
        Push (s, a, s') tuple to replay memory where 'state', 'action', and 'next_state' are not yet normalized.
        Parameters
        ----------
        state : np.array
        action : np.array
        next_state : np.array
        rl : bool
            True if 'a' in (s, a, s') is chosen by MPC (not random)
        """
        # Update means and variances
        if self.__len__() > 1 and self.normalize:
            self.state_mean = (self.__len__()*self.state_mean + state) / (self.__len__() + 1)
            self.action_mean = (self.__len__()*self.action_mean + action) / (self.__len__() + 1)
            self.state_var = ((self.__len__() - 1)*self.state_var +
                              np.square(state - self.state_mean))/(self.__len__() + 1 - 1)
            self.action_var = ((self.__len__() - 1)*self.action_var +
                               np.square(action - self.action_mean))/(self.__len__() + 1 - 1)

        # Push normalized data into replay buffer
        transition = Transition(state, action, next_state)
        if rl:
            self.rl_data.append(transition)
        else:
            self.rand_data.append(transition)

    def sample(self, batch_size, rl_prop=0):
        """
        Sample a batch of experiences from replay. Note that the third element returned is
        'next_state - state', NOT 'next_state'.

        Parameters
        ----------
        batch_size : int
            Number of tuples to sample from buffer
        rl_prop : float in [0, 1]
            Fraction of samples that come from rl_data (as opposed to rand_data)
        """

        # Get samples from random actions
        rand_transitions = random.sample(self.rand_data, int(np.ceil(batch_size * (1-rl_prop))))
        rand_batch = Transition(*zip(*rand_transitions))

        rand_state = np.array(rand_batch.state)
        rand_action = np.array(rand_batch.action)
        rand_next_state = np.array(rand_batch.next_state)

        if rl_prop > 0:
            # Get  samples from MPC actions
            rl_batch_size = int(np.min([np.floor(batch_size * rl_prop), len(self.rl_data)]))
            rl_transitions = random.sample(self.rl_data, rl_batch_size)
            rl_batch = Transition(*zip(*rl_transitions))

            rl_state = np.array(rl_batch.state)
            rl_action = np.array(rl_batch.action)
            rl_next_state = np.array(rl_batch.next_state)

            # Combine samples
            state = np.concatenate((rand_state, rl_state), axis=0)
            action = np.concatenate((rand_action, rl_action), axis=0)
            next_state = np.concatenate((rand_next_state, rl_next_state), axis=0)
        else:
            state = rand_state
            action = rand_action
            next_state = rand_next_state

        # Normalize data
        if self.normalize:
            n_state, n_action, d_n_state = self.normalize_tuple(state, action, next_state, batch_size)
            return n_state, n_action, d_n_state

        return state, action, next_state - state

    def normalize_tuple(self, state, action, next_state, batch_size):
        """
        Normalize state, action, and next_state - state. Assume covariance matrix of action
        and states have zero on the diagonal.
        """
        n_state = (state - self.state_mean) @ np.diagflat(np.reciprocal(np.sqrt(self.state_var)))
        n_action = (action - self.action_mean) @ np.diagflat(np.reciprocal(np.sqrt(self.action_var)))
        n_next_state = (next_state - self.state_mean) @ np.diagflat(np.reciprocal(np.sqrt(self.state_var)))
        d_n_state = n_next_state - n_state

        # Add zero-mean gaussian noise to tuple variables
        n_state += self.sample_state_gaussian(batch_size)
        n_action += self.sample_action_gaussian(batch_size)
        d_n_state += self.sample_state_gaussian(batch_size)

        return n_state, n_action, d_n_state

    def sample_state_gaussian(self, size):
        return np.random.multivariate_normal(mean=np.zeros(self.state_dim),
                                             cov=0.01*np.eye(self.state_dim),
                                             size=size)

    def sample_action_gaussian(self, size):
        return np.random.multivariate_normal(mean=np.zeros(self.action_dim),
                                             cov=0.01*np.eye(self.action_dim),
                                             size=size)

    def get_state_mean(self):
        return self.state_mean

    def get_state_var(self):
        return self.state_var

    def get_action_mean(self):
        return self.action_mean

    def get_action_var(self):
        return self.action_var

    def __len__(self):
        return len(self.rand_data) + len(self.rl_data)

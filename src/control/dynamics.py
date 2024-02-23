import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, normalize=False):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of states.
        action_dim : int
            Dimension of actions.
        normalize : boolean
            Normalize data.
        """
        # super(DynamicsModel, self).__init__()
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_var = None
        self.state_mean = None
        self.action_var = None
        self.action_mean = None
        self.normalize = normalize

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

    def forward_np(self, state, action):
        if self.normalize:
            n_state, n_action = self.normalize_state_action(state, action)
            x = np.concatenate((n_state, n_action))
            x_torch = torch.from_numpy(x).float()
            n_next_state = self.forward(x_torch).detach().numpy() + n_state
            output = n_next_state @ np.diagflat(self.state_var) + self.state_mean
        else:
            x = np.concatenate((state, action))
            x_torch = torch.from_numpy(x).float()
            output = self.forward(x_torch).detach().numpy() + state

        return output

    def update_state_var(self, state_var):
        self.state_var = state_var

    def update_state_mean(self, state_mean):
        self.state_mean = state_mean

    def update_action_var(self, action_var):
        self.action_var = action_var

    def update_action_mean(self, action_mean):
        self.action_mean = action_mean

    def normalize_state_action(self, state, action):
        n_state = (state - self.state_mean) @ np.diagflat(np.reciprocal(self.state_var))
        n_action = (action - self.action_mean) @ np.diagflat(np.reciprocal(self.action_var))
        return n_state, n_action

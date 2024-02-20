import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of states.
        action_dim: int
            Dimension of actions.
        """
        # super(DynamicsModel, self).__init__()
        super().__init__()
        self.flatten = nn.Flatten()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, x):
        # x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output

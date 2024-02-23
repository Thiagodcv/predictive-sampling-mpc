import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, replay_buffer=None):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of states.
        action_dim: int
            Dimension of actions.
        replay_buffer : ReplayBuffer
            The replay buffer with which the dynamics model is trained. Only pass in the
            replay buffer if you want to train the model with normalized data.
        """
        # super(DynamicsModel, self).__init__()
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.replay_buffer = replay_buffer

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    # def forward(self, x):
    #     output = self.linear_relu_stack(x)
    #     return output

    def forward_np(self, state, action):
        x = np.concatenate((state, action))
        x_torch = torch.from_numpy(x).float()
        output = self.forward(x_torch).detach().numpy() + state
        if self.replay_buffer is not None:
            output = (output @ np.diagflat(self.replay_buffer.get_state_var()) +
                      self.replay_buffer.get_state_mean())
        return output

import numpy as np
import torch


class MPC:
    """
    A class containing several different sampling-based MPC algorithms.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : torch.nn.Module
        """
        self.model = model

    def random_shooting(self, state, past_actions, num_traj, gamma, horizon, reward):
        """
        Parameters
        ----------
        state: torch.Tensor
        past_actions: torch.Tensor
            A list containing the past three actions taken by the agent.
        num_traj: int
            Number of trajectories to sample.
        gamma: float
            Discount factor.
        horizon: int
            Number of steps optimized over by the MPC controller.
        reward: function
            The instantaneous reward given at each timestep.

        Return
        ------
        torch.Tensor: The first action in the optimal sequence of actions.
        """
        # Sample actions
        mean_action = torch.mean(past_actions, dim=0)
        action_seqs = (torch.distributions.MultivariateNormal(mean_action, torch.eye(len(mean_action))).
                       sample(sample_shape=torch.Size([num_traj, horizon])))

        # Evaluate action sequences
        rets = torch.zeros(num_traj)
        for t in range(horizon):
            for seq in range(num_traj):
                rets[seq] = (gamma ** t) * reward(state, action_seqs[seq, t, :])
                next_state = self.model.forward(torch.cat((state, action_seqs[seq, t, :]))) + state
                state = next_state

        # Return first action of optimal sequence
        opt_seq_idx = torch.argmax(rets)
        return action_seqs[opt_seq_idx, 0, :]

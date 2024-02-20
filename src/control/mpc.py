import numpy as np
import torch


class MPC:
    """
    A class containing several different sampling-based MPC algorithms.
    """

    def __init__(self, model, num_traj, gamma, horizon, reward):
        """
        Parameters
        ----------
        model : torch.nn.Module
        num_traj: int
            Number of trajectories to sample.
        gamma: float
            Discount factor.
        horizon: int
            Number of steps optimized over by the MPC controller.
        reward: function
            The instantaneous reward given at each timestep.
        """
        self.model = model
        self.num_traj = num_traj
        self.gamma = gamma
        self.horizon = horizon
        self.reward = reward

    def random_shooting(self, state, past_action_mean):
        """
        Parameters
        ----------
        state: np.array
        past_action_mean: np.array
            The mean of the last 3 actions

        Return
        ------
        torch.Tensor: The first action in the optimal sequence of actions.
        """
        # Sample actions
        state = torch.from_numpy(state).float()
        mean_action = torch.from_numpy(past_action_mean).float()
        action_seqs = (torch.distributions.MultivariateNormal(mean_action, torch.eye(len(mean_action))).
                       sample(sample_shape=torch.Size([self.num_traj, self.horizon])))

        # Evaluate action sequences
        rets = torch.zeros(self.num_traj)
        for t in range(self.horizon):
            for seq in range(self.num_traj):
                rets[seq] = (self.gamma ** t) * self.reward(state, action_seqs[seq, t, :])
                input = torch.cat((state, action_seqs[seq, t, :]))
                next_state = self.model.forward(input) + state
                state = next_state

        # Return first action of optimal sequence
        opt_seq_idx = torch.argmax(rets)
        return action_seqs[opt_seq_idx, 0, :]

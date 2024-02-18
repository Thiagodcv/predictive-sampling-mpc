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
        state: np.array
        past_actions: list
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
        np.array: The first action in the optimal sequence of actions.
        """
        # Sample actions
        mean_action = np.mean(past_actions)
        action_seqs = np.random.normal(loc=mean_action, scale=1, size=(num_traj, horizon))

        # Evaluate action sequences
        rets = np.zeros(shape=num_traj)
        for t in range(horizon):
            for seq in range(num_traj):
                rets[seq] = (gamma ** t) * reward(state, action_seqs[seq, t])
                next_state = self.model(state, action_seqs[seq, t])
                state = next_state

        # Return first action of optimal sequence
        opt_seq_idx = np.argmax(rets)
        return action_seqs[opt_seq_idx, 0]

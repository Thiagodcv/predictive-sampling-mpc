import numpy as np


class MPC:
    """
    A class containing several different sampling-based MPC algorithms.
    """

    def __init__(self, model, num_traj, gamma, horizon, reward, terminate=None):
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
        terminate : function
             For a given (s, a, t) tuple returns true if episode has ended.
        """
        self.model = model
        self.num_traj = num_traj
        self.gamma = gamma
        self.horizon = horizon
        self.reward = reward
        self.terminate = terminate

    def random_shooting(self, state0):
        """
        Parameters
        ----------
        state0: np.array

        Return
        ------
        np.array: The first action in the optimal sequence of actions.
        """
        # Sample actions
        action_seqs = np.random.uniform(low=-3.0, high=3.0, size=(self.num_traj, self.horizon, 1))

        # Evaluate action sequences
        rets = np.zeros(self.num_traj)
        for seq in range(self.num_traj):
            state = np.copy(state0)
            for t in range(self.horizon):
                rets[seq] = (self.gamma ** t) * self.reward(state, action_seqs[seq, t, :])
                if self.terminate is not None and self.terminate(state, action_seqs[seq, t, :], t):
                    break
                input = np.concatenate((state, action_seqs[seq, t, :]))
                next_state = self.model.forward_np(input) + state
                state = next_state

        # Return first action of optimal sequence
        opt_seq_idx = np.argmax(rets)
        return action_seqs[opt_seq_idx, 0, :]

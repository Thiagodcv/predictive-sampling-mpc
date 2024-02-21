import numpy as np
import torch


class MPC:
    """
    A class containing several different sampling-based MPC algorithms.
    """

    def __init__(self, model, num_traj, gamma, horizon, reward, terminate=None, device=torch.device('cpu')):
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
        device: torch.device
        """
        self.model = model
        self.num_traj = num_traj
        self.gamma = gamma
        self.horizon = horizon
        self.reward = reward
        self.terminate = terminate
        self.device = device

    def random_shooting(self, state, past_action_mean):
        """
        Parameters
        ----------
        state: np.array
        past_action_mean: np.array
            The mean of the last 3 actions

        Return
        ------
        np.array: The first action in the optimal sequence of actions.
        """
        # Sample actions
        state0 = torch.from_numpy(state).float().to(self.device)
        # mean_action = torch.from_numpy(past_action_mean).float()
        # action_seqs = (torch.distributions.MultivariateNormal(mean_action, torch.eye(len(mean_action))).
        #                sample(sample_shape=torch.Size([self.num_traj, self.horizon]))).to(self.device)

        action_seqs = torch.FloatTensor(self.num_traj, self.horizon, 1).uniform_(-3, 3).to(self.device)

        # Evaluate action sequences
        rets = torch.zeros(self.num_traj).to(self.device)
        for seq in range(self.num_traj):
            state = state0.detach().clone().to(self.device)
            for t in range(self.horizon):
                rets[seq] = (self.gamma ** t) * self.reward(state, action_seqs[seq, t, :])
                if self.terminate is not None and self.terminate(state, action_seqs[seq, t, :], t):
                    break
                input = torch.cat((state, action_seqs[seq, t, :]))
                next_state = self.model.forward(input) + state
                state = next_state

        # Return first action of optimal sequence
        opt_seq_idx = torch.argmax(rets)
        return action_seqs[opt_seq_idx, 0, :].cpu().detach().numpy()

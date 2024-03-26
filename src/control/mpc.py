import numpy as np
import multiprocessing


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
        self.past_actions = []

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
        # action_seqs = np.random.binomial(n=1, p=0.5, size=(self.num_traj, self.horizon, 1))  # cartpole
        action_seqs = np.random.uniform(low=-10, high=10, size=(self.num_traj, self.horizon, 1))  # pendulum

        # Multiprocessing
        # TODO: Figure out if there's any way this can be fixed. If set num_workers=1 still slow?
        pool = multiprocessing.Pool(8)

        # Evaluate action sequences
        rets = np.zeros(self.num_traj)
        for seq in range(self.num_traj):
            # rets[seq] = self.do_rollout(state0, action_seqs[seq, :, :])
            res = pool.apply_async(self.do_rollout, args=(state0, action_seqs[seq, :, :]))
            rets[seq] = res.get()

        pool.close()
        pool.join()

        # Return first action of optimal sequence
        opt_seq_idx = np.argmax(rets)
        opt_action = action_seqs[opt_seq_idx, 0, :]
        self.append_past_action(opt_action)
        return opt_action

    def do_rollout(self, state0, action_seq):
        """
        Parameters
        ----------
        state0: np.ndarray
            First state
        action_seq: np.ndarray
            array of actions

        Return
        ------
        float: the rollout return
        """
        state = np.copy(state0)
        ret = 0
        for t in range(self.horizon):
            ret += (self.gamma ** t) * self.reward(state, action_seq[t, :])
            if self.terminate is not None and self.terminate(state, action_seq[t, :], t):
                break
            next_state = self.model.forward_np(state, action_seq[t, :])
            state = next_state

        return ret

    def append_past_action(self, action):
        """
        Append an action to a list of past optimal actions taken by the MPC.
        Parameters
        ----------
        action : np.array
            An action taken by the MPC.
        """
        self.past_actions.append(action)

    def empty_past_action_list(self):
        self.past_actions = []

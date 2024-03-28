import numpy as np
from multiprocessing.pool import ThreadPool
import multiprocessing
import ray
from src.control.dynamics import forward_np_static


class MPC:
    """
    A class containing several different sampling-based MPC algorithms.
    """

    def __init__(self, model, num_traj, gamma, horizon, reward, multithreading, terminate=None):
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
        multithreading: bool
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
        self.multithreading = multithreading

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

        # Evaluate action sequences
        if not self.multithreading:
            rets = np.zeros(self.num_traj)
            for seq in range(self.num_traj):
                rets[seq] = self.do_rollout(state0, action_seqs[seq, :, :])

        else:
            nn_params_ref = ray.put(self.model.create_nn_params())
            mpc_params_ref = ray.put({'gamma': self.gamma, 'horizon': self.horizon})
            action_seqs_ref = ray.put(action_seqs)

            rets_ref = []
            for seq in range(self.num_traj):
                rets_ref.append(do_rollout_static.remote(nn_params_ref, mpc_params_ref, state0, action_seqs_ref, seq))
            rets = ray.get(rets_ref)

            del rets_ref
            del nn_params_ref
            del mpc_params_ref
            del action_seqs_ref

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


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def reward(state, action):
    th = state[0]
    thdot = state[1]
    u = action
    return - (angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2))


@ray.remote
def do_rollout_static(nn_params, mpc_params, state0, action_seq, seq_num):
    """
    Parameters
    ----------
    nn_params : dict
    mpc_params : dict
    state0 : np.ndarray
    action_seq : np.ndarray
    seq_num : int
    """
    horizon = mpc_params['horizon']
    gamma = mpc_params['gamma']

    state = np.copy(state0)
    ret = 0
    for t in range(horizon):
        ret += (gamma ** t) * reward(state, action_seq[seq_num, t, :])
        # if self.terminate is not None and self.terminate(state, action_seq[t, :], t):
        #     break
        next_state = forward_np_static(nn_params, state, action_seq[seq_num, t, :])
        state = next_state
    return ret

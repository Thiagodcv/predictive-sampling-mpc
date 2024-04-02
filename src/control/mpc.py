import numpy as np
import ray
from src.control.dynamics import forward_np_static
import itertools


class MPC:
    """
    A class implementation of a predictive sampling MPC.
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
        self.past_trajectory = None

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
        if self.past_trajectory is None:
            self.past_trajectory = np.zeros(shape=(self.num_traj, self.horizon, 8))
            return self.past_trajectory[0, 0, :]
        else:
            action_seqs = self.past_trajectory + np.random.normal(loc=0, scale=1.0,
                                                                  size=(self.num_traj, self.horizon, 8))
            action_seqs = np.clip(action_seqs, -1.0, 1.0)

        # Evaluate action sequences
        if not self.multithreading:
            rets = np.zeros(self.num_traj)
            for seq in range(self.num_traj):
                rets[seq] = self.do_rollout(state0, action_seqs[seq, :, :])

        else:
            nn_params_ref = ray.put(self.model.create_nn_params())
            mpc_params_ref = ray.put({'gamma': self.gamma, 'horizon': self.horizon})
            action_seqs_ref = ray.put(action_seqs)
            reward_ref = ray.put(self.reward)
            terminate_ref = ray.put(self.terminate)
            state0_ref = ray.put(state0)

            rets_ref = []
            seqs = list(range(self.num_traj))
            batch_size = int(self.num_traj / 16)
            for i in range(0, self.num_traj, batch_size):
                batch = seqs[i:i+batch_size]
                rets_ref.append(do_batch_rollout_static.remote(nn_params_ref, mpc_params_ref, reward_ref,
                                                               terminate_ref, state0_ref, action_seqs_ref, batch))
            # rets = ray.get(rets_ref)
            rets = list(itertools.chain(*ray.get(rets_ref)))

            del rets_ref
            del nn_params_ref
            del mpc_params_ref
            del action_seqs_ref
            del reward_ref
            del terminate_ref
            del state0_ref

        # Return first action of optimal sequence
        opt_seq_idx = np.argmax(rets)
        self.past_trajectory = action_seqs[opt_seq_idx, :, :]
        opt_action = action_seqs[opt_seq_idx, 0, :]
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

    def empty_past_trajectory(self):
        self.past_trajectory = None


def do_rollout_static(nn_params, mpc_params, reward, terminate, state0, action_seq, seq_num):
    """
    Parameters
    ----------
    nn_params : dict
    mpc_params : dict
    reward : function
    terminate : function
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
        if terminate is not None and terminate(state, action_seq[t, :], t):
            break
        next_state = forward_np_static(nn_params, state, action_seq[seq_num, t, :])
        state = next_state
    return ret


@ray.remote
def do_batch_rollout_static(nn_params, mpc_params, reward, terminate, state0, action_seq, batch_seq_num):
    """
    Parameters
    ----------
    nn_params : dict
    mpc_params : dict
    reward : function
    terminate : function
    state0 : np.ndarray
    action_seq : np.ndarray
    batch_seq_num : list of int
    """
    return [do_rollout_static(nn_params, mpc_params, reward, terminate, state0, action_seq, idx)
            for idx in batch_seq_num]

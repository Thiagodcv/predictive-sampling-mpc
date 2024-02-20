from unittest import TestCase
from src.control.replay_buffer import ReplayBuffer
import numpy as np


class TestReplayBuffer(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_replay_buffer(self):
        """
        A simple test.
        """
        state1 = np.array([1., 1.])
        state2 = np.array([2., 2.])
        state3 = np.array([3., 3.])
        state4 = np.array([4., 4.])
        state5 = np.array([5., 5.])
        action1 = np.array([0.5, 0.5])
        action2 = np.array([1., 1.])
        action3 = np.array([1.5, 1.5])
        action4 = np.array([2., 2.])

        replay_buffer = ReplayBuffer(state_dim=2, action_dim=2, normalize=False)
        replay_buffer.push(state1, action1, state2)
        replay_buffer.push(state2, action2, state3)
        replay_buffer.push(state3, action3, state4)
        replay_buffer.push(state4, action4, state5)

        n_state, n_action, d_n_state = replay_buffer.sample(batch_size=4)
        print(n_state)

    def test_get_last_3_actions_mean(self):
        """
        A simple test.
        """
        state1 = np.array([1., 1.])
        state2 = np.array([2., 2.])
        state3 = np.array([3., 3.])
        state4 = np.array([4., 4.])
        state5 = np.array([5., 5.])
        action1 = np.array([0.5, 0.5])
        action2 = np.array([1., 3.])
        action3 = np.array([2., 3.])
        action4 = np.array([3., 6.])

        replay_buffer = ReplayBuffer(state_dim=2, action_dim=2, normalize=False)
        replay_buffer.push(state1, action1, state2)
        replay_buffer.push(state2, action2, state3)
        replay_buffer.push(state3, action3, state4)
        replay_buffer.push(state4, action4, state5)

        print(replay_buffer.get_last_3_actions_mean())

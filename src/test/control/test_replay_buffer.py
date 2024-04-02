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
        replay_buffer.push(state1, action1, state2, False)
        replay_buffer.push(state2, action2, state3, False)
        replay_buffer.push(state3, action3, state4, False)
        replay_buffer.push(state4, action4, state5, False)

        n_state, n_action, d_n_state = replay_buffer.sample(batch_size=4)
        print(n_state)

    def test_rand_rl_mix(self):
        replay_buffer = ReplayBuffer(state_dim=2, action_dim=1, normalize=False)
        for i in range(50):
            replay_buffer.push(np.array([1., 1.]), 1,
                               np.array([1.5, 1.5]), False)

        for i in range(50):
            replay_buffer.push(np.array([2., 2.]), 2,
                               np.array([2.2, 2.2]), True)

        s, a, n_s = replay_buffer.sample(batch_size=50, rl_prop=0.2)

        epsilon = 1e-5
        for i in range(40):
            self.assertTrue(np.linalg.norm(np.array([1., 1.]) - s[i, :]) < epsilon)
            self.assertTrue(np.linalg.norm(1 - a[i]) < epsilon)
            self.assertTrue(np.linalg.norm(np.array([0.5, 0.5]) - n_s[i, :]) < epsilon)

        for i in range(40, 50):
            self.assertTrue(np.linalg.norm(np.array([2., 2.]) - s[i, :]) < epsilon)
            self.assertTrue(np.linalg.norm(2 - a[i]) < epsilon)
            self.assertTrue(np.linalg.norm(np.array([0.2, 0.2]) - n_s[i, :]) < epsilon)

        self.assertTrue(len(a) == 50)

from unittest import TestCase
from src.control.dynamics import DynamicsModel
from control.mpc import MPC
from control.mbrl import MBRLLearner
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


class TestMBRL(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mbrl(self):
        """
        Test to see if can run an example without crashing.
        """
        state_dim = 4
        action_dim = 1
        env = gym.make("InvertedPendulum-v4")
        num_episodes = 4000
        episode_len = 100
        batch_size = 32

        def reward(state, action):
            return 1

        def terminate(state, action, t):
            # If episode is at t>=1000, terminate episode
            if t >= 1000:
                return True
            # If absolute value of vertical angle between pole and cart is greater than 0.2,
            # terminate episode
            elif state[1].item() > 0.2 or state[1].item() < -0.2:
                return True
            else:
                return False

        learner = MBRLLearner(state_dim=state_dim, action_dim=action_dim, env=env,
                              num_episodes=num_episodes, episode_len=episode_len, reward=reward,
                              terminate=terminate, batch_size=batch_size)
        learner.train()

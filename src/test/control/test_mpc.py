from unittest import TestCase
from src.control.dynamics import DynamicsModel
from control.mpc import MPC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestMPC(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_nn_random_shooting(self):
        """
        Test to see if can run an example without crashing.
        """
        state_dim = 2
        action_dim = 2

        # input params
        state0 = np.array([1., 1.])
        past_action_mean = np.array([0.5, 0.5])
        num_traj = 10
        gamma = 0.99
        horizon = 5

        def reward(state, action):
            return -torch.matmul(state, state) - torch.matmul(action, action)

        model = DynamicsModel(state_dim, action_dim)
        mpc = MPC(model, num_traj, gamma, horizon, reward)
        mpc.random_shooting(state0, past_action_mean)

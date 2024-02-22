from unittest import TestCase
from src.control.dynamics import DynamicsModel
from control.mpc import MPC
import gymnasium as gym
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
        PATH = "C:/Users/thiag/Git/random-shooting-mpc/models/good_model.pt"
        state_dim = 4
        action_dim = 1

        # input params
        num_traj = 1000
        gamma = 0.99
        horizon = 10

        env = gym.make("InvertedPendulum-v4")  # ,render_mode="human")
        model = DynamicsModel(state_dim, action_dim)
        model.load_state_dict(torch.load(PATH))

        def reward(state, action):
            return 1

        def terminate(state, action, t):
            # If episode is at t>=1000, terminate episode
            if t >= 1000:
                return True
            # If absolute value of vertical angle between pole and cart is greater than 0.2,
            # terminate episode
            elif state[1] > 0.2 or state[1] < -0.2:
                return True
            else:
                return False

        mpc = MPC(model, num_traj, gamma, horizon, reward, terminate)

        state, _ = env.reset()
        for t in range(100):
            print(t)
            action = mpc.random_shooting(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            state = next_state

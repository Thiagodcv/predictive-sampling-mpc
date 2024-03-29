from unittest import TestCase
from src.control.dynamics import DynamicsModel
from src.constants import MODELS_PATH
from control.mpc import MPC
import gymnasium as gym
import numpy as np
import torch
import os
import time


class TestMPC(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_nn_random_shooting(self):
    #     """
    #     Test to see if can run an example without crashing.
    #     """
    #     PATH = "C:/Users/thiag/Git/random-shooting-mpc/models/good_model.pt"
    #     state_dim = 4
    #     action_dim = 1
    #
    #     # input params
    #     num_traj = 1000
    #     gamma = 0.99
    #     horizon = 10
    #
    #     env = gym.make("InvertedPendulum-v4")  # ,render_mode="human")
    #     model = DynamicsModel(state_dim, action_dim)
    #     model.load_state_dict(torch.load(PATH))
    #
    #     def reward(state, action):
    #         return 1
    #
    #     def terminate(state, action, t):
    #         # If episode is at t>=1000, terminate episode
    #         if t >= 1000:
    #             return True
    #         # If absolute value of vertical angle between pole and cart is greater than 0.2,
    #         # terminate episode
    #         elif state[1] > 0.2 or state[1] < -0.2:
    #             return True
    #         else:
    #             return False
    #
    #     mpc = MPC(model, num_traj, gamma, horizon, reward, terminate)
    #
    #     state, _ = env.reset()
    #     for t in range(100):
    #         print(t)
    #         action = mpc.random_shooting(state)
    #         next_state, reward, terminated, truncated, _ = env.step(action)
    #         if terminated or truncated:
    #             break
    #         state = next_state

    def test_find_discrete_optimal_action(self):
        """
        Test to see if random shooting mpc can find optimal set of actions.
        Note that the action space = {0, 1}.
        """
        # input params
        num_traj = 100
        gamma = 1e-5
        horizon = 20

        # Defining the model just to get the code to run
        state_dim = 4
        action_dim = 1
        model = DynamicsModel(state_dim, action_dim)
        model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "testv8.pt")))

        # Defining the reward
        optimal_action = 1

        def reward(state, action):
            return -(action.item() - optimal_action)**2

        mpc = MPC(model, num_traj, gamma, horizon, reward)

        state_dummy = np.zeros(state_dim)
        for i in range(100):
            print("action: ", mpc.random_shooting(state_dummy))

    def test_termination_function(self):
        """
        Test to see if random shooting mpc avoid outputting "1" if environment
        terminates when action=1 is taken.
        """
        # input params
        num_traj = 100
        gamma = 1e-5
        horizon = 20

        # Defining the model just to get the code to run
        state_dim = 4
        action_dim = 1
        model = DynamicsModel(state_dim, action_dim)
        model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "testv8.pt")))

        def reward(state, action):
            return 1

        def terminate(state, action, t):
            if action == 1:
                return True
            else:
                return False

        mpc = MPC(model, num_traj, gamma, horizon, reward, terminate)

        state_dummy = np.zeros(state_dim)
        for i in range(100):
            print("action: ", mpc.random_shooting(state_dummy))

    def test_random_sampling_time(self):
        start_time = time.time()
        for i in range(200):
            # action_seqs = np.random.uniform(low=-10, high=10, size=(7000, 15, 1))
            action_seqs = np.random.standard_normal(size=(2000, 15, 6))
        print("--- %s seconds ---" % (time.time() - start_time))

    def test_call_times(self):
        start_time = time.time()

        a = torch.zeros(size=(3, 1))
        b = np.ones(3)
        mat1 = np.identity(3)
        mat2 = np.identity(3)
        for i in range(200):
            for seq in range(200):
                for t in range(15):
                    np.matmul(mat1, mat2)
                    # np.reciprocal(b)
                    # mat1 @ mat2
                    # np.sqrt(b)
                    # a.detach().numpy()

        print("--- %s seconds ---" % (time.time() - start_time))

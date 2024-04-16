import numpy as np
import torch
import gymnasium as gym
from src.control.mpc import MPC
from src.control.dynamics import DynamicsModel
from src.control.mbrl import MBRLLearner
from src.constants import MODELS_PATH
import os
from multiprocessing.pool import ThreadPool
import math
import multiprocessing
import ray

import time
import cProfile


def reward(state, action):
    x_vel = state[13]
    # return x_vel + 0.5 - 0.005 * np.linalg.norm(action/150) ** 2
    return x_vel + 0.5 - 0.5 / 8 * np.linalg.norm(action) ** 2


def terminate(state, action, t):
    return state[0] < 0.2 or state[0] > 1.0


def ant():
    state_dim = 27
    action_dim = 8
    episode_len = 200
    env = gym.make("Ant-v4", render_mode="human")

    save_name = "ant-task-4-9-run0"
    dir_path = os.path.join(MODELS_PATH, save_name)

    model = DynamicsModel(state_dim, action_dim, normalize=True)
    model.load_state_dict(torch.load(os.path.join(dir_path, save_name + '.pt')))

    num_traj = 1024  # Make sure it's divisible by num_workers
    gamma = 0.99
    horizon = 15

    # Ray stuff
    num_workers = multiprocessing.cpu_count()
    print("Number of workers: ", num_workers)
    ray.init(num_cpus=num_workers)

    mpc = MPC(model, num_traj, gamma, horizon, reward, terminate, True)

    start_time = time.time()
    MBRLLearner.static_eval_model(env, episode_len, mpc, gamma, reward_func=reward, terminate_func=terminate)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # cProfile.run('pendulum()')
    ant()

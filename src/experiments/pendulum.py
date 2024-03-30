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


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def reward(state, action):
    th = state[0]
    thdot = state[1]
    u = action
    return - (angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2))


def pendulum():
    state_dim = 2
    action_dim = 1
    episode_len = 200
    env = gym.make("Pendulum-v1", render_mode="human")

    model = DynamicsModel(state_dim, action_dim, normalize=True)
    model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "pend_demo_256.pt")))

    num_traj = 200
    gamma = 0.95
    horizon = 15

    # Ray stuff
    num_workers = multiprocessing.cpu_count()
    print("Number of workers: ", num_workers)
    ray.init(num_cpus=num_workers)

    mpc = MPC(model, num_traj, gamma, horizon, reward, multithreading=True)

    start_time = time.time()
    MBRLLearner.static_eval_model(env, episode_len, mpc, gamma)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # cProfile.run('pendulum()')
    pendulum()
    
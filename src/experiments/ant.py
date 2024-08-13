from src.control.mpc import MPC
from src.control.dynamics import DynamicsModel
from src.control.mbrl import MBRLLearner
from src.constants import MODELS_PATH, RECORDINGS_PATH
import numpy as np
import torch
import gymnasium as gym
import os
import multiprocessing
import ray
import time
from gymnasium.wrappers import RecordVideo
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

    save_name = "ant-task-4-9-run2"
    dir_path = os.path.join(MODELS_PATH, save_name)

    # For recording. If want to record, and need to set render_mode = "rgb_array".
    # env = gym.make("Ant-v4", render_mode="rgb_array")
    # env = RecordVideo(env, video_folder=RECORDINGS_PATH, name_prefix="Demo", episode_trigger=lambda x: True)

    env = gym.make("Ant-v4", render_mode="human")

    model = DynamicsModel(state_dim, action_dim, normalize=True)
    model.load_state_dict(torch.load(os.path.join(dir_path, save_name + '.pt'), weights_only=True))

    num_traj = 1024  # Make sure it's divisible by num_workers
    gamma = 0.99
    horizon = 10

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

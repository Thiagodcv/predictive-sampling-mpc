from control.mbrl import MBRLLearner
import gymnasium as gym
import numpy as np
import multiprocessing
import ray


def run_mbrl():
    state_dim = 2
    action_dim = 1
    env = gym.make("Pendulum-v1")
    num_episodes = 2020
    episode_len = 200
    batch_size = 256
    num_rand_eps = 2000  # Right now have it set to only supervised learning

    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reward(state, action):
        th = state[0]
        thdot = state[1]
        u = action
        return - (angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2))

    # Ray stuff
    num_workers = multiprocessing.cpu_count()
    print("Number of workers: ", num_workers)
    ray.init(num_cpus=num_workers)

    learner = MBRLLearner(state_dim=state_dim, action_dim=action_dim, env=env,
                          num_episodes=num_episodes, episode_len=episode_len, reward=reward,
                          terminate=None, batch_size=batch_size, num_rand_eps=num_rand_eps,
                          save_name="demo-4-1-2024", normalize=True)
    learner.train()


if __name__ == "__main__":
    run_mbrl()

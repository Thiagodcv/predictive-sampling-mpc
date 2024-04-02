from control.mbrl import MBRLLearner
import gymnasium as gym
import numpy as np
import multiprocessing
import ray


def run_mbrl():
    state_dim = 27
    action_dim = 8
    env = gym.make("Ant-v4")
    num_episodes = 2100
    episode_len = 200
    batch_size = 256
    num_rand_eps = 2000

    def reward(state, action):
        x_vel = state[13]
        # return x_vel + 0.5 - 0.005 * np.linalg.norm(action/150)**2
        return x_vel + 0.5 - 0.5/8 * np.linalg.norm(action) ** 2

    def terminate(state, action, t):
        return state[0] < 0.2 or state[0] > 1.0

    # Ray stuff
    num_workers = multiprocessing.cpu_count()
    print("Number of workers: ", num_workers)
    ray.init(num_cpus=num_workers)

    learner = MBRLLearner(state_dim=state_dim, action_dim=action_dim, env=env,
                          num_episodes=num_episodes, episode_len=episode_len, reward=reward,
                          terminate=terminate, batch_size=batch_size, num_rand_eps=num_rand_eps,
                          save_name="ant-4-2", normalize=True)
    learner.train()


if __name__ == "__main__":
    run_mbrl()

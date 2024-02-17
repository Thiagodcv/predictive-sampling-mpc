import numpy as np
import torch


class MBRLLearner:
    """
    A class for training a model-based reinforcement learning agent.
    """

    def __init__(self):
        pass

    def train(self, env, num_episodes, episode_len):
        replay_buffer = []

        for ep in range(num_episodes):
            # Train model on replay buffer
            o, _ = env.reset()
            for t in range(episode_len):
                action = 0
                next_o, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
                replay_buffer.append((o, action, next_o))
            env.close()

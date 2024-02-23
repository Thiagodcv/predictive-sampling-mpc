import numpy as np
import torch
import gymnasium as gym


def inverted_pendulum():
    env = gym.make("InvertedPendulum-v4", render=False)
    o = env.reset()

import numpy as np
import torch
from core import DynamicsModel


class MBRLLearner:
    """
    A class for training a model-based reinforcement learning agent.
    """

    def __init__(self, state_dim, action_dim):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of the state space.
        action_dim : int
            Dimension of the action space.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self.model = DynamicsModel(state_dim, action_dim).to(device)

    def train(self, env, num_episodes, episode_len):
        """
        Parameters
        ----------
        env : gym.Env
        num_episodes : int
            Number of episodes to train for.
        episode_len : int
            The length of each episode.
        """
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

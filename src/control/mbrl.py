import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import DynamicsModel


class MBRLLearner:
    """
    A class for training a model-based reinforcement learning agent.
    """

    def __init__(self, state_dim, action_dim, lr=1e-3):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of the state space.
        action_dim : int
            Dimension of the action space.
        lr : float
            Learning rate for dynamics model.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.replay_buffer = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DynamicsModel(self.state_dim, self.action_dim).to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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

        for ep in range(num_episodes):
            self.update_dynamics()
            o, _ = env.reset()
            for t in range(episode_len):
                action = 0 # MPC goes here
                next_o, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
                self.replay_buffer.append((o, action, next_o))
                o = next_o
            env.close()

    def update_dynamics(self):
        """
        Update the dynamics model using sampled (s,a,s') triplets stored in replay_buffer.
        """
        input = []  # Figure out how to get input and target.
        target = []

        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()

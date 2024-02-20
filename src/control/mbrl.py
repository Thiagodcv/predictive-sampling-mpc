import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.control.dynamics import DynamicsModel
from src.control.mpc import MPC
from replay_buffer import ReplayBuffer


class MBRLLearner:
    """
    A class for training a model-based reinforcement learning agent.
    """

    def __init__(self, state_dim, action_dim, env, num_episodes, episode_len, reward, lr=1e-3, batch_size=16):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of the state space.
        action_dim : int
            Dimension of the action space.
        env : gym.Env
        num_episodes : int
            Number of episodes to train for.
        episode_len : int
            The length of each episode.
        reward : function
            The instantaneous reward function at each timestep.
        lr : float
            Learning rate for dynamics model.
        batch_size : int
            Batch size for dynamics model training.
        """
        # RL Training Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env
        self.num_episodes = num_episodes
        self.episode_len = episode_len

        # Dynamics Model Trainings Parameters
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DynamicsModel(self.state_dim, self.action_dim).to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # MPC Parameters
        self.num_traj = 50
        self.gamma = 0.95
        self.horizon = 10
        self.reward = reward
        self.policy = MPC(self.model, self.num_traj, self.gamma, self.horizon, self.reward)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)  # TODO: Pass in device as param

    def train(self):
        """
        Train the MBRL agent.
        """

        for ep in range(self.num_episodes):
            self.update_dynamics()
            o, _ = self.env.reset()
            for t in range(self.episode_len):
                action = self.policy.random_shooting(o, self.replay_buffer.get_last_3_actions_mean())
                next_o, reward, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break
                self.replay_buffer.push(o, action, next_o)
                o = next_o
            self.env.close()

    def update_dynamics(self):
        """
        Update the dynamics model using sampled (s,a,s') triplets stored in replay_buffer.
        """
        n_state, n_action, d_n_state = self.replay_buffer.sample(self.batch_size)
        input = torch.cat((n_state, n_action), dim=1)
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss(output, d_n_state)
        loss.backward()
        self.optimizer.step()

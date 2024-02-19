import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import DynamicsModel
from mpc import MPC


class MBRLLearner:
    """
    A class for training a model-based reinforcement learning agent.
    """

    def __init__(self, state_dim, action_dim, env, num_episodes, episode_len, lr=1e-3, batch_size=16):
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
        self.replay_buffer = []

        # Dynamics Model Trainings Parameters
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DynamicsModel(self.state_dim, self.action_dim).to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # MPC Parameters
        self.policy = MPC(self.model)
        self.num_traj = 50
        self.gamma = 0.95
        self.horizon = 10

        def reward(state, action):
            return 0
        self.reward = reward

    def train(self):
        """
        Train the MBRL agent.
        """

        for ep in range(self.num_episodes):
            self.update_dynamics()
            o, _ = self.env.reset()
            for t in range(self.episode_len):
                action = self.policy.random_shooting(o, [self.replay_buffer[-i][1] for i in range(3)],
                                                     self.num_traj, self.gamma, self.horizon, self.reward)
                next_o, reward, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break
                self.replay_buffer.append((o, action, next_o))
                o = next_o
            self.env.close()

    def update_dynamics(self):
        """
        Update the dynamics model using sampled (s,a,s') triplets stored in replay_buffer.
        """
        input = []
        target = []

        batch_idx = torch.randint(len(self.replay_buffer), size=(self.batch_size,)).item()
        batch = self.replay_buffer[batch_idx]
        for i in range(len(batch)):
            input[i].append(np.concatenate((batch[i][0], batch[i][1])))
            s_diff = batch[i][2] - batch[i][0]
            target.append(s_diff)

        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()

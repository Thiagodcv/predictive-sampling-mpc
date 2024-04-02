import numpy as np
import torch
import torch.nn as nn
from src.control.dynamics import DynamicsModel
from src.control.mpc import MPC
from src.control.replay_buffer import ReplayBuffer
import os
from datetime import datetime
from src.constants import MODELS_PATH


class MBRLLearner:
    """
    A class for training a model-based reinforcement learning agent.
    """

    def __init__(self, state_dim, action_dim, env, num_episodes, episode_len,
                 reward, terminate=None, lr=1e-3, batch_size=16, num_rand_eps=2000,
                 save_name=None, normalize=False):
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
        terminate : function
            For a given (s, a, t) tuple returns true if episode has ended.
        lr : float
            Learning rate for dynamics model.
        batch_size : int
            Batch size for dynamics model training.
        num_rand_eps : int
            Number of episodes in the beginning of training where MPC not utilized (random action taken).
        save_name : str
            The name of the trained dynamics model saved.
        normalize : boolean
            If true, normalizes the data dynamics mode is trained on.
        """
        # RL Training Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env
        self.num_episodes = num_episodes
        self.episode_len = episode_len
        self.eval_num = 5
        self.num_rand_eps = num_rand_eps
        self.normalize = normalize
        self.override_env_reward = True
        self.override_env_terminate = True

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, normalize=self.normalize)
        self.rl_prop = 0.4  # After fully random episodes are done, batches should contain this fraction of RL data

        # Dynamics Model Trainings Parameters
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DynamicsModel(self.state_dim, self.action_dim, self.normalize).to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.save_name = save_name

        # MPC Parameters
        self.num_traj = 1024  # 50
        self.gamma = 0.99
        self.horizon = 15
        self.reward = reward
        self.terminate = terminate
        self.policy = MPC(self.model, self.num_traj, self.gamma, self.horizon, self.reward, True, self.terminate)

    def train(self):
        """
        Train the MBRL agent.
        """
        for ep in range(self.num_episodes):
            if self.replay_buffer.__len__() > self.batch_size:
                self.update_model_statistics()
                self.update_dynamics(ep - 5)  # Only use rl data after 5 rl episodes

            o, _ = self.env.reset()
            ep_ret = 0
            ep_len = self.episode_len  # If episode doesn't terminate from gym, it's len will be episode_len
            self.policy.empty_past_trajectory()
            for t in range(self.episode_len):

                # Only start MPC after num_rand_eps number of episodes where only random actions taken
                if ep >= self.num_rand_eps:
                    action = self.policy.random_shooting(o)
                else:
                    action = np.random.uniform(low=-1, high=1, size=(8,))

                next_o, reward, terminated, truncated, _ = self.env.step(action)

                # Use custom reward function
                if self.override_env_reward:
                    reward = self.reward(o, action)
                ep_ret += (self.gamma ** t)*reward

                # Use custom termination condition
                if self.override_env_terminate:
                    terminated = self.terminate(o, action, t)
                    truncated = False

                if terminated or truncated:
                    ep_len = t
                    break

                self.replay_buffer.push(o, action, next_o, ep >= self.num_rand_eps)
                o = next_o
            self.env.close()

            print("Episode {} finished after {} time steps with return {}".format(ep, ep_len, ep_ret))
            # if ep % self.eval_num == 0 and ep >= self.num_rand_eps:  # Keep latter condition in for a bit
            #     self.eval_model()

        # Save trained dynamics model
        if self.save_name is None:
            now = datetime.now()
            self.save_name = now.strftime("%Y%m%d-%H%M%S")
        torch.save(self.model.state_dict(), os.path.join(MODELS_PATH, self.save_name + ".pt"))

    def update_dynamics(self, ep):
        """
        Update the dynamics model using sampled (s,a,s'-s) triplets stored in replay_buffer.
        Parameters
        ----------
        ep : int
            Episode number (in training)
        """
        for i in range(4):
            state, action, d_state = self.replay_buffer.sample(self.batch_size, self.rl_prop * (ep >= self.num_rand_eps))
            input = torch.from_numpy(np.concatenate((state, action), axis=1)).float().to(self.device)
            target = torch.from_numpy(d_state).float().to(self.device)
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

    def eval_model(self):

        o, _ = self.env.reset()
        self.policy.empty_past_trajectory()
        ret = 0
        for t in range(self.episode_len):
            action = self.policy.random_shooting(o)
            next_o, reward, terminated, truncated, _ = self.env.step(action)

            # Use custom reward function
            if self.override_env_reward:
                reward = self.reward(o, action)
            ret += (self.gamma ** t) * reward

            # Use custom termination condition
            if self.override_env_terminate:
                terminated = self.terminate(o, action, t)
                truncated = False

            if terminated or truncated:
                break
            o = next_o
        self.env.close()

        print("----------------------------------------")
        print("Model Evaluation: ret = {}".format(ret))
        print("----------------------------------------")

    @staticmethod
    def static_eval_model(env, episode_len, policy, gamma, reward_func=None, terminate_func=None):
        """
        A static version of eval_model.
        """
        o, _ = env.reset()
        ret = 0
        for t in range(episode_len):
            action = policy.random_shooting(o)
            next_o, reward, terminated, truncated, _ = env.step(action)

            # Use custom reward function
            if reward_func is not None:
                reward = reward_func(o, action)
            ret += gamma ** t * reward

            # Use custom termination condition
            if terminate_func is not None:
                terminated = terminate_func(o, action, t)
                truncated = False

            if terminated or truncated:
                break
            o = next_o
        env.close()

        print("----------------------------------------")
        print("Model Evaluation: ret = {}".format(ret))
        print("----------------------------------------")

    def update_model_statistics(self):
        self.model.update_state_var(self.replay_buffer.get_state_var())
        self.model.update_state_mean(self.replay_buffer.get_state_mean())
        self.model.update_action_var(self.replay_buffer.get_action_var())
        self.model.update_action_mean(self.replay_buffer.get_action_mean())

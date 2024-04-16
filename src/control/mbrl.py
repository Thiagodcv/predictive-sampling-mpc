import numpy as np
import torch
import torch.nn as nn
from src.control.dynamics import DynamicsModel
from src.control.mpc import MPC
from src.control.replay_buffer import ReplayBuffer
import os
from datetime import datetime
from src.constants import MODELS_PATH
import json


class MBRLLearner:
    """
    A class for training a model-based reinforcement learning agent.
    """

    def __init__(self, env_dict, train_dict, mpc_dict, misc_dict):
        """
        Parameters
        ----------
        env_dict : dict
            A dictionary containing parameters related to the environment. Keys-value pairs are
            - state_dim : int
                Dimension of the state space
            - action_dim : int
                Dimension of the action space
            - env : gym.Env

        train_dict : dict
            A dictionary containing parameters related to training. Key-value pairs are
            - num_episodes : int
                Number of episodes to train for
            - episode_len : int
                Number of time steps in each episode
            - reward : function
                Reward function at each timestep
            - terminate : function
                Termination condition at each timestep
            - lr : float
                Learning rate for training dynamics model
            - batch_size : int
                Batch size for training dynamics model
            - num_rand_eps : int
                Number of episodes at beginning of training where all actions are randomly chosen
            - rl_prop : float in [0, 1]
                Proportion of data in each batch that comes from MPC-chosen actions
            - epsilon : float in [0, 1]
                Fraction of actions taken per episode that are randomly generated (Epsilon Greedy)

        mpc_dict : dict
            A dictionary containing parameters related to the MPC controller. Key-value pairs are
            - num_traj : int
                Number of trajectories to sample at each timestep
            - gamma : float in (0, 1]
                Discount factor for computing returns (both in mbrl.py and mpc.py)
            - horizon : int
                Number of timesteps to estimate optimal trajectories at each timestep

        misc_dict : dict
            A dictionary containing miscellaneous parameters. Key-value paris are
            - save_name : str
                Name of dynamics model that will be saved
            - save_every_n_episodes : int
                Dynamics model will be saved every n episodes
            - print_every_n_episodes : int
                Training results will be printed every n episodes
            - normalize : bool
                If true, state-action will be normalized (if using multithreading, must be set to True)
            - override_env_reward : bool
                If true, reward from environment will be overriden by reward given in train_dict
            - override_env_terminate : bool
                If true, termination condition from environment will be overriden by termination
                function given in train_dict
        """
        # Environment Parameters
        self.state_dim = env_dict['state_dim']
        self.action_dim = env_dict['action_dim']
        self.env = env_dict['env']

        # Training Parameters
        self.num_episodes = train_dict['num_episodes']
        self.episode_len = train_dict['episode_len']
        self.reward = train_dict['reward']
        self.terminate = train_dict['terminate']
        self.lr = train_dict['lr']
        self.batch_size = train_dict['batch_size']
        self.num_rand_eps = train_dict['num_rand_eps']
        self.rl_prop = train_dict['rl_prop']
        self.epsilon = train_dict['epsilon']

        # Miscellaneous Parameters
        self.print_every_n_episodes = misc_dict['print_every_n_episodes']
        self.normalize = misc_dict['normalize']
        self.override_env_reward = misc_dict['override_env_reward']
        self.override_env_terminate = misc_dict['override_env_terminate']
        self.save_name = misc_dict['save_name']
        self.save_every_n_episodes = misc_dict['save_every_n_episodes']

        if self.save_name is None:
            now = datetime.now()
            self.save_name = now.strftime("%Y%m%d-%H%M%S")

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, normalize=self.normalize)

        # Dynamics Model Trainings Parameters
        self.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DynamicsModel(self.state_dim, self.action_dim, self.normalize).to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # MPC Parameters
        self.num_traj = mpc_dict['num_traj']
        self.gamma = mpc_dict['gamma']
        self.horizon = mpc_dict['horizon']
        self.policy = MPC(self.model, self.num_traj, self.gamma, self.horizon, self.reward, self.terminate, True)

        # Make model directory
        self.dir_path = os.path.join(MODELS_PATH, self.save_name)
        self.dict_file_name = 'dict_file.txt'
        self.train_file_name = 'train.txt'
        self.eval_file_name = 'eval.txt'
        self.make_model_directory(env_dict, train_dict, mpc_dict, misc_dict)


    def make_model_directory(self, env_dict, train_dict, mpc_dict, misc_dict):
        """
        Make a directory containing training information and model parameters
        """
        os.mkdir(self.dir_path)

        # Delete key-value pairs with values that can't be converted to string
        env_dict_copy = env_dict.copy()
        train_dict_copy = train_dict.copy()

        del env_dict_copy['env']
        del train_dict_copy['reward']
        del train_dict_copy['terminate']

        f_dict = open(os.path.join(self.dir_path, self.dict_file_name), 'a')
        f_dict.write('env_dict:\n')
        f_dict.write('---------\n')
        f_dict.write(json.dumps(env_dict_copy))
        f_dict.write('\n\n')

        f_dict.write('train_dict:\n')
        f_dict.write('---------\n')
        f_dict.write(json.dumps(train_dict_copy))
        f_dict.write('\n\n')

        f_dict.write('mpc_dict:\n')
        f_dict.write('---------\n')
        f_dict.write(json.dumps(mpc_dict))
        f_dict.write('\n\n')

        f_dict.write('misc_dict:\n')
        f_dict.write('---------\n')
        f_dict.write(json.dumps(misc_dict))
        f_dict.close()

        f_train = open(os.path.join(self.dir_path, self.train_file_name), 'a')
        f_train.write('Episodes, Mean Ret, StDev, Mean Termination\n')
        f_train.close()

        f_eval = open(os.path.join(self.dir_path, self.eval_file_name), 'a')
        f_eval.write('Episode, Ret\n')
        f_eval.close()

    def print_and_save_results(self, first_ep, last_ep, mean_ret, stdev, mean_termination):
        # Print to stdout
        print("Episodes {}-{} finished | mean return: {:.2f} | return stdev: {:.2f} | mean time of termination: {:.2f}"
              .format(first_ep, last_ep,  mean_ret, stdev, mean_termination))

        # Write to train file
        f_train = open(os.path.join(self.dir_path, self.train_file_name), 'a')
        f_train.write('{}-{}, {:.2f}, {:.2f}, {:.2f}\n'.format(first_ep, last_ep, mean_ret, stdev, mean_termination))
        f_train.close()

    def train(self):
        """
        Train the MBRL agent.
        """
        ret_list = []
        trunc_list = []
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
                if ep < self.num_rand_eps or np.random.uniform(low=0, high=1.0) < self.epsilon:
                    action = np.random.uniform(low=-0.3, high=0.3, size=(8,))
                else:
                    action = self.policy.random_shooting(o)

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

            # Results from training
            ret_list.append(ep_ret)
            trunc_list.append(ep_len)

            if (ep + 1) % self.print_every_n_episodes == 0 and ep != 0:
                self.print_and_save_results(first_ep=ep - self.print_every_n_episodes + 1,
                                            last_ep=ep,
                                            mean_ret=np.mean(ret_list),
                                            stdev=np.std(ret_list),
                                            mean_termination=np.mean(trunc_list))
                ret_list.clear()
                trunc_list.clear()

            # Save trained dynamics model every n episodes, and do MPC eval
            if (ep + 1) % self.save_every_n_episodes == 0 and ep != 0:
                torch.save(self.model.state_dict(), os.path.join(self.dir_path, self.save_name + ".pt"))
                self.eval_model(ep)  # Whenever a model is saved, run model with MPC
                print("-- Model saved --")

        # Save when training ends
        torch.save(self.model.state_dict(), os.path.join(self.dir_path, self.save_name + ".pt"))
        print("-- Model saved --")

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

    def eval_model(self, ep):
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
        print("Model Evaluation: ret = {:.2f}".format(ret))
        print("----------------------------------------")

        # Write to eval.txt
        f_eval = open(os.path.join(self.dir_path, self.eval_file_name), 'a')
        f_eval.write('{}, {:.2f}\n'.format(ep, ret))
        f_eval.close()


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
        print("Model Evaluation: ret = {:.2f}".format(ret))
        print("----------------------------------------")

    def update_model_statistics(self):
        self.model.update_state_var(self.replay_buffer.get_state_var())
        self.model.update_state_mean(self.replay_buffer.get_state_mean())
        self.model.update_action_var(self.replay_buffer.get_action_var())
        self.model.update_action_mean(self.replay_buffer.get_action_mean())

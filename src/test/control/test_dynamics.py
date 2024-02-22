from unittest import TestCase
from src.control.dynamics import DynamicsModel
from src.control.replay_buffer import ReplayBuffer
import gymnasium as gym
from control.mpc import MPC
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TestDynamics(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dynamics_training(self):
        state_dim = 4
        action_dim = 1
        lr = 1e-3
        model = DynamicsModel(state_dim=state_dim, action_dim=action_dim)
        replay_buffer = ReplayBuffer(state_dim=state_dim,
                                     action_dim=action_dim,
                                     normalize=False)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        env = gym.make("InvertedPendulum-v4")
        nn_evals = []

        # Training Parameters
        num_episodes = 4000
        episode_len = 100
        batch_size = 256

        # Save model
        PATH = "C:/Users/thiag/Git/random-shooting-mpc/models/good_model.pt"

        def eval_nn(h):
            # Simulate states in environment with zero action
            o, _ = env.reset()
            o_est = np.copy(o)
            action = np.zeros(1)
            t = 0
            env_states = [np.copy(o)]
            while t < episode_len:
                next_o, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
                o = next_o
                env_states = env_states + [np.copy(o)]
                t += 1
            env.close()

            # Simulate states in model with zero action
            k = 0
            model_states = [np.copy(o_est)]
            while k < t:
                input = torch.from_numpy(np.concatenate((o_est, action))).float()
                o_est = model(input).detach().numpy() + o_est
                model_states = model_states + [np.copy(o_est)]
                k += 1
            env.close()

            # Compute difference error
            err = np.linalg.norm(np.array(env_states)[0:h, :] - np.array(model_states)[0:h, :], 2)
            print("---------------------------------")
            print("Difference error in {} states: {}".format(h, err))
            print("---------------------------------")
            return err

        def update_dynamics():
            state, action, d_state = replay_buffer.sample(batch_size)
            input = torch.from_numpy(np.concatenate((state, action), axis=1)).float()

            target = torch.from_numpy(d_state).float()
            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            print("loss: ", loss.item())

        for ep in range(num_episodes):
            print("Episode {}".format(ep))
            if replay_buffer.__len__() > batch_size:
                update_dynamics()
            o, _ = env.reset()
            for t in range(episode_len):
                action = np.random.uniform(low=-3.0, high=3.0, size=action_dim)
                next_o, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

                replay_buffer.push(o, action, next_o)
                o = next_o
            env.close()

            if ep % 50 == 0:
                nn_evals.append(eval_nn(15))

        print("+++++++++++++++++++++++++")
        print(nn_evals)
        print("Average score on last 10 evaluations:")
        print(np.mean(nn_evals[-10:]))
        print("+++++++++++++++++++++++++")
        torch.save(model.state_dict(), PATH)

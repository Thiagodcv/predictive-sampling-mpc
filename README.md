# Model-Based Reinforcement Learning with Neural Network Dynamics

This repository is an implementation of a model-based reinforcement learning agent based off of [1]. The agent uses online, off-policy training data in order 
to learn the system dynamics using a neural network. An MPC using a predictive sampling scheme is then used by the agent to determine the optimal action at each timestep within the environment.
I've implemented training and evaluation on the Ant-v4 task.

PLEASE NOTE: The "pendulum" and "cartpole" environment files no longer work on the main branch. Will include a requirements file soon.

## Citations
[1] Nagabandi, Anusha, et al. "Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning." 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018.

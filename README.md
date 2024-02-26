# Model-Based Reinforcement Learning with Neural Network Dynamics

This repository is an implementation of a model-based reinforcement learning agent based off of [1]. The agent utilizes a mixture of data gathered from online and offline training in order 
to learn the system dynamics using a neural network. An MPC using a random-sampling scheme is then used by the agent to determine the optimal action at each timestep within an environment.
So far, I got the algorithm to solve the CartPole task. An application of this agent to the Ant task is coming soon.


## Citations
[1] Nagabandi, Anusha, et al. "Neural network dynamics for model-based deep reinforcement learning with model-free fine-tuning." 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018.

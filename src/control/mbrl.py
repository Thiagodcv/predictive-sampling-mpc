

class MBRLLearner:

    def mbrl_training(self, env, num_episodes, episode_len):
        replay_buffer = []

        for ep in range(num_episodes):
            # Train model on replay buffer
            # reset environment
            for t in range(episode_len):
                # action = policy(state)
                # next_state = env.step(action)
                # replay_buffer.add((state, action, next_state))
                pass
        pass

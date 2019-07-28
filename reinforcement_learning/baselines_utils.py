import numpy as np


def evaluate(model, env, episodes: int, max_steps: int = 1e6):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param episodes: (int) number of episodes to evaluate
    :return: (float) Mean reward for the last 100 episodes
    """
    obs = env.reset()

    episode_rewards = [0.0]
    total_steps = 0
    state = None
    done = [False for _ in range(env.num_envs)]
    while True:
        # _states are only useful when using LSTM policies
        action, state = model.predict(obs, state=state, mask=done)
        # here, action, rewards and dones are arrays
        # because we are using vectorized env
        obs, rewards, dones, info = env.step(action)

        # Stats
        episode_rewards[-1] += rewards[0]
        total_steps += 1
        if total_steps > max_steps:
            print("Reached max steps for evaluation.")
            break
        if dones[0]:
            if len(episode_rewards) == episodes:
                break
            obs = env.reset()
            episode_rewards.append(0.0)

    mean_reward = np.mean(episode_rewards)
    print(f"Mean reward: {mean_reward:.3f}, Num episodes: {len(episode_rewards)}")
    return mean_reward

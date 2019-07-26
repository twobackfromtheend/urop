import os

import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

from ifttt_webhook import trigger_event
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.utils.states import get_ghz_state
from reinforcement_learning.Environments.evolving_qubit_env import EvolvingQubitEnv
from reinforcement_learning.cleanup import process_log_file

gym.logger.setLevel(gym.logger.INFO)

if __name__ == '__main__':
    job_id = os.getenv("PBS_JOBID")
    trigger_event("job_progress", value1="Job started", value2=job_id)

    def evaluate(model, episodes: int, max_steps: int = 1e6):
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

        # Compute mean reward for the last 100 episodes
        mean_reward = np.mean(episode_rewards)
        print(f"Mean reward: {mean_reward:.3f}, Num episodes: {len(episode_rewards)}")

        return mean_reward


    def make_gym_env():
        N = 2
        t = 8
        env = EvolvingQubitEnv(N=N, V=1, geometry=RegularLattice1D(), t_list=np.linspace(0, t, 10),
                               ghz_state=get_ghz_state(N))

        # env = gym.make('CartPole-v0')
        return env


    # env = DummyVecEnv([lambda: make_gym_env()])  # The algorithms require a vectorized environment to run

    n_cpu = int(os.getenv('N_CPU'))
    env = SubprocVecEnv([lambda: make_gym_env() for i in range(n_cpu)])

    model = PPO2(
        MlpLstmPolicy, env,
        learning_rate=3e-3,
        verbose=1,
        nminibatches=1,
        tensorboard_log='./tensorboard_logs'
    )

    evaluate(model, episodes=10)

    trigger_event("job_progress", value1="Starting learn", value2=job_id)

    model.learn(total_timesteps=10000, log_interval=3)

    evaluate(model, episodes=10)

    process_log_file(make_gym_env())
    trigger_event("job_progress", value1="Job ended", value2=job_id)

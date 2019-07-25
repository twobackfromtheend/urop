import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, TRPO, ACKTR

from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.utils.states import get_ghz_state
from reinforcement_learning.Environments.evolving_qubit_env import EvolvingQubitEnv

if __name__ == '__main__':

    def evaluate(model, episodes: int, max_steps: int = 1e6):
        """
        Evaluate a RL agent
        :param model: (BaseRLModel object) the RL Agent
        :param episodes: (int) number of episodes to evaluate
        :return: (float) Mean reward for the last 100 episodes
        """
        episode_rewards = [0.0]
        obs = env.reset()
        total_steps = 0
        while True:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
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


    N = 2
    t = 1
    env = EvolvingQubitEnv(N=N, V=1, geometry=RegularLattice1D(), t_list=np.linspace(0, t, 300),
                           ghz_state=get_ghz_state(N))


    def make_gym_env():
        N = 2
        t = 1
        env = EvolvingQubitEnv(N=N, V=1, geometry=RegularLattice1D(), t_list=np.linspace(0, t, 300),
                               ghz_state=get_ghz_state(N))

        # env = gym.make('CartPole-v0')
        return env


    # env = DummyVecEnv([lambda: make_gym_env()])  # The algorithms require a vectorized environment to run

    n_cpu = 12
    env = SubprocVecEnv([lambda: make_gym_env() for i in range(n_cpu)])

    # model = ACKTR(MlpPolicy, env, verbose=1)
    # model = PPO2(MlpPolicy, env, verbose=1)
    model = PPO2(MlpLstmPolicy, env,
                 learning_rate=3e-3,
                 verbose=1,
                 nminibatches=1,
                 policy_kwargs={'n_lstm': 128})

    mean_reward = evaluate(model, episodes=10)

    model.learn(total_timesteps=100000, log_interval=3)

    mean_reward = evaluate(model, episodes=10)

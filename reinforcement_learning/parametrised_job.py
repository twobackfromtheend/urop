import os
import time

import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

from ifttt_webhook import trigger_event
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.utils.states import get_ghz_state
from reinforcement_learning import baselines_utils
from reinforcement_learning.Environments.evolving_qubit_env import EvolvingQubitEnv
from reinforcement_learning.cleanup import process_log_file

gym.logger.setLevel(gym.logger.INFO)

if __name__ == '__main__':
    job_id = os.getenv("PBS_JOBID")
    n_envs = int(os.getenv('N_ENVS'))
    EnvType = SubprocVecEnv if bool(eval(os.getenv('USE_SUBPROC_ENV', "False"))) else DummyVecEnv

    N = int(os.getenv("QUBIT_N"))
    t = float(os.getenv("QUBIT_T"))
    t_num = int(os.getenv("QUBIT_T_NUM"))

    trigger_event("job_progress", value1="Job started", value2=job_id)
    start_time = time.time()


    def make_gym_env():
        env = EvolvingQubitEnv(N=N, V=1, geometry=RegularLattice1D(), t_list=np.linspace(0, t, t_num),
                               ghz_state=get_ghz_state(N))
        return env


    generating_envs_start_time = time.time()
    env = EnvType([lambda: make_gym_env() for i in range(n_envs)])
    generating_envs_end_time = time.time()
    print(f"Generated {n_envs} envs in {generating_envs_end_time - generating_envs_start_time:.3f}s")

    model = PPO2(
        MlpLstmPolicy, env,
        learning_rate=3e-3,
        verbose=1,
        nminibatches=1,
        tensorboard_log='./tensorboard_logs'
    )

    baselines_utils.evaluate(model, env, episodes=20)

    model_learn_start_time = time.time()
    total_timesteps = int(os.getenv("MODEL_LEARN_TIMESTEPS"))
    model.learn(total_timesteps=total_timesteps, log_interval=3)
    model_learn_end_time = time.time()
    print(f"\nLearned for {total_timesteps} steps in {model_learn_end_time - model_learn_start_time:.3f}s")

    baselines_utils.evaluate(model, env, episodes=20)

    process_log_file(make_gym_env())

    trigger_event("job_progress", value1="Job ended", value2=job_id)

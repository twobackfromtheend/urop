import os
import time
from datetime import datetime

import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

import interaction_constants
from ifttt_webhook import trigger_event
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.utils.states import get_ghz_state
from reinforcement_learning import baselines_utils
from reinforcement_learning.Environments.evolving_qubit_env import EvolvingQubitEnv
from reinforcement_learning.cleanup import process_log_file

gym.logger.setLevel(gym.logger.INFO)

if __name__ == '__main__':
    job_id = os.getenv("PBS_JOBID")
    n_envs = int(os.getenv('ENV_N'))
    EnvType = SubprocVecEnv if bool(eval(os.getenv('ENV_USE_SUBPROC', "False"))) else DummyVecEnv
    ENV_VERBOSE = bool(eval(os.getenv("ENV_VERBOSE")))

    N = int(os.getenv("QUBIT_N"))
    t = float(os.getenv("QUBIT_T"))
    t_num = int(os.getenv("QUBIT_T_NUM"))
    N_RYD = int(os.getenv("QUBIT_N_RYD"))
    C6 = interaction_constants.get_C6(N_RYD)
    LATTICE_SPACING = 4e-6
    OMEGA_RANGE = eval(os.getenv("QUBIT_OMEGA_RANGE"))
    DELTA_RANGE = eval(os.getenv("QUBIT_DELTA_RANGE"))

    assert len(OMEGA_RANGE) == len(DELTA_RANGE) == 2, f"QUBIT_OMEGA_RANGE and QUBIT_DELTA_RANGE must be of length 2, " \
                                                      f"not {len(OMEGA_RANGE)} and {len(DELTA_RANGE)}."

    trigger_event("job_progress", value1="Job started", value2=job_id)
    start_time = time.time()


    def make_gym_env():
        env = EvolvingQubitEnv(N=N, V=C6, geometry=RegularLattice1D(LATTICE_SPACING), t_list=np.linspace(0, t, t_num),
                               Omega_range=OMEGA_RANGE, Delta_range=DELTA_RANGE,
                               ghz_state=get_ghz_state(N), verbose=ENV_VERBOSE)
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
        n_steps=t_num,
        tensorboard_log='./tensorboard_logs'
    )

    baselines_utils.evaluate(model, env, episodes=20)

    model_learn_start_time = time.time()
    total_timesteps = int(os.getenv("MODEL_LEARN_TIMESTEPS"))
    model.learn(total_timesteps=total_timesteps, log_interval=3,
                tb_log_name=f"{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    model_learn_end_time = time.time()
    print(f"\nLearned for {total_timesteps} steps in {model_learn_end_time - model_learn_start_time:.3f}s")

    baselines_utils.evaluate(model, env, episodes=20)

    process_log_file(make_gym_env())

    trigger_event("job_progress", value1="Job ended", value2=job_id)

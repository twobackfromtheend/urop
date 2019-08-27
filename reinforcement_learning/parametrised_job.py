import os
import time
from datetime import datetime
from pathlib import Path

import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecCheckNan

import interaction_constants
from ifttt_webhook import trigger_event
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.geometry.regular_lattice_2d import RegularLattice2D
from qubit_system.geometry.regular_lattice_3d import RegularLattice3D
from qubit_system.utils.ghz_states import StandardGHZState
from reinforcement_learning import baselines_utils
from reinforcement_learning.Environments.evolving_qubit_env import EvolvingQubitEnv
from reinforcement_learning.Environments.ti_evolving_qubit_env import TIEvolvingQubitEnv
from reinforcement_learning.Environments.ti_evolving_qubit_env_2 import TIEvolvingQubitEnv2
from reinforcement_learning.cleanup import process_log_file

gym.logger.setLevel(gym.logger.INFO)

if __name__ == '__main__':
    job_id = os.getenv("PBS_JOBID")
    n_envs = int(os.getenv('ENV_N'))
    EnvType = SubprocVecEnv if bool(eval(os.getenv('ENV_USE_SUBPROC', "False"))) else DummyVecEnv
    ENV_VERBOSE = bool(eval(os.getenv("ENV_VERBOSE")))

    load_trained_model = bool(eval(os.getenv("LOAD_TRAINED_MODEL", "False")))

    N = int(os.getenv("QUBIT_N"))
    t = float(os.getenv("QUBIT_T"))
    t_num = int(os.getenv("QUBIT_T_NUM"))
    N_RYD = int(os.getenv("QUBIT_N_RYD"))
    C6 = interaction_constants.get_C6(N_RYD)
    LATTICE_SPACING = 1.5e-6
    OMEGA_RANGE = eval(os.getenv("QUBIT_OMEGA_RANGE"))
    DELTA_RANGE = eval(os.getenv("QUBIT_DELTA_RANGE"))

    qubit_env_dict = {
        'EQE': EvolvingQubitEnv,
        'TI': TIEvolvingQubitEnv,
        'TI2': TIEvolvingQubitEnv2
    }
    QubitEnv = qubit_env_dict[os.getenv("QUBIT_ENV")]

    geometry_envvar = eval(os.getenv("QUBIT_GEOMETRY"))
    if geometry_envvar == 1:
        geometry = RegularLattice1D(LATTICE_SPACING)
    elif len(geometry_envvar) == 2:
        geometry = RegularLattice2D(geometry_envvar, spacing=LATTICE_SPACING)
    elif len(geometry_envvar) == 3:
        geometry = RegularLattice3D(geometry_envvar, spacing=LATTICE_SPACING)
    else:
        raise ValueError('QUBIT_GEOMETRY has to be either "1", "(X, Y)", or "(X, Y, Z)"')

    LEARNING_RATE = float(os.getenv("POLICY_LR"))

    assert len(OMEGA_RANGE) == len(DELTA_RANGE) == 2, f"QUBIT_OMEGA_RANGE and QUBIT_DELTA_RANGE must be of length 2, " \
                                                      f"not {len(OMEGA_RANGE)} and {len(DELTA_RANGE)}."

    print(
        "Parameters:\n"
        f"\tjob_id: {job_id}\n"
        f"\tn_envs: {n_envs}\n"
        f"\tEnvType: {EnvType.__name__}\n"
        f"\tENV_VERBOSE: {ENV_VERBOSE}\n"
        f"\tQubitEnv: {QubitEnv}\n"
        f"\tQUBIT_N: {N}\n"
        f"\tQUBIT_T: {t}\n"
        f"\tQUBIT_T_NUM: {t_num}\n"
        f"\tQUBIT_N_RYD: {N_RYD}\n"
        f"\tQUBIT_OMEGA_RANGE: {OMEGA_RANGE}\n"
        f"\tQUBIT_DELTA_RANGE: {DELTA_RANGE}\n"
        f"\tQUBIT_GEOMETRY: {geometry_envvar}\n"
    )

    trigger_event("job_progress", value1="Job started", value2=job_id)
    start_time = time.time()


    def make_gym_env():
        env = QubitEnv(N=N, V=C6, geometry=geometry, t_list=np.linspace(0, t, t_num),
                       Omega_range=OMEGA_RANGE, Delta_range=DELTA_RANGE,
                       ghz_state=StandardGHZState(N), verbose=ENV_VERBOSE)
        return env


    generating_envs_start_time = time.time()
    env = EnvType([lambda: make_gym_env() for i in range(n_envs)])
    env = VecCheckNan(env, raise_exception=True)
    generating_envs_end_time = time.time()
    print(f"Generated {n_envs} envs in {generating_envs_end_time - generating_envs_start_time:.3f}s")

    model = PPO2(
        MlpLstmPolicy, env,
        learning_rate=LEARNING_RATE,
        verbose=1,
        nminibatches=1,
        n_steps=t_num,
        tensorboard_log='./tensorboard_logs'
    )

    if load_trained_model:
        trained_model_path = Path(__file__).parent / "trained_model.pkl"
        model.load(str(trained_model_path))

    baselines_utils.evaluate(model, env, episodes=20)

    model_learn_start_time = time.time()
    total_timesteps = int(os.getenv("MODEL_LEARN_TIMESTEPS"))
    model.learn(total_timesteps=total_timesteps, log_interval=3,
                tb_log_name=f"{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    model_learn_end_time = time.time()
    print(f"\nLearned for {total_timesteps} steps in {model_learn_end_time - model_learn_start_time:.3f}s")

    baselines_utils.evaluate(model, env, episodes=20)

    model.save('trained_model')

    process_log_file(make_gym_env())

    trigger_event("job_progress", value1="Job ended", value2=job_id)

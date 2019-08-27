import logging
import time
from typing import Tuple

import gym
import numpy as np
# from quicktracer import trace
from qutip import Qobj

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.qubit_system_classes import EvolvingQubitSystem
from qubit_system.utils.ghz_states import BaseGHZState
from qubit_system.utils.interpolation import get_hamiltonian_coeff_linear_interpolation

logger = logging.getLogger(__name__)

ObservationType = int


class EvolvingQubitEnv(gym.Env):
    reward_range = (0, 1)

    def __init__(self, N: int, V: float, geometry: BaseGeometry,
                 t_list: np.ndarray, ghz_state: BaseGHZState,
                 Omega_range: Tuple[float, float], Delta_range: Tuple[float, float],
                 psi_0: Qobj = None, verbose: bool = False):
        self.verbose = verbose

        self.evolving_qubit_system_kwargs = {
            'N': N,
            'V': V,
            'geometry': geometry,
            't_list': t_list,
            'ghz_state': ghz_state,
            'psi_0': psi_0,
        }

        self.required_steps = len(t_list)
        # Actions for all ts needed, Omega forced to 0 at start and end.

        self.recorded_steps = {
            'Omega': [],
            'Delta': [],
        }
        self.step_number = 0

        self.action_normalisation = Omega_range, Delta_range
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32)
        )
        # Using action_normalisation as PPO policy generates actions of order 1

        self.observation_space = gym.spaces.Discrete(self.required_steps)

        self._maximum_fidelity_achieved = 0.505

    def step(self, action: np.ndarray) -> Tuple[ObservationType, float, bool, object]:
        action = self._get_action_from_normalised_action(action)
        self.recorded_steps['Omega'].append(
            0 if (self.step_number == 0 or self.step_number == self.required_steps - 1) else action[0]
        )  # Force Omega to start and end at 0
        self.recorded_steps['Delta'].append(action[1])
        self.step_number += 1

        observation = self._get_observation()

        done = self.step_number == self.required_steps  # step_number starts from 0
        # done if next action not needed.
        # E.g. t_list = np.linspace(0, 2, 3) = [0, 1, 2]. required_steps = 3. done when observation is step = 3.
        # (actions taken for step = 0, step = 1, and step = 2)

        reward = 0 if not done else self._get_ghz_state_fidelity()

        return observation, reward, done, {}

    def reset(self) -> ObservationType:
        self.recorded_steps = {
            'Omega': [],
            'Delta': [],
        }
        self.step_number = 0

        return self._get_observation()

    def render(self, mode='human'):
        logger.warning("Rendering not supported")
        return

    def _get_action_from_normalised_action(self, action: np.array) -> np.ndarray:
        unnormalised_action = []

        for i, _action in enumerate(action):
            action_normalisation_i = self.action_normalisation[i]
            action_i_range = action_normalisation_i[1] - action_normalisation_i[0]
            action_i_min = action_normalisation_i[0]
            unnormalised_action.append(_action * action_i_range + action_i_min)
        return np.array(unnormalised_action)

    def _get_observation(self) -> ObservationType:
        return self.step_number

    def _get_ghz_state_fidelity(self):
        assert len(self.recorded_steps['Omega']) == self.required_steps
        assert len(self.recorded_steps['Delta']) == self.required_steps

        evolving_qubit_system = self._create_evolving_qubit_system()
        start_solve_time = time.time()
        evolving_qubit_system.solve()
        print(f"solved in {time.time() - start_solve_time:.3f}")
        fidelity_achieved = max(
            evolving_qubit_system.get_fidelity_with("ghz"),
            evolving_qubit_system.get_fidelity_with("ghz_antisymmetric")
        )
        fidelity_with_ground = evolving_qubit_system.get_fidelity_with("ground")
        fidelity_with_excited = evolving_qubit_system.get_fidelity_with("excited")

        if self.verbose:
            gym.logger.info(f"fidelity_achieved: {fidelity_achieved:.4f}, "
                            f"with: (g: {fidelity_with_ground:.4f}), (e: {fidelity_with_excited:.4f})")

        if fidelity_achieved > self._maximum_fidelity_achieved:
            gym.logger.info(f"fidelity_achieved: {fidelity_achieved:.4f}, \n"
                            f"fidelity with: (g: {fidelity_with_ground:.5f}), (e: {fidelity_with_excited:.5f}), \n"
                            f"reward: {fidelity_with_ground * fidelity_with_excited} \n"
                            f"actions: {self.recorded_steps}\n")
            self._maximum_fidelity_achieved = fidelity_achieved
        # trace(fidelity_achieved)
        # return fidelity_achieved
        return fidelity_with_ground * fidelity_with_excited

    def _create_evolving_qubit_system(self):
        t_list = self.evolving_qubit_system_kwargs['t_list']
        _Omegas = np.array(self.recorded_steps['Omega'])
        Omega = get_hamiltonian_coeff_linear_interpolation(t_list, self.recorded_steps['Omega'])
        _Deltas = np.array(self.recorded_steps['Delta'])
        Delta = get_hamiltonian_coeff_linear_interpolation(t_list, _Deltas)

        # print("o:", _Omegas)
        # print("d:", _Deltas)
        evolving_qubit_system = EvolvingQubitSystem(
            **self.evolving_qubit_system_kwargs,
            Omega=Omega,
            Delta=Delta
        )

        return evolving_qubit_system

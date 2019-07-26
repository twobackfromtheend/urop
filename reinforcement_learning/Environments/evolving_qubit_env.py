import logging
from typing import Tuple

import gym
import numpy as np
# from quicktracer import trace
from qutip import Qobj

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.qubit_system_classes import EvolvingQubitSystem
from qubit_system.utils.interpolation import get_hamiltonian_coeff_linear_interpolation

logger = logging.getLogger(__name__)

ObservationType = int


class EvolvingQubitEnv(gym.Env):
    reward_range = (0, 1)

    def __init__(self, N: int, V: float, geometry: BaseGeometry,
                 t_list: np.ndarray, ghz_state: Qobj, psi_0: Qobj = None):
        self.evolving_qubit_system_kwargs = {
            'N': N,
            'V': V,
            'geometry': geometry,
            't_list': t_list,
            'ghz_state': ghz_state,
            'psi_0': psi_0,
        }

        self.required_steps = len(t_list) - 1
        # Actions for last t not needed: Omega = 0, Delta repeated from previous step

        self.recorded_steps = {
            'Omega': [],
            'Delta': [],
        }
        self.step_number = 0

        self.action_space = gym.spaces.Box(low=np.array([0, -10]), high=np.array([30, 10]))
        self.observation_space = gym.spaces.Discrete(self.required_steps)

    def step(self, action: np.ndarray) -> Tuple[ObservationType, float, bool, object]:
        self.recorded_steps['Omega'].append(action[0])
        self.recorded_steps['Delta'].append(action[1])
        self.step_number += 1

        observation = self._get_observation()

        done = self.step_number == self.required_steps
        # done if next action not needed.
        # E.g. t_list = [0, 1, 2]. required_steps = 2. done when observation is step = 2.
        # (actions taken for step = 0 and step = 1)

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

    def _get_observation(self) -> ObservationType:
        return self.step_number

    def _get_ghz_state_fidelity(self):
        assert len(self.recorded_steps['Omega']) == self.required_steps
        assert len(self.recorded_steps['Delta']) == self.required_steps

        evolving_qubit_system = self._create_evolving_qubit_system()
        evolving_qubit_system.solve()

        fidelity_achieved = evolving_qubit_system.get_fidelity_with("ghz")
        fidelity_with_ground = evolving_qubit_system.get_fidelity_with("ground")
        fidelity_with_excited = evolving_qubit_system.get_fidelity_with("excited")

        gym.logger.info(f"fidelity_achieved: {fidelity_achieved:.3f}\n"
                        f"fidelity with: (g: {fidelity_with_ground}), (e: {fidelity_with_excited})")

        if fidelity_achieved > 0.5:
            gym.logger.info(f"fidelity_achieved: {fidelity_achieved:.3f}, \n"
                            f"fidelity with: (g: {fidelity_with_ground}), (e: {fidelity_with_excited}), \n"
                            f"reward: {fidelity_with_ground * fidelity_with_excited} \n"
                            f"actions: {self.recorded_steps}\n")
        # trace(fidelity_achieved)
        # return fidelity_achieved
        return fidelity_with_ground * fidelity_with_excited

    def _create_evolving_qubit_system(self):
        t_list = self.evolving_qubit_system_kwargs['t_list']
        _Omegas = np.array(self.recorded_steps['Omega'] + [0])
        Omega = get_hamiltonian_coeff_linear_interpolation(t_list, self.recorded_steps['Omega'] + [0])
        _Deltas = np.array(self.recorded_steps['Delta'] + [self.recorded_steps['Delta'][-1]])
        Delta = get_hamiltonian_coeff_linear_interpolation(t_list, _Deltas)

        evolving_qubit_system = EvolvingQubitSystem(
            **self.evolving_qubit_system_kwargs,
            Omega=Omega,
            Delta=Delta
        )

        return evolving_qubit_system

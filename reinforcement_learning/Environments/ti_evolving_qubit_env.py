import logging
import time
from typing import Tuple, Optional

import gym
import numpy as np
from qutip import Qobj, tensor

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.qubit_system_classes import TimeIndependentEvolvingQubitSystem
from qubit_system.utils.ghz_states import BaseGHZState
from qubit_system.utils.states import get_ground_states

logger = logging.getLogger(__name__)

ObservationType = int


class TIEvolvingQubitEnv(gym.Env):
    """
    Uses step functions for Omega and Delta
    (Named as such because it uses the time-independent TimeIndependentEvolvingQubitSystem)
    """
    reward_range = (0, 1)

    def __init__(self, N: int, V: float, geometry: BaseGeometry,
                 t_list: np.ndarray, ghz_state: BaseGHZState,
                 Omega_range: Tuple[float, float], Delta_range: Tuple[float, float],
                 verbose: bool = False):
        self.verbose = verbose

        self.ti_evolving_qubit_system_kwargs = {
            'N': N,
            'V': V,
            'geometry': geometry,
            'ghz_state': ghz_state,
            't_list': np.linspace(0, t_list[1], 10)
        }
        self.psi_0 = tensor(get_ground_states(N))
        self.t_list = t_list

        self.required_steps = len(t_list)
        # Actions for all ts needed, Omega forced to 0 at start and end.

        self.recorded_steps = {
            'Omega': [],
            'Delta': [],
        }
        self.step_number = 0
        self.latest_evolving_qubit_system: Optional[TimeIndependentEvolvingQubitSystem] = None
        self.total_solve_time = 0

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
        self.recorded_steps['Omega'].append(action[0])
        self.recorded_steps['Delta'].append(action[1])

        self.step_number += 1
        self._update_latest_evolving_qubit_system_with_action(action[0], action[1])

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
        self.latest_evolving_qubit_system = None
        print(f"solved in {self.total_solve_time:.3f}")
        self.total_solve_time = 0

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

    def _get_latest_state(self) -> Qobj:
        return self.psi_0 if self.latest_evolving_qubit_system is None \
            else self.latest_evolving_qubit_system.solve_result.states[-1]

    def _update_latest_evolving_qubit_system_with_action(self, Omega: float, Delta: float):
        latest_state = self._get_latest_state()
        evolving_qubit_system = TimeIndependentEvolvingQubitSystem(
            **self.ti_evolving_qubit_system_kwargs,
            psi_0=latest_state,
            Omega=Omega,
            Delta=Delta
        )
        start_solve_time = time.time()
        evolving_qubit_system.solve()
        self.total_solve_time += time.time() - start_solve_time
        self.latest_evolving_qubit_system = evolving_qubit_system

    def _get_ghz_state_fidelity(self):
        assert len(self.recorded_steps['Omega']) == self.required_steps
        assert len(self.recorded_steps['Delta']) == self.required_steps

        fidelity_achieved = self.latest_evolving_qubit_system.get_fidelity_with("ghz")
        fidelity_with_ground = self.latest_evolving_qubit_system.get_fidelity_with("ground")
        fidelity_with_excited = self.latest_evolving_qubit_system.get_fidelity_with("excited")

        if self.verbose:
            gym.logger.info(f"fidelity_achieved: {fidelity_achieved:.4f}, "
                            f"with: (g: {fidelity_with_ground:.4f}), (e: {fidelity_with_excited:.4f})")

        if fidelity_achieved > self._maximum_fidelity_achieved:
            gym.logger.info(f"fidelity_achieved: {fidelity_achieved:.4f}, \n"
                            f"fidelity with: (g: {fidelity_with_ground:.5f}), (e: {fidelity_with_excited:.5f}), \n"
                            f"reward: {fidelity_with_ground * fidelity_with_excited} \n"
                            f"actions: {self.recorded_steps}\n")
            self._maximum_fidelity_achieved = fidelity_achieved
        return fidelity_with_ground * fidelity_with_excited


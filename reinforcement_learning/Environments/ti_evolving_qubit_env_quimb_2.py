from typing import Tuple

import gym
import numpy as np

from qubit_system.geometry.base_geometry import BaseGeometry
from qubit_system.utils.ghz_states import BaseGHZState
from reinforcement_learning.Environments.ti_evolving_qubit_env_quimb import TIEvolvingQubitEnvQ

ObservationType = np.ndarray


class TIEvolvingQubitEnvQ2(TIEvolvingQubitEnvQ):

    def __init__(self, N: int, V: float, geometry: BaseGeometry, t_list: np.ndarray, ghz_state: BaseGHZState,
                 Omega_range: Tuple[float, float], Delta_range: Tuple[float, float],
                 verbose: bool = False):
        super().__init__(N, V, geometry, t_list, ghz_state, Omega_range, Delta_range, verbose)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([1, 1], dtype=np.float32)
        )

    def _get_observation(self) -> ObservationType:
        latest_state = self._get_latest_state()
        latest_state_data = latest_state.data.toarray().flatten()

        P_ee = latest_state_data[0]
        P_gg = latest_state_data[-1]

        return np.absolute(np.array([P_ee, P_gg]))

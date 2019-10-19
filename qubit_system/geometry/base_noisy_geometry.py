from abc import ABC

import numpy as np

from qubit_system.geometry.base_geometry import BaseGeometry


class BaseNoisyGeometry(BaseGeometry, ABC):
    def __init__(self):
        self.coordinates: np.ndarray = None

    def get_distance(self, i: int, j: int, with_noise: bool = False) -> float:
        raise NotImplementedError

    def reset_coordinates(self):
        self.coordinates = self._generate_coordinates()

    def _generate_coordinates(self) -> np.ndarray:
        raise NotImplementedError

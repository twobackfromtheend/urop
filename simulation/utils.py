from typing import Callable, Optional

import numpy as np
from scipy import interpolate


def get_hamiltonian_coeff_linear_interpolation(x: np.array, y: np.array) -> Callable[[float], float]:
    """
    Clips to 0 beyond x range.
    :param x:
    :param y:
    :return: Function to use as Hamiltonian coefficient
    """
    f = interpolate.interp1d(x, y)

    def coeff_fn(t: float, args: dict = None):
        if t < min(x) or t > max(x):
            return 0
        return f(t)
    return coeff_fn

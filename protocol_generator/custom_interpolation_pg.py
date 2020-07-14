from functools import partial
from typing import Tuple, Union, Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

from protocol_generator.base_protocol_generator import BaseProtocolGenerator


class CustomInterpolationPG(BaseProtocolGenerator):
    def __init__(self, t_list: np.ndarray, kind: Union[str, int], tukey_alpha: float, noise: float):
        super().__init__(t_list)
        self.kind = kind
        self.interp = partial(interp1d, kind=kind)
        self.tukey_alpha = tukey_alpha
        self.noise = noise

    def get_protocol(self, input_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input_timesteps = len(input_) // 2
        input_t_list = np.linspace(0, self.t_list[-1], input_timesteps + 1)

        noise_offsets = np.random.standard_normal(input_.shape) * self.noise * input_
        input_ += noise_offsets

        Omega_params = input_[:input_timesteps]
        Delta_params = input_[input_timesteps:]

        Omega_func: Callable[[float], float] = self.interp(input_t_list, np.hstack((Omega_params, Omega_params[-1])))
        Omega_shape_window = tukey(self.timesteps + 1, alpha=self.tukey_alpha)
        Omega = np.array([Omega_func(_t) * Omega_shape_window[_i] for _i, _t in enumerate(self.t_list[:-1])])

        Delta_func: Callable[[float], float] = self.interp(input_t_list, np.hstack((Delta_params, Delta_params[-1])))
        Delta = np.array([Delta_func(_t) for _t in self.t_list[:-1]])
        return Omega, Delta

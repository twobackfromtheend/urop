from typing import Tuple, Union

import numpy as np

from protocol_generator.interpolation_pg import InterpolationPG


class NoisyInterpolationPG(InterpolationPG):
    """
    Adds random uniform noise to the input_ before generating protocol.
    """
    def __init__(self, t_list: np.ndarray, kind: Union[str, int], noise: float):
        super().__init__(t_list, kind)
        self.noise = noise

    def get_protocol(self, input_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        noise_offsets = (np.random.random(input_.shape) * 2 - 1) * self.noise * input_
        input_ = noise_offsets + input_
        return super().get_protocol(input_)


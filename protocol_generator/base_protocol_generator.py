from typing import Tuple

import numpy as np


class BaseProtocolGenerator:
    def __init__(self, t_list: np.ndarray):
        self.t_list = t_list
        self.timesteps = len(t_list) - 1

    def get_protocol(self, input_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

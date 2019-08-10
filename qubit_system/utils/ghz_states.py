from typing import Tuple

from qutip import Qobj, tensor, basis

from qubit_system.utils.states import get_ground_states, get_excited_states


class BaseGHZState:
    def __init__(self, N: int):
        self.N = N
        self.components = self._get_components()

    def _get_components(self) -> Tuple[Qobj, Qobj]:
        raise NotImplementedError

    def get_state_tensor(self, symmetric: bool = True) -> Qobj:
        ghz_1, ghz_2 = self.components
        if symmetric:
            return (ghz_1 + ghz_2).unit()
        else:
            return (ghz_1 - ghz_2).unit()


class StandardGHZState(BaseGHZState):
    def _get_components(self) -> Tuple[Qobj, Qobj]:
        return tensor(get_ground_states(self.N)), tensor(get_excited_states(self.N))


class AlternatingGHZState(BaseGHZState):
    def _get_components(self) -> Tuple[Qobj, Qobj]:
        ghz_1 = tensor([basis(2, 1) if _ % 2 == 0 else basis(2, 0) for _ in range(self.N)])
        ghz_2 = tensor([basis(2, 1) if _ % 2 == 1 else basis(2, 0) for _ in range(self.N)])
        return ghz_1, ghz_2

from typing import Tuple, List

from qutip import *


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
        return tensor([basis(2, 1) for _ in range(self.N)]), tensor([basis(2, 0) for _ in range(self.N)])


class CustomGHZState(BaseGHZState):
    def __init__(self, N: int, single_component: List[bool]):
        self.single_component = single_component
        assert len(single_component) == N, \
            f"single_component has to be a list of len N ({N}), not {len(single_component)}"
        super().__init__(N)

    def _get_components(self) -> Tuple[Qobj, Qobj]:
        ghz_1 = tensor([basis(2, 1) if self.single_component[i] else basis(2, 0) for i in range(self.N)])
        ghz_2 = tensor([basis(2, 1) if not self.single_component[i] else basis(2, 0) for i in range(self.N)])
        return ghz_1, ghz_2


__all__ = ['BaseGHZState', 'StandardGHZState', 'CustomGHZState']

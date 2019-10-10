from typing import Tuple, List

import quimb as q
from qubit_system.utils.states_quimb import get_ground_states, get_excited_states


class BaseGHZState:
    def __init__(self, N: int):
        self.N = N
        self.components = self._get_components()

    def _get_components(self) -> Tuple[q.qarray, q.qarray]:
        raise NotImplementedError

    def get_state_tensor(self, symmetric: bool = True) -> q.qarray:
        ghz_1, ghz_2 = self.components
        if symmetric:
            return q.normalize(ghz_1 + ghz_2, inplace=True)
        else:
            return q.normalize(ghz_1 - ghz_2, inplace=True)


class StandardGHZState(BaseGHZState):
    def _get_components(self) -> Tuple[q.qarray, q.qarray]:
        return q.kron(*get_ground_states(self.N, sparse=True)), q.kron(*get_excited_states(self.N, sparse=True))


class AlternatingGHZState(BaseGHZState):
    def _get_components(self) -> Tuple[q.qarray, q.qarray]:
        ghz_1 = q.kron(
            *[q.ket([0, 1], sparse=True) if _ % 2 == 0 else q.ket([1, 0], sparse=True)
              for _ in range(self.N)]
        )
        ghz_2 = q.kron(
            *[q.ket([0, 1], sparse=True) if _ % 2 == 1 else q.ket([1, 0], sparse=True)
              for _ in range(self.N)]
        )
        return ghz_1, ghz_2


class CustomGHZState(BaseGHZState):
    def __init__(self, N: int, single_component: List[bool]):
        self.single_component = single_component
        assert len(single_component) == N, \
            f"single_component has to be a list of len N ({N}), not {len(single_component)}"
        super().__init__(N)

    def _get_components(self) -> Tuple[q.qarray, q.qarray]:
        ghz_1 = q.kron(
            *[q.ket([0, 1], sparse=True) if self.single_component[i] else q.ket([1, 0], sparse=True)
              for i in range(self.N)]
        )
        ghz_2 = q.kron(
            *[q.ket([0, 1], sparse=True) if not self.single_component[i] else q.ket([1, 0], sparse=True)
              for i in range(self.N)]
        )
        return ghz_1, ghz_2


__all__ = ['BaseGHZState', 'StandardGHZState', 'AlternatingGHZState', 'CustomGHZState']

if __name__ == '__main__':
    N = 3
    ghz_state = StandardGHZState(N)
    print(ghz_state.get_state_tensor(True))
    print(ghz_state.get_state_tensor(False))

from itertools import combinations
from typing import List

import quimb as q
from quimb.core import make_immutable


def get_ground_states(N: int, **kwargs) -> List[q.qarray]:
    return [q.ket([0, 1], **kwargs) for _ in range(N)]


def get_excited_states(N: int, **kwargs) -> List[q.qarray]:
    return [q.ket([1, 0], **kwargs) for _ in range(N)]


def get_states(N: int, **kwargs) -> List[List[q.qarray]]:
    """
    Returns all basis states.
    E.g. for L = 2, returns
        [
            [g, g],
            [e, g],
            [g, e],
            [e, e]
        ]
    where 'g' and 'e' correspond to q.ket([0, 1]) and q.ket([1, 0]) respectively

    :param N: number of qubits
    :return:
    """
    states = []
    g = get_ground_states(1, **kwargs)[0]
    e = get_excited_states(1, **kwargs)[0]

    make_immutable(g)
    make_immutable(e)

    for number_excited in range(N + 1):
        for excited_indices in combinations(range(N), number_excited):
            qubits = []
            for i in range(N):
                _is_excited = int(i in excited_indices)
                qubits.append(e if _is_excited else g)

            states.append(qubits)

    return states


def is_excited(state: q.qarray):
    return q.expec(q.pauli("Z"), state) == 1


def get_label_from_state(_state: List[q.qarray]) -> str:
    return "".join(["e" if is_excited(spin) else "g" for spin in _state])


def get_product_basis_states_index(state: List[q.qarray]) -> int:
    return q.kron(*state).argmax()


__all__ = ['get_ground_states', 'get_excited_states', 'get_states', 'is_excited',
           'get_label_from_state', 'get_product_basis_states_index']

if __name__ == '__main__':
    for state in get_states(3):
        print("State:")
        print(list(is_excited(spin) for spin in state))

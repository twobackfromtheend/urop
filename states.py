from typing import List
from qutip import *
from itertools import combinations


def get_ground_states(L: int) -> List[Qobj]:
    return [basis(2, 1) for _ in range(L)]


def get_states(L: int) -> List[List[Qobj]]:
    """
    Returns all basis states.
    E.g. for L = 2, returns
        [
            [g, g],
            [g, e],
            [e, g],
            [e, e]
        ]
    where 'g' and 'e' correspond to Qobj's basis(2, 1) and basis(2, 0) respectively

    :param L: number of qubits
    :return:
    """
    states = []

    for number_excited in range(L + 1):
        for excited_indices in combinations(range(L), number_excited):
            qubits = []
            for i in range(L):
                _is_excited = int(i not in excited_indices)
                qubits.append(basis(2, _is_excited))

            states.append(qubits)

    return states


def is_excited(state: Qobj):
    return expect(sigmaz(), state) == 1


def get_ghz_state(L: int, _type: str = 'typical') -> Qobj:
    ghz_1 = tensor([basis(2, 1) if _ % 2 == 0 else basis(2, 0) for _ in range(L)])
    ghz_2 = tensor([basis(2, 1) if _ % 2 == 1 else basis(2, 0) for _ in range(L)])

    ghz = (ghz_1 + ghz_2).unit()
    return ghz


def get_label_from_state(_state: List[Qobj]) -> str:
    return "".join(["e" if is_excited(spin) else "g" for spin in _state])


if __name__ == '__main__':
    for state in get_states(3):
        print("State:")
        print(list(is_excited(spin) for spin in state))

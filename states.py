from typing import List
from qutip import *
from itertools import combinations


def get_states(L: int) -> List[List[Qobj]]:
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


if __name__ == '__main__':
    for state in get_states(3):
        print("State:")
        print(list(is_excited(spin) for spin in state))


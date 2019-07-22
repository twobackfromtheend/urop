from enum import Enum
from typing import List
from qutip import *
from itertools import combinations


def get_exp_list(N: int):
    si = qeye(2)

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(N):
        op_list = []
        for m in range(N):
            op_list.append(si)

        op_list[n] = sigmax()  # Replaces the nth element of identity matrix list with the Sx matrix
        sx_list.append(tensor(op_list))
        # Resulting tensor operates on the nth qubit only --> sigmax() operates on the nth qubit,
        # depending on where sigmax() was appended
        # sx_list contains the n sigmax() that operate on the n qubits, with each index operating on a certain qubit

        op_list[n] = sigmay()
        sy_list.append(tensor(op_list))

        op_list[n] = sigmaz()
        sz_list.append(tensor(op_list))

    exp_list = [sx_list, sy_list, sz_list]
    return exp_list


def get_ground_states(N: int) -> List[Qobj]:
    return [basis(2, 1) for _ in range(N)]


def get_states(N: int) -> List[List[Qobj]]:
    """
    Returns all basis states.
    E.g. for L = 2, returns
        [
            [g, g],
            [e, g],
            [g, e],
            [e, e]
        ]
    where 'g' and 'e' correspond to Qobj's basis(2, 1) and basis(2, 0) respectively

    :param N: number of qubits
    :return:
    """
    states = []

    for number_excited in range(N + 1):
        for excited_indices in combinations(range(N), number_excited):
            qubits = []
            for i in range(N):
                _is_excited = int(i not in excited_indices)
                qubits.append(basis(2, _is_excited))

            states.append(qubits)

    return states


def is_excited(state: Qobj):
    return expect(sigmaz(), state) == 1


class GHZStateType(Enum):
    STANDARD = 0
    ALTERNATING = 1


def get_ghz_state(N: int, _type: GHZStateType = GHZStateType.STANDARD) -> Qobj:
    if _type == GHZStateType.STANDARD:
        return ghz_state(N)
    elif _type == GHZStateType.ALTERNATING:
        ghz_1 = tensor([basis(2, 1) if _ % 2 == 0 else basis(2, 0) for _ in range(N)])
        ghz_2 = tensor([basis(2, 1) if _ % 2 == 1 else basis(2, 0) for _ in range(N)])
        return (ghz_1 + ghz_2).unit()
    else:
        raise ValueError(f"Unknown _type: {_type}. _type has to be GHZStateType.")


def get_label_from_state(_state: List[Qobj]) -> str:
    return "".join(["e" if is_excited(spin) else "g" for spin in _state])


if __name__ == '__main__':
    for state in get_states(3):
        print("State:")
        print(list(is_excited(spin) for spin in state))

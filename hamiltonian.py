from qutip import *
from scipy import constants


def get_exp_list(L):
    si = qeye(2)

    sx_list = []
    sy_list = []
    sz_list = []

    for n in range(L):
        op_list = []
        for m in range(L):
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


def get_hamiltonian(
        L: int,
        Omega: float,
        Delta: float,
        V: float = 2 * constants.pi * 24e6
) -> Qobj:
    H: Qobj = 0
    sx_list, sy_list, sz_list = get_exp_list(L)

    for i in range(L):
        H += Omega / 2 * sx_list[i]
        n_i = (sz_list[i] + qeye(1)) / 2
        H -= Delta * n_i

        for j in range(i):
            n_j = (sz_list[j] + qeye(1)) / 2

            H += V / abs(j - i) ** 6 * n_i * n_j
    return H


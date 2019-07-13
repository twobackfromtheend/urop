from typing import Callable

from qutip import *

import paper_data


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


def get_td_hamiltonian(
        L: int,
        Omega: Callable[[float, dict], float],
        Delta: Callable[[float, dict], float],
        V: float = paper_data.V
):
    sx_list, sy_list, sz_list = get_exp_list(L)
    time_independent_terms = 0
    Omega_coeff_terms = 0
    Delta_coeff_terms = 0

    for i in range(L):
        Omega_coeff_terms += 1 / 2 * sx_list[i]
        n_i = (sz_list[i] + qeye(1)) / 2
        Delta_coeff_terms -= n_i

        for j in range(i):
            n_j = (sz_list[j] + qeye(1)) / 2

            time_independent_terms += V / abs(j - i) ** 6 * n_i * n_j
    return [
        time_independent_terms,
        [Omega_coeff_terms, Omega],
        [Delta_coeff_terms, Delta]
    ]

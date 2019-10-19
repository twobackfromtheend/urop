# def calculate_subsystem_entropy(state: Qobj):
#     rhoA = ptrace(state, np.arange(e_qs.N / 2))
#     # rhoB = ptrace(final_state, np.arange(e_qs.N / 2, e_qs.N))
#     # return entropy_vn(rhoA)
#     return (np.e ** entropy_vn(rhoA, sparse=True)) / (e_qs.N / 2)
from typing import List

import quimb as q
import numpy as np


def print_entropy(var: str, i: List[int], N: int):
    entropy = q.entropy_subsys(eval(var), [2] * N, i)
    print(f"{var}:\t {entropy:.3f} \t (max: {N/2:.3f})")

qe_ = q.ket([1, 0])
qg_ = q.ket([0, 1])

ghz_4 = q.normalize(
    q.kron(qg_, qg_, qg_, qg_) +
    q.kron(qe_, qe_, qe_, qe_)
)
print_entropy("ghz_4", [0, 1], 4)

C_1 = q.normalize(
    q.kron(qg_, qg_, qg_, qg_) +
    q.kron(qe_, qe_, qg_, qg_) +
    q.kron(qg_, qg_, qe_, qe_) -
    q.kron(qe_, qe_, qe_, qe_)
)
print_entropy("C_1", [0, 2], 4)

C_2 = q.normalize(
    q.kron(qg_, qg_, qg_, qg_) +
    q.kron(qg_, qe_, qe_, qg_) +
    q.kron(qe_, qg_, qg_, qe_) -
    q.kron(qe_, qe_, qe_, qe_)
)
print_entropy("C_2", [0, 1], 4)


psi_5 = q.normalize(
    q.kron(qg_, qg_, qg_, qg_, qg_) +
    q.kron(qe_, qg_, qg_, qe_, qe_) +
    q.kron(qg_, qg_, qe_, qg_, qe_) +
    q.kron(qe_, qe_, qg_, qg_, qe_) +
    q.kron(qe_, qg_, qe_, qe_, qg_) +
    q.kron(qg_, qe_, qe_, qe_, qe_) +
    q.kron(qe_, qe_, qe_, qg_, qg_)
)
print_entropy("psi_5", [0, 1], 5)


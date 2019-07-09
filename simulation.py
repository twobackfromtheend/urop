from scipy import  constants
from qutip import *

L = 3
# Omega = 3e6  # 3 megahertz
Omega = 0
Delta = [2 * constants.pi * 5e6 for _ in range(L)]
V = 2 * constants.pi * 24e6

si = qeye(2)
# n_i = Qobj([[1, 0], [0, 0]])

sx_list = []
sy_list = []
sz_list = []

for n in range(L):
    op_list = []
    for m in range(L):
        op_list.append(si)

    op_list[n] = sigmax() / 2  # Replaces the nth element of identity matrix list with the Sx matrix
    sx_list.append(tensor(op_list))
    # Resulting tensor operates on the nth qubit only --> sigmax() operates on the nth qubit,
    # depending on where sigmax() was appended
    # sx_list contains the n sigmax() that operate on the n qubits, with each index operating on a certain qubit

    op_list[n] = sigmay() / 2
    sy_list.append(tensor(op_list))

    op_list[n] = sigmaz() / 2
    sz_list.append(tensor(op_list))

exp_list = [sx_list, sy_list, sz_list]

H = 0

for i in range(L):
    H += Omega / 2 * sz_list[i]
    n_i = sz_list[i] ** 2 + qeye(1)
    H -= Delta[i] * n_i

    for j in range(i):
        n_j = sz_list[j] ** 2 + qeye(1)

        H += V / abs(j - i) ** 6 * n_i * n_j

result = mesolve(H, psi0, tlist, c_op_list, sz_list)


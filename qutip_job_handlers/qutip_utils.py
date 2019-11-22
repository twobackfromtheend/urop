from qutip import *


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

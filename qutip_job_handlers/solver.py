import numpy as np
from qutip import *


def solve(hamiltonian: list, psi_0: Qobj, t_list: np.ndarray):
    return sesolve(
        hamiltonian,
        psi_0,
        t_list,
        options=Options(store_states=True, nsteps=5000),
    )

import numpy as np
from qutip import *

from paper_data import get_interpolated_function, Omega, Delta
from simulation.time_dependent_hamiltonian import get_td_hamiltonian, get_exp_list

L = 4

psi_0 = tensor([basis(2, 1) for _ in range(L)])
# psi_0 = tensor([basis(2, 0) for _ in range(L)])

t_list = np.linspace(0, 1.1e-6, 1000)

Omega_interpolated_fn = get_interpolated_function(Omega[L])


def get_Omega(t: float, args: dict = None):
    return Omega_interpolated_fn(t)


Delta_interpolated_fn = get_interpolated_function(Delta[L])


def get_Delta(t: float, args: dict = None):
    return Delta_interpolated_fn(t)


import matplotlib.pyplot as plt
plt.figure()
plt.plot(t_list, [get_Omega(t) for t in t_list])
plt.xlabel("t")
plt.ylabel("Omega")
plt.figure()

plt.plot(t_list, [get_Delta(t) for t in t_list])
plt.xlabel("t")
plt.ylabel("Delta")
plt.show()

H = get_td_hamiltonian(L, get_Omega, get_Delta)
result = mesolve(H, psi_0, t_list, e_ops=get_exp_list(L)[2], options=Options(store_states=True))
s = result.states[-1]

from hamiltonian import get_hamiltonian

ti_h = get_hamiltonian(L, 0, 0)


ghz_1 = tensor([basis(2, 1) if _ % 2 == 0 else basis(2, 0) for _ in range(L)])
ghz_1_hc = tensor([bra("1") if _ % 2 == 0 else bra("0") for _ in range(L)])
ghz_2 = tensor([basis(2, 1) if _ % 2 == 1 else basis(2, 0) for _ in range(L)])
ghz_2_hc = tensor([bra("1") if _ % 2 == 1 else bra("0") for _ in range(L)])

import matplotlib.pyplot as plt


for sz_expectation in result.expect:
    plt.plot(t_list, sz_expectation)

plt.show()
pass

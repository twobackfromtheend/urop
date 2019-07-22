import numpy as np
from qutip import *
from scipy import constants

import paper_data
from paper_data import Omega, Delta, get_hamiltonian_coeff_fn
from simulation.time_dependent_hamiltonian import get_td_hamiltonian, get_exp_list, get_ti_hamiltonian
from states import get_ground_states

L = 4

psi_0 = tensor(get_ground_states(L))

t_list = np.linspace(0, 1.1e-6, 1000)

get_Omega = get_hamiltonian_coeff_fn(Omega, L)
get_Delta = get_hamiltonian_coeff_fn(Delta, L)


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

H = get_td_hamiltonian(L, get_Omega, get_Delta, V=paper_data.V)
result = mesolve(H, psi_0, t_list, e_ops=get_exp_list(L)[2], options=Options(store_states=True))
s = result.states[-1]


ti_h = get_ti_hamiltonian(L, 0, 0,  2 * constants.pi * 24e6)


ghz_1 = tensor([basis(2, 1) if _ % 2 == 0 else basis(2, 0) for _ in range(L)])
ghz_1_hc = tensor([bra("1") if _ % 2 == 0 else bra("0") for _ in range(L)])
ghz_2 = tensor([basis(2, 1) if _ % 2 == 1 else basis(2, 0) for _ in range(L)])
ghz_2_hc = tensor([bra("1") if _ % 2 == 1 else bra("0") for _ in range(L)])

ghz = (ghz_1 + ghz_2).unit()

import matplotlib.pyplot as plt


for sz_expectation in result.expect:
    plt.plot(t_list, sz_expectation)

plt.show()
pass


fidelity(ghz_1, s)
fidelity((ghz_1 + ghz_2).unit(), s)


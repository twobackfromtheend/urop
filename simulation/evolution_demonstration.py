# %%
import numpy as np
from qutip import *

from simulation.plotter import plot_state_overlaps, plot_Omega_and_Delta, plot_detuning_energy_levels
from simulation.time_dependent_hamiltonian import get_td_hamiltonian, get_exp_list
from simulation.utils import get_hamiltonian_coeff_linear_interpolation
from states import get_ground_states

L = 4
t = 1

Omega = get_hamiltonian_coeff_linear_interpolation(
    x=np.array([0, t / 5, t * 4 / 5, t]),
    y=np.array([0, 1, 1, 0])
)

Delta = get_hamiltonian_coeff_linear_interpolation(
    x=np.array([0, t]),
    y=np.array([-1, 1])
)

V = 1
psi_0 = tensor(get_ground_states(L))

t_list = np.linspace(0, t, 1000)

H = get_td_hamiltonian(L, Omega, Delta, V=V)
result = mesolve(H, psi_0, t_list, e_ops=get_exp_list(L)[2], options=Options(store_states=True))
s = result.states[-1]

detuning = np.linspace(Delta(0), Delta(t), 2 ** L)
plot_detuning_energy_levels(L, V, detuning, plot_state_names=True, savefig_name="detuning.png")
plot_Omega_and_Delta(t_list, Omega, Delta, ghz_state(L), result, savefig_name="omega_and_delta.png")
plot_state_overlaps(L, t_list, result, show=True, savefig_name="state_overlaps.png")

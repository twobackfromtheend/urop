import numpy as np

import interaction_constants
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.qubit_system_classes import StaticQubitSystem, EvolvingQubitSystem
from qubit_system.utils.ghz_states import StandardGHZState
from qubit_system.utils.interpolation import get_hamiltonian_coeff_linear_interpolation

import matplotlib.pyplot as plt
plt.rc('text', usetex=False)
plt.rc('font', family="serif", serif="CMU Serif")

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
print(f"Characteristic V: {C6 / (LATTICE_SPACING ** 6):.3e} Hz")

t = 2e-6

N = 4
s_qs = StaticQubitSystem(
    N=N, V=C6,
    geometry=RegularLattice1D(spacing=LATTICE_SPACING),
    Omega=0, Delta=np.linspace(-2e9, 2e9, 2 ** N)
)
s_qs.plot_detuning_energy_levels(True, savefig_name="energy_levels.png", show=True)


def solve_and_print_stats(e_qs: EvolvingQubitSystem):
    e_qs.solve()
    fidelity_with_ghz = e_qs.get_fidelity_with("ghz")
    fidelity_with_ghz_asymmetric = e_qs.get_fidelity_with("ghz_antisymmetric")
    print(f"fidelity with GHZ: {fidelity_with_ghz:.4f} (with antisymmetric: {fidelity_with_ghz_asymmetric:.4f})")
    fidelity_with_ground = e_qs.get_fidelity_with("ground")
    fidelity_with_excited = e_qs.get_fidelity_with("excited")
    superposition_probability = fidelity_with_ground + fidelity_with_excited
    print(
        f"superposition probability: {superposition_probability:.4f} (g: {fidelity_with_ground:.4f}, e: {fidelity_with_excited:.4f})")

    e_qs.plot(with_antisymmetric_ghz=True, savefig_name="evolving_fidelities.png", show=True)


e_qs = EvolvingQubitSystem(
    N=N, V=C6, geometry=RegularLattice1D(spacing=LATTICE_SPACING),
    Omega=get_hamiltonian_coeff_linear_interpolation([0, t / 2, t], [0, 1.4148e9, 0]),
    Delta=get_hamiltonian_coeff_linear_interpolation([0, t], [1.5e9, 1e9]),
    t_list=np.linspace(0, t, 300),
    ghz_state=StandardGHZState(N)
)
solve_and_print_stats(e_qs)

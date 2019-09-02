# Investigate different GHZ states

import logging

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj

import interaction_constants
from demonstration_utils import solve_and_print_stats, get_normalised_hamiltonian
from qubit_system.geometry.regular_lattice_1d import RegularLattice1D
from qubit_system.geometry.regular_lattice_2d import RegularLattice2D
from qubit_system.qubit_system_classes import EvolvingQubitSystem

from qubit_system.utils.ghz_states import StandardGHZState, AlternatingGHZState
from qubit_system.utils.interpolation import get_hamiltonian_coeff_interpolation

logging.basicConfig(level=logging.INFO)

N_RYD = 50
C6 = interaction_constants.get_C6(N_RYD)

LATTICE_SPACING = 1.5e-6

print(f"C6: {C6:.3e}")
characteristic_V = C6 / (LATTICE_SPACING ** 6)
print(f"Characteristic V: {characteristic_V:.3e} Hz")

# Plot lattice arrangement

# RegularLattice2D((2, 4), spacing=LATTICE_SPACING).plot(show=True)

# Show system energy levels

# N = 8
# s_qs = StaticQubitSystemQ(
#     N=N, V=C6,
#     # geometry=RegularLattice1D(spacing=LATTICE_SPACING),
#     geometry=RegularLattice2D((2, 4), spacing=LATTICE_SPACING),
#     Omega=0, Delta=np.linspace(-1e9, 3e9, 3)
# )
# s_qs.plot_detuning_energy_levels(
#     plot_state_names=False, show=True,
#     highlight_states_by_label=[
#         # ''.join(['eg' for _ in range(int(N / 2))]),
#         'egeggege',
#         ''.join(['e' for _ in range(N)])
#     ]
# )


# GRAPE
N = 8
t = 2e-6
import qutip.control.pulseoptim as cpo
import qutip.logging_utils

norm_V = C6 / (LATTICE_SPACING ** 6) / characteristic_V
norm_H_d, norm_H_c, psi_0 = get_normalised_hamiltonian(N, norm_V)
n_ts = 50

norm_t = t * characteristic_V


def optimise_and_evaluate_fidelity(target_state_: Qobj):
    result = cpo.optimize_pulse_unitary(
        norm_H_d, norm_H_c,
        psi_0, target_state_,
        n_ts, norm_t,
        amp_lbound=-3, amp_ubound=3,
        # pulse_scaling=1, pulse_offset=1,
        gen_stats=True,
        alg="GRAPE",
        init_pulse_type="RND",
        max_wall_time=30, max_iter=5000, fid_err_targ=1e-3,
        log_level=qutip.logging_utils.WARN,

    )
    result.stats.report()

    final_fidelity = qutip.fidelity(target_state_, result.evo_full_final) ** 2
    print(f"Final fidelity: {final_fidelity:.5f}")

    print(f"Final gradient normal: {result.grad_norm_final:.3e}")
    print(f"Terminated due to: {result.termination_reason}")

    return final_fidelity


optimise_and_evaluate_fidelity(StandardGHZState(N).get_state_tensor(True))
optimise_and_evaluate_fidelity(AlternatingGHZState(N).get_state_tensor(True))



